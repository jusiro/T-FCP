import torch
import os
import clip
from tqdm.auto import tqdm
from datetime import datetime
import json
from time import time

import numpy as np

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32

# Set seeds for reproducibility
from utils.misc import set_seeds
set_seeds(42, use_cuda=device == 'cuda')

# Pats and configs
from data.configs.tasks_batch import *
from local_data.constants import *

from modeling.adapter.parallel.model import Linear

from utils.metrics import accuracy, evaluate_conformal, cov_perc_satisfaction
from utils.split import random_split, balance_split
from conformal.utils import get_prediction_sets, calculate_conformal_value

import torch.profiler

def process(args):

    # Create folder to save features.
    results = {}
    results["avg"] = {}
    path_results = os.path.join(PATH_RESULTS, args.backbone, args.procedure)
    if not os.path.exists(path_results):
        os.makedirs(path_results)
    exp_path = os.path.join(path_results, "solver_{solver}_K_{K}_alphaICP_{alphaICP}_alphaFCP_{alphaFCP}_date_{time}".format(solver=args.solver, K=str(args.K), alphaICP=str((args.alpha_icp) * 100), alphaFCP=str((args.alpha_fcp) * 100), time=datetime.now().strftime("%m%d%H%M")))

    # Set input paths
    path_vision_features = os.path.join(PATH_VISION_FEATURES, args.backbone)
    path_text_features = os.path.join(PATH_TEXT_PROTOTYPES, args.backbone)
    path_labels = PATH_LABELS
    # Loop over all datasets.
    for task in args.tasks:
        print("Processing: [{dataset}]".format(dataset=task))
        results[task] = {"avg": {}, "per_seed": {}}
        
        # Loop over seeds.
        acc_seeds, cov_seeds, mean_size_seeds, median_size_seeds, singletons_seeds, empties_seeds, ccv_seeds, singleton_cov_seeds, times_seeds, pGPU_seeds = [], [], [], [], [], [], [], [], [], []
        for seed in tqdm(range(args.seeds), leave=True, desc="  {:10s}".format(task)):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Load data
            if args.split == "train-test":
                features_train = torch.load(os.path.join(path_vision_features, task + "_train.pt"))
                labels_train   = torch.load(os.path.join(path_labels, task + "_train.pt"))
            features_test      = torch.load(os.path.join(path_vision_features, task + "_test.pt"))
            labels_test        = torch.load(os.path.join(path_labels, task + "_test.pt"))
            textual_prototypes = torch.load(os.path.join(path_text_features, task + ".pt"))

            # Get dimensionality
            F = features_test.shape[-1]
            C = len(np.unique(labels_test))
        
            # Sample calibration set as in split conformal prediction
            pC = np.bincount(labels_test) / len(labels_test) # Label-marginal distribution.
            if args.split == "train-test": # Retrieve few shots from adaptation data.
                features_cal, labels_cal, _, _ = balance_split(features_train, labels_train, N=args.K*C, p=pC, seed=seed)
                N = len(labels_cal) 
                # Split calibration set into adaptation and calibration using 50% of data.
                features_cal, labels_cal, features_adapt, labels_adapt = balance_split(features_cal, labels_cal, N//2, p=pC, seed=seed)
            elif args.split == "test":     # Split test data - as standard in CP literature - to avoid exchangeability violations.
                features_cal, labels_cal, features_test, labels_test = balance_split(features_test, labels_test, N=(args.K*C)//2, p=pC, seed=seed)
                features_adapt, labels_adapt, features_test, labels_test = balance_split(features_test, labels_test, N=(args.K*C)//2, p=pC, seed=seed)

            # Combine both adaptation and calibration data for full conformal procedure
            features_cal = torch.cat([features_adapt, features_cal], dim=0)
            labels_cal = torch.cat([labels_adapt, labels_cal], dim=0)
            N = len(labels_cal) # Re-compute N, in case there was a drift during targeted sampling.

            # Limit test samples to speed-up experimentaiton
            if args.limit_test and len(labels_test) > (16*C):
                features_test, labels_test, _, _ = balance_split(features_test, labels_test, N=16*C, p=pC, seed=seed)
            M = features_test.size(0)

            # Move to device feature tensors and convert to float 16 for GPU efficiency.
            features_cal = features_cal.to(device=device, dtype=dtype)
            features_test = features_test.to(device=device, dtype=dtype)
            textual_prototypes = textual_prototypes.to(device=device, dtype=dtype)

            # Set linear classifier.
            head = Linear(F=F, C=C, W0=textual_prototypes).to(device)

            # Stage 1 - run split conformal prediction:
            if args.procedure == "tfcp":
                # Compute zero-shot probabilities.
                with torch.no_grad():
                    probs_cal = torch.softmax(head.forward_batched(features_cal), dim=-1)
                    probs_test = torch.softmax(head.forward_batched(features_test), dim=-1)

                # Compute non-conformity score on calibration data.
                lac = 1 - probs_cal[0, torch.arange(N), labels_cal]

                # Search conformal quantile.
                t1 = calculate_conformal_value(lac.to(torch.float32), args.alpha_icp)

                # Produce conformal sets on test data.
                sets_test = get_prediction_sets(1 - probs_test[0], t1)
            else:
                sets_test = [list(range(C)) for i in range(M)]

            # Stage 2 - run full conformal procedure:

            # Init classifier weights.
            if args.solver == 'gd':
                head.fit_gd(features_cal, labels_cal)
            elif args.solver == 'lda':
                head.fit_lda(features_cal, labels_cal)
            else:
                print("Solver not implemented...")
                break

            # Pre-compute accuracy results to compute accuracy.
            with torch.no_grad():
                probs_test = torch.softmax(head.forward_batched(features_test), dim=-1)

            # Evaluate discriminative performance.
            acc = accuracy(labels_test.numpy(), probs_test[0,:,:].detach().cpu().numpy())            

            # Run full conformal procedure
           
            # Loop over test samples
            times_samples = []
            for i in tqdm(range(M), leave=True, desc="  {:10s} - seed {:2s}/{:2s}".format(task, str(seed+1), str(args.seeds))):    
                if len(sets_test[i]) > 0:
                    # Select candidate Y
                    Y_candidate = torch.tensor(sets_test[i])

                    # Track pipeline time per sample
                    t_init_sample=time()

                    sets_test_i, idx, max_c = [], 0, 100
                    while idx <= len(Y_candidate):

                        # Adapt the model using the joint calibration and test sample for each candidate label
                        if args.solver == 'gd':
                            with torch.amp.autocast(device_type=device, dtype=torch.float16):
                                online_weights = head.fit_gd(features_cal, labels_cal, x_online=features_test[i, :].unsqueeze(0), Y_candidate=Y_candidate[idx:idx+max_c])
                        elif args.solver == 'lda':
                            with torch.amp.autocast(device_type=device, dtype=torch.float32):
                                online_weights = head.update_lda_from_inverse(features_test[i, :].unsqueeze(0), Y_candidate[idx:idx+max_c])
                               
                        # Compute probabilities with new class weights
                        with torch.no_grad():
                            with torch.amp.autocast(device_type=device, dtype=torch.float16):
                                probs_cal = torch.softmax(head.forward_linear_batched(features_cal, online_weights, bs=8192*2), dim=-1)
                                probs_test = torch.softmax(head.forward_linear_batched(features_test[i, :].unsqueeze(0), online_weights, bs=8192*2), dim=-1)

                                # Non-conformity scores and quantile search
                                t = calculate_conformal_value((1 - probs_cal[:, torch.arange(N, device=device), labels_cal]).t(), args.alpha_fcp)

                                # Selection
                                mask = (1 - probs_test[torch.arange(probs_test.size(0), device=device), 0, Y_candidate[idx:idx+max_c]]) <= t
                                selected = torch.nonzero(mask, as_tuple=False).squeeze(1).tolist()
                            sets_test_i.extend(Y_candidate[idx:idx+max_c].cpu().numpy()[selected].tolist())
                            idx += max_c
                    sets_test[i] = sets_test_i

                    # Track pipeline time and store
                    t_end_sample=time()
                    times_samples.append((t_end_sample-t_init_sample))
                
            # Evaluate conformal sets
            cov, mean_size, median_size, singletons, empties, ccv, singleton_cov = evaluate_conformal(sets_test, labels_test, alpha=(args.alpha_icp+args.alpha_fcp)) 
            # Store metrics
            acc_seeds.append(acc), cov_seeds.append(cov), mean_size_seeds.append(mean_size), median_size_seeds.append(median_size), singletons_seeds.append(singletons), singleton_cov_seeds.append(singleton_cov), empties_seeds.append(empties), ccv_seeds.append(ccv), times_seeds.append(np.mean(times_samples)), pGPU_seeds.append(torch.cuda.max_memory_allocated() / 1e6)

        # Compute and store average metrics
        results[task]["avg"]["acc"]                    = np.round(np.mean(acc_seeds), 2).item()
        results[task]["avg"]["cov-avg(%)"]             = np.round(np.mean(cov_seeds) * 100, 2).item()
        results[task]["avg"]["cov-med(%)"]             = np.round(np.median(cov_seeds) * 100, 2).item()
        results[task]["avg"]["cov-2sigma(%)"]          = np.round(2 * np.std(cov_seeds) * 100, 2).item()
        results[task]["avg"]["cov_satisfaction(%)"]    = np.round(cov_perc_satisfaction(cov_seeds, alpha=(args.alpha_icp+args.alpha_fcp), tol=0.005) * 100, 2).item()
        results[task]["avg"]["mean_size"]              = np.round(np.mean(mean_size_seeds), 2).item()
        results[task]["avg"]["median_size"]            = np.round(np.mean(median_size_seeds), 2).item()
        results[task]["avg"]["singleton(%)"]           = np.round(np.mean(singletons_seeds) * 100, 2).item()
        results[task]["avg"]["empties(%)"]             = np.round(np.mean(empties_seeds) * 100, 2).item()
        results[task]["avg"]["ccv(%)"]                 = np.round(np.mean(ccv_seeds), 2).item()
        results[task]["avg"]["singletons_cov(%)"]      = np.round(np.nanmean(singleton_cov_seeds) * 100, 2).item()
        results[task]["avg"]["samples_per_second"]     = np.round(1/np.mean(times_seeds), 2).item()
        results[task]["avg"]["pGPU[MB]"]               = np.round(np.mean(pGPU_seeds), 2).item()
        print(results[task]["avg"])
        results[task]["per_seed"]["cov"]            = cov_seeds

        # Store ongoing experiment results
        with open(exp_path, "w") as file: 
            json.dump(results, file, indent=1)

    # Average performance for all datasets
    print("Processing: [{dataset}]".format(dataset="AVERAGE"))
    for ikey in list(results[task]["avg"].keys()):
        results["avg"][ikey] = np.round(np.mean([results[task]["avg"][ikey] for task in args.tasks]), 2).item()
    print(results["avg"])

    # Store experiment results
    with open(exp_path, "w") as file: 
        json.dump(results, file, indent=1)