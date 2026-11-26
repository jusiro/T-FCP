# General imports
import torch
import os
import clip
import json

from tqdm.auto import tqdm
from datetime import datetime
from time import time

import numpy as np

# Project imports
from data.configs.tasks_batch import *
from local_data.constants import *
from modeling.adapter.base.model import Linear
from utils.metrics import accuracy, evaluate_conformal, cov_perc_satisfaction
from utils.split import random_split, balance_split
from conformal.utils import get_prediction_sets, calculate_conformal_value

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32

# Set seeds for reproducibility
from utils.misc import set_seeds
set_seeds(42, use_cuda=device == 'cuda')


def process(args):

    # Create folder to save features.
    results = {}
    results["avg"] = {}
    path_results = os.path.join(PATH_RESULTS, args.backbone, args.procedure)
    if not os.path.exists(path_results):
        os.makedirs(path_results)
    exp_path = os.path.join(path_results, "K_{K}_alphaICP_{alphaICP}_date_{time}".format(K=str(args.K), alphaICP=str((args.alpha_icp) * 100), time=datetime.now().strftime("%m%d%H%M")))

    # Set input paths
    path_vision_features = os.path.join(PATH_VISION_FEATURES, args.backbone)
    path_text_features = os.path.join(PATH_TEXT_PROTOTYPES, args.backbone)
    path_labels = PATH_LABELS

    # Loop over all datasets.
    for task in args.tasks:
        print("Processing: [{dataset}]".format(dataset=task))
        results[task] = {"avg": {}, "per_seed": {}}
        
        # Loop over seeds.
        acc_seeds, cov_seeds, mean_size_seeds, median_size_seeds, singletons_seeds, empties_seeds, ccv_seeds, singleton_cov_seeds = [], [], [], [], [], [], [], []
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
            Ncal = len(labels_cal) # Re-compute N, in case there was a drift during targeted sampling.

            # Limit test samples to speed-up experimentation
            if args.limit_test and len(labels_test) > (16*C):
                features_test, labels_test, _, _ = balance_split(features_test, labels_test, N=16*C, p=pC, seed=seed)
            M = features_test.size(0)

            # Move to device feature tensors and convert to float 16 for GPU efficiency.
            features_cal = features_cal.to(device=device, dtype=dtype)
            features_test = features_test.to(device=device, dtype=dtype)
            textual_prototypes = textual_prototypes.to(device=device, dtype=dtype)

            # Set linear classifier.
            head = Linear(F=F, C=C, W0=textual_prototypes).to(device)

            # Compute zero-shot probabilities.
            with torch.no_grad():
                probs_cal = torch.softmax(head.forward_batched(features_cal), dim=-1)
                probs_test = torch.softmax(head.forward_batched(features_test), dim=-1)

            # Compute LAC non-conformity score on calibration data.
            lac = 1 - probs_cal[torch.arange((Ncal)), labels_cal]

            # Search conformal quantile.
            t = calculate_conformal_value(lac, args.alpha_icp)

            # Produce conformal sets on test data
            sets_test = get_prediction_sets(1 - probs_test, t)

            # Evaluate conformal sets and discriminative metrics
            acc = accuracy(labels_test.numpy(), probs_test.detach().cpu().numpy())
            cov, mean_size, median_size, singletons, empties, ccv, singleton_cov = evaluate_conformal(sets_test, labels_test, alpha=args.alpha_icp) 

            # Store metrics
            acc_seeds.append(acc), cov_seeds.append(cov), mean_size_seeds.append(mean_size), median_size_seeds.append(median_size), empties_seeds.append(empties), singletons_seeds.append(singletons), ccv_seeds.append(ccv), singleton_cov_seeds.append(singleton_cov)

        # Compute and store average metrics
        results[task]["avg"]["acc"]                 = np.round(np.mean(acc_seeds), 2).item()
        results[task]["avg"]["cov-avg(%)"]          = np.round(np.mean(cov_seeds) * 100, 2).item()
        results[task]["avg"]["cov-med(%)"]          = np.round(np.median(cov_seeds) * 100, 2).item()
        results[task]["avg"]["cov-2sigma(%)"]       = np.round(2 * np.std(cov_seeds) * 100, 2).item()
        results[task]["avg"]["cov_satisfaction(%)"] = np.round(cov_perc_satisfaction(cov_seeds, alpha=args.alpha_icp, tol=0.005) * 100, 2).item()
        results[task]["avg"]["mean_size"]           = np.round(np.mean(mean_size_seeds), 2).item()
        results[task]["avg"]["median_size"]         = np.round(np.mean(median_size_seeds), 2).item()
        results[task]["avg"]["singleton(%)"]        = np.round(np.mean(singletons_seeds) * 100, 2).item()
        results[task]["avg"]["empties(%)"]          = np.round(np.mean(empties_seeds) * 100, 2).item()
        results[task]["avg"]["ccv(%)"]              = np.round(np.mean(ccv_seeds), 2).item()
        results[task]["avg"]["singletons_cov(%)"]   = np.round(np.nanmean(singleton_cov_seeds) * 100, 2).item()
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