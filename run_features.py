
import argparse
import torch
import os
import clip

import numpy as np

from data.utils import set_loader
from modeling.vlms.vision import extract_vision_features
from modeling.vlms.text import extract_text_prototypes

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set seeds for reproducibility
from utils.misc import set_seeds
set_seeds(42, use_cuda=device == 'cuda')

# Pats and configs
from data.configs.tasks_batch import *
from modeling.vlms.configs.constants import *
from local_data.constants import *

def process(args):

    # Create folder to save features.
    path_vision_features = os.path.join(PATH_VISION_FEATURES, args.backbone)
    if not os.path.exists(path_vision_features):
        os.makedirs(path_vision_features)
    if not os.path.exists(PATH_LABELS):
        os.makedirs(PATH_LABELS)
    path_text_features = os.path.join(PATH_TEXT_PROTOTYPES, args.backbone)
    if not os.path.exists(path_text_features):
        os.makedirs(path_text_features)

    # Load CLIP Model
    if args.backbone.split("-")[0] == "CLIP":
        model_clip, transforms = clip.load(vlms_loadid_from_name[args.backbone])
        model_clip.to(device).float()
        model_clip.eval()  # Disable dropout and batch-norm statistics update
    elif args.backbone.split("-")[0] == "MetaCLIP":
        from transformers import AutoProcessor, AutoModel
        transforms = AutoProcessor.from_pretrained(vlms_loadid_from_name[args.backbone]).image_processor
        model_clip = AutoModel.from_pretrained(vlms_loadid_from_name[args.backbone])
        model_clip.to(device).float()
        model_clip.eval() # Disable dropout and batch-norm statistics update
    else:
        print("Architecture not suported...")
        return

    # Loop over all datasets.
    for task in args.tasks:
        
        # Testing data loader & feature extraction.
        print("  Feature extraction on: [Te-{dataset}]".format(dataset=task))
        test_loader = set_loader(task, partition="test", transforms=transforms, batch_size=args.bs)
        features, labels = extract_vision_features(model_clip, test_loader, clip_id=args.backbone)

        # Set zero-shot prototypes
        textual_prototypes, _ = extract_text_prototypes(model_clip, test_loader.dataset.classnames, test_loader.dataset.templates, clip_id=args.backbone)

        # Store testing data
        torch.save(torch.tensor(features).to(torch.float16), os.path.join(path_vision_features, task + "_test.pt"))
        torch.save(torch.tensor(labels).to(torch.int), os.path.join(PATH_LABELS, task + "_test" + ".pt"))
        torch.save(textual_prototypes.to(torch.float16).cpu(), os.path.join(path_text_features, task + ".pt"))

def main():
    parser = argparse.ArgumentParser()

    # Tasks
    parser.add_argument('--tasks', default=TASKS_FG)

    # Model to employ
    parser.add_argument('--backbone', default='CLIP-ViTB16',
     help='"CLIP-ViTB32", "CLIP-ViTB16", "CLIP-ViTL14", "MetaCLIP-ViTB16", "MetaCLIP-ViTL14", "MetaCLIP-ViTH14"'
          )
    # Other hyperparameters
    parser.add_argument('--bs', default=128, type=int, help='Batch size')

    args, unknown = parser.parse_known_args()

    process(args=args)


if __name__ == "__main__":
    main()