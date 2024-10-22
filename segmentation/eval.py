#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import argparse
import cv2
import os
import torch

from utils.config import create_config
from utils.common_config import get_val_dataset, get_val_transformations,\
                                get_val_dataloader, get_model
from utils.evaluate_utils import save_results_to_disk, eval_segmentation_supervised_offline


# Parser
parser = argparse.ArgumentParser(description='Evaluate segmantion model')
parser.add_argument('--config_env',
                    help='Config file for the environment')
parser.add_argument('--config_exp',
                    help='Config file for the experiment')
parser.add_argument('--crf-postprocess', action='store_true',
                    help='Apply CRF post-processing during evaluation')
parser.add_argument('--state-dict', 
                    help='Model state dict to test')
args = parser.parse_args()

def main():
    cv2.setNumThreads(1)

    # Retrieve config file
    p = create_config(args.config_env, args.config_exp)
    print('Python script is {}'.format(os.path.abspath(__file__)))
    print(p)

    # Get model
    print('Retrieve model')
    model = get_model(p)
    print(model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # CUDNN
    print('Set CuDNN benchmark') 
    torch.backends.cudnn.benchmark = True

    # Dataset
    print('Retrieve dataset')
    
    # Transforms 
    val_transforms = get_val_transformations(p)
    val_dataset = get_val_dataset(p, val_transforms)
    true_val_dataset = get_val_dataset(p, None) # True validation dataset without reshape 
    val_dataloader = get_val_dataloader(p, val_dataset)
    print('Val samples %d' %(len(val_dataset)))

    # Evaluate best model at the end
    print('Evaluating model at {}'.format(args.state_dict))
    model.load_state_dict(torch.load(args.state_dict, map_location='cpu'))
    save_results_to_disk(p, val_dataloader, model, crf_postprocess=args.crf_postprocess)
    eval_stats = eval_segmentation_supervised_offline(p, true_val_dataset, verbose=True)

if __name__ == "__main__":
    main()
