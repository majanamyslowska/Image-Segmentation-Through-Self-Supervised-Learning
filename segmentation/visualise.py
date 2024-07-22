from data.dataloaders.oxoford  import Oxford
import argparse
import cv2
import os
import numpy as np

import torch

from PIL import Image
from utils.config import create_config
from utils.common_config import get_val_dataset, get_val_transformations,\
                                get_val_dataloader, get_model
from utils.evaluate_utils import save_results_to_disk, eval_segmentation_supervised_offline

# Parser
parser = argparse.ArgumentParser(description='Fully-supervised segmentation - Finetune linear layer')
parser.add_argument('--pred_path',
                    help='Path to the predictions')

args = parser.parse_args()


from utils.common_config import get_train_dataset, get_train_transformations,\
                                get_val_dataset, get_val_transformations,\
                                get_train_dataloader, get_val_dataloader

def main():
    #get list of files in directory
    Img_list = os.listdir(args.pred_path)
    os.makedirs(f"{args.pred_path}/../restored/", exist_ok=True)
    print(Img_list)
    for image_path in Img_list:

        color_map = {
            0: (0, 0, 0),  # Background (black)
            1: (0, 0, 255),  # Gray 1 to Red
            2: (255, 0, 0)   # Gray 2 to Green
        }
        # Load image
        #image = Image.open(image_path)
        # Restore colour
        restored = restore_colour(f"{args.pred_path}/{image_path}", color_map)
        
        
        Image.fromarray(restored).save(f"{args.pred_path}/../restored/{os.path.basename(image_path)}")
        
        

def restore_colour(image_path, color_map):
        #print(self.split)
        gray_image = np.array(Image.open(image_path).convert('RGB'))

        for gray_level, rgb_color in color_map.items():
            gray_image[(gray_image[:,:,0] == gray_level), :] = rgb_color

        return gray_image
        isCatImg = os.path.basename(path).split('.')[0][0].isupper()#true if cat
        #original
        o_background = [0,0,0]
        o_cat = [1,1,1]
        o_dof = [2,2,2]
        #wanted
        cat_colour= [0,0,255]
        dog_colour = [255,0,0]


        mask_b = np.all(_semseg == o_background, axis=-1)
        mask_pet = np.all(_semseg == o_pet, axis=-1)
        if isCatImg:
            _semseg[mask_pet] = cat_colour
        else:
            _semseg[mask_pet] = dog_colour

        _semseg = np.array(Image.fromarray(_semseg).convert('L'))

        return _semseg

if __name__ == "__main__":
    main()

