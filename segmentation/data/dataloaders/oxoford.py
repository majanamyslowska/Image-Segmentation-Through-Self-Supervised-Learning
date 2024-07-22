#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os
import sys
import errno
import cv2
import hashlib
import glob
import tarfile

import numpy as np
import torch.utils.data as data
import torch
from PIL import Image

from data.util.mypath import Path
from data.util.google_drive import download_file_from_google_drive
from utils.utils import mkdir_if_missing


class Oxford(data.Dataset):
    
   
    CATEGORY_NAMES = ['background',
                          'cat', 'dog']

    def __init__(self, root=Path.db_root_dir('oxford-iiit-pet'),
                 split='val', transform=None, download=False, ignore_classes=[]):
        # Set paths
        self.root = root
        valid_splits = ['trainaug', 'train', 'val']
        assert(split in valid_splits)
        self.split = split
         
        if split == 'trainaug':
            _semseg_dir = os.path.join(self.root, 'SegmentationClass')
        else:
            _semseg_dir = os.path.join(self.root, 'SegmentationClass')

        _image_dir = os.path.join(self.root, 'images')


        # Download
        if download:
            self._download()

        # Transform
        self.transform = transform

        # Splits are pre-cut
        print("Initializing dataloader for oxford-iiit-pet {} set".format(''.join(self.split)))
        split_file = os.path.join(self.root, 'sets', self.split + '.txt')
        self.images = []
        self.semsegs = []
        
        with open(split_file, "r") as f:
            lines = f.read().splitlines()

        for ii, line in enumerate(lines):
            # Images
            _image = os.path.join(_image_dir, line + ".jpg")
            assert os.path.isfile(_image)
            self.images.append(_image)

            # Semantic Segmentation
            _semseg = os.path.join(_semseg_dir, line + '.png')
            assert os.path.isfile(_semseg)
            self.semsegs.append(_semseg)

        assert(len(self.images) == len(self.semsegs))

        # Display stats
        print('Number of dataset images: {:d}'.format(len(self.images)))

        # List of classes which are remapped to ignore index.
        # This option is used for comparing with other works that consider only a subset of the pascal classes.
        self.ignore_classes = [self.CATEGORY_NAMES.index(class_name) for class_name in ignore_classes]

    def __getitem__(self, index):
        sample = {}

        # Load image
        _img = self._load_img(index)
        sample['image'] = _img

        # Load pixel-level annotations
        _semseg = self._load_semseg(index)
        if _semseg.shape != _img.shape[:2]:
            _semseg = cv2.resize(_semseg, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        sample['semseg'] = _semseg
	
        sample['meta'] = {'im_size': (_img.shape[0], _img.shape[1]),
                          'image_file': self.images[index],
                          'image': os.path.basename(self.semsegs[index]).split('.')[0]}
            
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.images)

    def _load_img(self, index):
        _img = np.array(Image.open(self.images[index]).convert('RGB'))
        return _img

    def _load_semseg(self, index):
        _semseg = np.array(Image.open(self.semsegs[index]))
        isCatImg = os.path.basename(self.semsegs[index]).split('.')[0][0].isupper()#true if cat
        #original
        o_background = 2
        o_border = 3
        o_pet = 1
        #wanted
        background = 0#first class
        border = 255
        cat_colour= 1#second class
        dog_colour = 2#third class
        
        # Create a mask where the colour condition is True
        if(self.split == "trainaug" or self.split == "val"):
            _semseg[_semseg == o_background] = background
            if isCatImg:
                _semseg[_semseg == o_border] = border
                _semseg[_semseg == o_pet] = cat_colour
            else:
                _semseg[_semseg == o_border] = border
                _semseg[_semseg == o_pet] = dog_colour

        # pil_image = Image.fromarray(_semseg)
        # Save the image
        # save_path = f"/cs/student/msc/dsml/2023/mdavudov/ADL/MaskContrast/segmentation/data/{self.split}/{os.path.basename(self.semsegs[index])}"  # Adjust the filename as needed
        # pil_image.save(save_path)    
        for ignore_class in self.ignore_classes:
            _semseg[_semseg == ignore_class] = 255
        return _semseg
    
    def _load_semseg_rgb(self, index):
        #print(self.split)
        _semseg = np.array(Image.open(self.semsegs[index]).convert('RGB'))
        
        #print(_semseg)
        # print(_semseg.shape)
        # print(os.path.basename(self.semsegs[index]))
        isCatImg = os.path.basename(self.semsegs[index]).split('.')[0][0].isupper()#true if cat
        #original
        o_background = [2,2,2]
        o_border = [3,3,3]
        o_pet = [1,1,1]
        #wanted
        background = [0,0,0]
        border_w = [255,255,255]#white
        border_g = [224,224,192]#grey
        cat_colour= [0,0,255]
        dog_colour = [255,0,0]
        
        # Create a mask where the colour condition is True
        if(self.split == "trainaug"):
            mask_b = np.all(_semseg == o_background, axis=-1)
            mask_pet = np.all(_semseg == o_pet, axis=-1)
            mask_bord = np.all(_semseg == o_border, axis=-1)
            _semseg[mask_b] = background
            _semseg[mask_bord] = border_w
            _semseg[mask_pet] = background#black and white image no fill
        else:#class fill
            mask_b = np.all(_semseg == o_background, axis=-1)
            mask_pet = np.all(_semseg == o_pet, axis=-1)
            mask_bord = np.all(_semseg == o_border, axis=-1)
            _semseg[mask_b] = background
            if isCatImg:
                _semseg[mask_pet] = cat_colour
            else:
                _semseg[mask_pet] = dog_colour
            _semseg[mask_bord] = border_g
        
        for ignore_class in self.ignore_classes:
            ig_color = [ignore_class, ignore_class, ignore_class]
            mask_ig = np.all(_semseg == ig_color, axis=-1)
            _semseg[mask_ig] = border_w
        
        # if self.split == "trainaug":
        _semseg = np.array(Image.fromarray(_semseg).convert('L'))
        
        pil_image = Image.fromarray(_semseg)

        # Save the image
        save_path = f"/cs/student/msc/dsml/2023/mdavudov/ADL/MaskContrast/segmentation/data/{self.split}/{os.path.basename(self.semsegs[index])}"  # Adjust the filename as needed
        pil_image.save(save_path)    
        # for ignore_class in self.ignore_classes:
        #     _semseg[_semseg == ignore_class] = 255
        return _semseg

    def get_img_size(self, idx=0):
        img = Image.open(os.path.join(self.root, 'JPEGImages', self.images[idx] + '.jpg'))
        return list(reversed(img.size))

    def __str__(self):
        return 'OXFORD(split=' + str(self.split) + ')' 

    def get_class_names(self):
        return self.CATEGORY_NAMES

    def _download(self):
        _fpath = os.path.join(Path.db_root_dir(), self.FILE)

        if os.path.isfile(_fpath):
            print('Files already downloaded')
            return
        else:
            print('Downloading dataset from google drive')
            mkdir_if_missing(os.path.dirname(_fpath))
            download_file_from_google_drive(self.GOOGLE_DRIVE_ID, _fpath)

        # extract file
        cwd = os.getcwd()
        print('\nExtracting tar file')
        tar = tarfile.open(_fpath)
        os.chdir(Path.db_root_dir())
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('Done!')



