import os
import sys
import numpy as np
import torch
import torch.utils.data as data
sys.path.append('pretrain')
from data.util.mypath import Path
from PIL import Image


class CATSNDOGS_Segmentation(torch.utils.data.Dataset):

    def __init__(self, root=Path.db_root_dir('CATSNDOGS'),
                 saliency='supervised_model',
                 transform=None, overfit=False, as_array = False):
        super(CATSNDOGS_Segmentation, self).__init__()

        self.root = root
        self.transform = transform
        self.as_array = as_array
        
        self.images_dir = os.path.join(self.root, 'images')
        valid_saliency = ['supervised_model', 'unsupervised_model']
        assert(saliency in valid_saliency)
        self.saliency = saliency
        self.sal_dir = os.path.join(self.root, 'saliency_' + self.saliency)
    
        self.images = []
        self.sal = []

        with open(os.path.join(self.root, 'sets/trainaug.txt'), 'r') as f:
            all_ = f.read().splitlines()

        for f in all_:
            _image = os.path.join(self.images_dir, f + ".jpg")
            _sal = os.path.join(self.sal_dir, f + ".png")
            if os.path.isfile(_image) and os.path.isfile(_sal):
                self.images.append(_image)
                self.sal.append(_sal)

        assert (len(self.images) == len(self.sal))

        if overfit:
            n_of = 32
            self.images = self.images[:n_of]
            self.sal = self.sal[:n_of]

        # Display stats
        print('Number of images: {:d}'.format(len(self.images)))

    def __getitem__(self, index):
        sample = {}

        sample['image'] = self._load_img(index)
        sample['sal'] = self._load_sal(index)

        if self.transform is not None:
            sample = self.transform(sample)
        
        if self.as_array:
            to_tensor = transforms.ToTensor()
            return to_tensor(sample['image'])
        sample['meta'] = {'image': str(self.images[index])}

        return sample

    def __len__(self):
            return len(self.images)

    def _load_img(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        return _img

    def _load_sal(self, index):
        _sal = Image.open(self.sal[index])
        return _sal

    def __str__(self):
        return 'CATSNDOGS(saliency=' + self.saliency + ')'

    def get_class_names(self):
        # Class names for sal
        return ['background', 'salient object']


if __name__ == '__main__':
    import torch
    from torch import nn
    import os
    from os import path
    import torchvision
    import torchvision.transforms as T
    from typing import Sequence
    from torchvision.transforms import functional as F
    import numbers
    import random
    import numpy as np
    from PIL import Image

    # Convert a pytorch tensor into a PIL image
    t2img = T.ToPILImage()
    # Convert a PIL image into a pytorch tensor
    img2t = T.ToTensor()
        
    
    # Sample from supervised saliency model
    dataset = CATSNDOGS_Segmentation(saliency='supervised_model')
    sample = dataset.__getitem__(1)

    print(sample)

    def trimap2f(trimap):
        return (img2t(trimap) * 255.0 - 1) / 2

    print(t2img(trimap2f(sample)))

   