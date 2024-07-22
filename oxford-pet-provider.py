import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt


class DuplicatedCompose(transforms.Compose):
    """
    Duplicates transformations for pairs of source and target images
    """
    def __init__(self, transforms):
        super().__init__(self)
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image = t(image)
            target = t(target)
        
        return image, target

def processed_oxford_iit_pet(split: str):
    """
    Split is either "train" or "test"
    """
    transformations = DuplicatedCompose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        # transforms.Grayscale()
    ])

    split = "train" if split == "trainval" else split
    return torchvision.datasets.OxfordIIITPet(root="./data", 
                                              target_types="segmentation",
                                              transforms=transformations,
                                              download=True,
                                              split=split)

if __name__ == "__main__":
    dataset = processed_oxford_iit_pet(split="test")

    figure = plt.figure(figsize=(6, 4))
    samples = 4
    for i in range(1, samples):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        source, target = dataset[sample_idx]
        
        # source
        figure.add_subplot(2, samples, i)        
        plt.axis("off")
        plt.imshow(source.squeeze(), cmap="gray")
        
        # target
        figure.add_subplot(2, samples, i + samples)        
        plt.axis("off")
        plt.imshow(target.squeeze(), cmap="gray")
    
    plt.show()

    # loader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=2)
