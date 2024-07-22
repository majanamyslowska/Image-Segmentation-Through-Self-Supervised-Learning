import torchvision
import torch 
import os 
import torchvision.transforms as T

data_path = "oxford_data"
pets_train_orig = torchvision.datasets.OxfordIIITPet(root=data_path, split="trainval", target_types="segmentation", download=True)
pets_test_orig = torchvision.datasets.OxfordIIITPet(root=data_path, split="test", target_types="segmentation", download=True)

def save_model_checkpoint(model, cp_name):
    torch.save(model.state_dict(), os.path.join(data_path, cp_name))

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# Send the Tensor or Model (input argument x) to the right device
def to_device(x):
    if torch.cuda.is_available():
        return x.cuda()
    else:
        return x.cpu()

# Eval model size 
def get_model_parameters(m):
    total_params = sum(
        param.numel() for param in m.parameters()
    )
    return total_params

# Print model params
def print_model_parameters(m):
    num_model_parameters = get_model_parameters(m)
    print(f"The Model has {num_model_parameters/1e6:.2f}M parameters")

# Custom augmentation class
class OxfordIIITPetsAugmented(torchvision.datasets.OxfordIIITPet):
    def __init__(
        self,
        root: str,
        split: str,
        target_types="segmentation",
        download=False,
        pre_transform=None,
        post_transform=None,
        pre_target_transform=None,
        post_target_transform=None,
        common_transform=None,
    ):
        super().__init__(
            root=root,
            split=split,
            target_types=target_types,
            download=download,
            transform=pre_transform,
            target_transform=pre_target_transform,
        )
        self.post_transform = post_transform
        self.post_target_transform = post_target_transform
        self.common_transform = common_transform

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        (input, target) = super().__getitem__(idx)

        if self.common_transform is not None:
            both = torch.cat([input, target], dim=0)
            both = self.common_transform(both)
            (input, target) = torch.split(both, 3, dim=0)
        # end if

        if self.post_transform is not None:
            input = self.post_transform(input)
        if self.post_target_transform is not None:
            target = self.post_target_transform(target)

        return (input, target)

class ToDevice(torch.nn.Module):
    """
    Sends the input object to the device specified in the
    object's constructor by calling .to(device) on the object.
    """
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, img):
        return img.to(self.device)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device={self.device})"
    
def tensor_trimap(t):
    x = t * 255
    x = x.to(torch.long)
    x = x - 1
    return x

def args_to_dict(**kwargs):
    return kwargs

transform_dict = args_to_dict(
    pre_transform=T.ToTensor(),
    pre_target_transform=T.ToTensor(),
    common_transform=T.Compose([
        ToDevice(get_device()),
        T.Resize((128, 128), interpolation=T.InterpolationMode.NEAREST),
        # Random Horizontal Flip as data augmentation.
        T.RandomHorizontalFlip(p=0.5),
    ]),
    post_transform=T.Compose([
        # Color Jitter as data augmentation.
        T.ColorJitter(contrast=0.3),
    ]),
    post_target_transform=T.Compose([
        T.Lambda(tensor_trimap),
    ]),
)

pets_train = OxfordIIITPetsAugmented(
        root=data_path,
        split="trainval",
        target_types="segmentation",
        download=False,
        **transform_dict,
    )

pets_test = OxfordIIITPetsAugmented(
        root=data_path,
        split="test",
        target_types="segmentation",
        download=False,
        **transform_dict,
    )

pets_train_loader = torch.utils.data.DataLoader(
        pets_train,
        batch_size=64,
        shuffle=True,
    )

pets_test_loader = torch.utils.data.DataLoader(
        pets_test,
        batch_size=16,
        shuffle=True,
    )

if __name__ == '__main__':
    # Validation: Check if CUDA is available
    print(f"CUDA: {torch.cuda.is_available()}")
    print(pets_train_orig)
    print(pets_test_orig)


    (train_pets_inputs, train_pets_targets) = next(iter(pets_train_loader))
    (test_pets_inputs, test_pets_targets) = next(iter(pets_test_loader))
    print(train_pets_inputs.shape, train_pets_targets.shape)
    print(test_pets_inputs.shape, test_pets_targets.shape)