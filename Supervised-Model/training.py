from torch import nn
import torch
import os
import time
from metrics import IoULoss
from network import ImageSegmentationDSC
from data_loader import to_device
from data_loader import pets_train_loader
from data_loader import pets_test_loader
from data_loader import print_model_parameters


def train_model(model, loader, optimizer):
    to_device(model.train())
    cel = True
    if cel:
        criterion = nn.CrossEntropyLoss(reduction='mean')
    else:
        criterion = IoULoss(softmax=True)
    # end if

    running_loss = 0.0
    running_samples = 0

    for batch_idx, (inputs, targets) in enumerate(loader, 0):
        optimizer.zero_grad()
        inputs = to_device(inputs)
        targets = to_device(targets)
        outputs = model(inputs)

        # The ground truth labels have a channel dimension (NCHW).
       
        if cel:
            targets = targets.squeeze(dim=1)
        # end if
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_samples += targets.size(0)
        running_loss += loss.item()
    # end for

    print("Trained {} samples, Loss: {:.4f}".format(
        running_samples,
        running_loss / (batch_idx+1),
    ))



def train_loop(model, loader, epochs, optimizer, scheduler, save_path):
    epoch_i, epoch_j = epochs
    for i in range(epoch_i, epoch_j):
        epoch = i
        print(f"Epoch: {i:02d}, Learning Rate: {optimizer.param_groups[0]['lr']}")
        train_model(model, loader, optimizer)

        if scheduler is not None:
            scheduler.step()
        # end if
        print("")

        # Save model weights
        torch.save(model.state_dict(), os.path.join(save_path, f'model_weights_epoch_{epoch}.pth'))


if __name__ == '__main__':
    
    (train_pets_inputs, train_pets_targets) = next(iter(pets_train_loader))
    (test_pets_inputs, test_pets_targets) = next(iter(pets_test_loader))

    mdsc = ImageSegmentationDSC(kernel_size=3)
    mdsc.eval()
    to_device(mdsc)
    mdsc(to_device(train_pets_inputs)).shape
    print_model_parameters(mdsc)
    to_device(mdsc)

    optimizer2 = torch.optim.Adam(mdsc.parameters(), lr=0.001)
    sgd = torch.optim.SGD(mdsc.parameters(), lr=0.004, momentum=0.9)
    scheduler2 = None

    # Train the model that uses depthwise separable convolutions.
    working_dir =  os.getcwd()
    save_path2 = os.path.join(working_dir, "segnet_weights") # Name to save to
    t0 = time.time()
    train_loop(mdsc, pets_train_loader, (1, 21), optimizer2, scheduler2, save_path2) # Change to SGD
    t1 = time.time()
    print("Training time: {:.2f} seconds".format(t1-t0))