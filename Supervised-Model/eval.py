from data_loader import to_device
import torch 
from torch import nn
from network import ImageSegmentationDSC
import torchvision.transforms as T
import torchmetrics as TM
from enum import IntEnum
from data_loader import pets_test_loader
from cat_dog_id import get_species_id
from cat_dog_id import dogs_test_loader
from cat_dog_id import cats_test_loader


class TrimapClasses(IntEnum):
    PET = 0
    BACKGROUND = 1
    BORDER = 2


if __name__ == '__main__':
    # Load the pretrained weights
    model = ImageSegmentationDSC(kernel_size=3)
    model.load_state_dict(torch.load('model_weights_ADAM.pth', map_location=torch.device('cpu')))

    splits = get_species_id()
    cat_predictions = []
    cat_labels = []

    dog_predictions = []
    dog_labels = []

    all_predictions = []
    all_labels = []

    for batch_idx, (test_pets_inputs, test_pets_targets) in enumerate(pets_test_loader):
        predictions = model(to_device(test_pets_inputs))
        pred = nn.Softmax(dim=1)(predictions)
        pred_labels = pred.argmax(dim=1).unsqueeze(1).to(torch.float)
        
        all_predictions.append(pred_labels)
        all_labels.append(test_pets_targets)
        print(batch_idx)

        if batch_idx > 10:
            break

    for batch_idx, (test_pets_inputs, test_pets_targets) in enumerate(cats_test_loader):
        predictions = model(to_device(test_pets_inputs))
        pred = nn.Softmax(dim=1)(predictions)
        pred_labels = pred.argmax(dim=1).unsqueeze(1).to(torch.float)
        
        cat_predictions.append(pred_labels)
        cat_labels.append(test_pets_targets)
        print(batch_idx)

        if batch_idx > 10:
            break

    for batch_idx, (test_pets_inputs, test_pets_targets) in enumerate(dogs_test_loader):
        predictions = model(to_device(test_pets_inputs))
        pred = nn.Softmax(dim=1)(predictions)
        pred_labels = pred.argmax(dim=1).unsqueeze(1).to(torch.float)
        
        dog_predictions.append(pred_labels)
        dog_labels.append(test_pets_targets)
        print(batch_idx)

        if batch_idx > 10:
            break



    all_cat_predictions = torch.cat(cat_predictions, dim=0)
    all_cat_labels = torch.cat(cat_labels, dim=0)

    all_dog_predictions = torch.cat(dog_predictions, dim=0)
    all_dog_labels = torch.cat(dog_labels, dim=0)

    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)


    # Calculate IoU for the entire test set
    iou = to_device(TM.classification.MulticlassJaccardIndex(3, average=None))
    iou_per_class = iou(all_predictions, all_labels)
    iou_cat = iou(all_cat_predictions, all_cat_labels)
    iou_dog = iou(all_dog_predictions, all_dog_labels)

    iou = to_device(TM.classification.MulticlassJaccardIndex(3, average='micro'))
    iou_accuracy = iou(all_predictions, all_labels)
    # ignore_index=TrimapClasses.BORDER

    # Print IoU for each class and average IoU
    print("IoU for Cat: {:.4f}".format(iou_cat[0].item()))
    print("IoU for Dog: {:.4f}".format(iou_dog[0].item()))

    # print("IoU for Pet: {:.4f}".format(iou_per_class[0].item()))
    print("IoU for Background: {:.4f}".format(iou_per_class[1].item()))
    print("IoU for Border: {:.3f}".format(iou_per_class[2].item()))
    print("Average IoU:", iou_accuracy.item())


    # Training time: 1636.61 seconds for 20 epochs 
    # Ref for code https://www.kaggle.com/code/dhruv4930/oxford-iiit-pets-segmentation-using-pytorch
    
    # IoU for Cat: 0.7707
    # IoU for Dog: 0.7240
    # IoU for Background: 0.8644
    # IoU for Border: 0.427
    # Average IoU: 0.76771080493927