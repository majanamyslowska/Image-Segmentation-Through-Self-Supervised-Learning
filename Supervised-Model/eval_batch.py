from data_loader import to_device
import torch 
from torch import nn
from network import ImageSegmentationDSC
import torchvision.transforms as T
import torchmetrics as TM
from enum import IntEnum

from metrics import IoUMetric
from data_loader import pets_test_loader


t2img = T.ToPILImage()
# Convert a PIL image into a pytorch tensor
img2t = T.ToTensor()

class TrimapClasses(IntEnum):
    PET = 0
    BACKGROUND = 1
    BORDER = 2

if __name__ == '__main__':
    # Load the pretrained weights
    model = ImageSegmentationDSC(kernel_size=3)
    model.load_state_dict(torch.load('model_weights_ADAM.pth', map_location=torch.device('cpu')))

    (test_pets_inputs, test_pets_targets) = next(iter(pets_test_loader))
    
    print(test_pets_targets.size())
    to_device(model.eval())
    predictions = model(to_device(test_pets_inputs))

    test_pets_labels = to_device(test_pets_targets)
    # print("Predictions Shape: {}".format(predictions.shape))
    pred = nn.Softmax(dim=1)(predictions)

    pred_labels = pred.argmax(dim=1)
    # Add a value 1 dimension at dim=1
    pred_labels = pred_labels.unsqueeze(1)
    pred_mask = pred_labels.to(torch.float)

    iou = to_device(TM.classification.MulticlassJaccardIndex(3, average=None))
    iou_per_class = iou(pred_mask, test_pets_labels)

    iou = to_device(TM.classification.MulticlassJaccardIndex(3, average='micro'))
    iou_accuracy = iou(pred_mask, test_pets_labels)

    print("IoU for Pet: {:.4f}".format(iou_per_class[0].item()))
    print("IoU for Background: {:.4f}".format(iou_per_class[1].item()))
    print("IoU for Border: {:.3f}".format(iou_per_class[2].item()))
    print("Average IoU:", iou_accuracy.item())

    # Training time: 1636.61 seconds for 20 epochs 
    # Ref for code https://www.kaggle.com/code/dhruv4930/oxford-iiit-pets-segmentation-using-pytorch