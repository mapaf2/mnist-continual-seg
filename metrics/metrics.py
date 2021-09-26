import numpy as np
import torch

class Metrics:
  def __init__(self, **kwargs):
    pass

  def init_values(self):
    pass

  def update(self, predictions, labels):
    pass

  def print(self):
    pass

class IoU(Metrics):
  def __init__(self, **kwargs):
    self.init_values()

  def init_values(self):
    self.total_intersection, self.total_union = 0, 0

  def update(self, predictions, labels):
    _, intersections, unions = iou_pytorch(predictions.long(), labels.long())
    self.total_intersection += intersections.sum().numpy()
    self.total_union += unions.sum().numpy()

  def print(self):
    print("Foreground Mean IOU : ", np.sum(self.total_intersection)/np.sum(self.total_union))

class ConfusionMatrix(Metrics):
    def __init__(self, **kwargs):
        self.n_classes = kwargs["n_classes"]
        self.init_values()
    
    def init_values(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        
    def update(self, predictions, labels):
        pred_flat = predictions.flatten().numpy()
        labels_flat = labels.flatten().numpy()
        mask = (labels_flat >= 0) & (labels_flat < self.n_classes)
        hist = np.bincount(
                    self.n_classes * labels_flat[mask].astype(int) + pred_flat[mask],
                    minlength=self.n_classes ** 2,
                ).reshape(self.n_classes, self.n_classes)
        self.confusion_matrix += hist
        
    def print(self):
        print(self.confusion_matrix)
        
class Acc(Metrics):
  def __init__(self, **kwargs):
    self.init_values()

  def init_values(self):
    self.correct_pixels = 0
    self.total_pixels = 0

  def update(self, predictions, labels):
    c = ((predictions==labels) & (labels!=0)).float().cpu().numpy().sum()
    self.correct_pixels += c
    self.total_pixels += (labels!=0).sum().float().numpy()

  def print(self):
    print("Foreground pixels accuracy : ", np.sum(self.correct_pixels)/self.total_pixels)
    
    
def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    #outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    SMOOTH = 1e-6
    
    intersection = ((outputs & labels)>0).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = ((outputs | labels)>0).float().sum((1, 2))         # Will be zero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    return iou, intersection, union