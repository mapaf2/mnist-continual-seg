import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Metrics:
  def __init__(self, **kwargs):
    pass

  def init_values(self):
    pass

  def get_values(self):
    pass

  def update(self, predictions, labels):
    pass

  def print(self):
    pass

  def show(self):
    pass

  def callbacks(self):
    pass

  def create_animation(self, filepath, sample_freq=1):
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
    self.save_matrices = kwargs["save_matrices"] if "save_matrices" in kwargs else False
    self.all_matrices = []
    self.init_values()
    
  def init_values(self):
    self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
    
  def get_values(self):
    return self.confusion_matrix 
    
  def save_matrix(self):
      self.all_matrices.append(np.copy(self.confusion_matrix))
        
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
    self.show()
    
  def callbacks(self):
    self.save_matrix()
        
  def show(self, normalization_axis=1):
    plt.imshow(self.confusion_matrix/(np.sum(self.confusion_matrix, axis=normalization_axis, keepdims=True)+1e-6), cmap="jet")
    plt.xlabel("Predicted")
    plt.ylabel("Groundtruth")
    plt.xticks(np.arange(11), ["bg"] + list(np.arange(10)))
    plt.yticks(np.arange(11), ["bg"] + list(np.arange(10)))
    plt.show()
    
  def create_animation(self, filepath, sample_freq=1):
    plt.ioff()
    fig, ax = plt.subplots()
    ims = []
    for i in range(0, len(self.all_matrices), sample_freq):
      title = ax.text(4.25,-0.8, f"Step {i}")
      plt.xlabel("Predicted")
      plt.ylabel("Groundtruth")
      plt.xticks(np.arange(11), ["bg"] + list(np.arange(10)))
      plt.yticks(np.arange(11), ["bg"] + list(np.arange(10)))
      im = ax.imshow(self.all_matrices[i]/(np.sum(self.all_matrices[i], axis=1, keepdims=True)+1e-6), cmap="jet", animated=True)
      if i == 0:
        ax.imshow(self.all_matrices[i]/(np.sum(self.all_matrices[i], axis=1, keepdims=True)+1e-6), cmap="jet")
        
      ims.append([im, title])
      
      
    ani = animation.ArtistAnimation(fig, ims, interval=30, blit=True,
                                repeat_delay=1000) 
    ani.save(filepath + ".mp4")
    
    
    
  def get_results(self):
    """Returns accuracy score evaluation result.
        - overall accuracy
        - mean accuracy
        - mean IU
        - fwavacc
    """
    EPS = 1e-6
    hist = self.confusion_matrix
    
    gt_sum = hist.sum(axis=1)
    mask = (gt_sum != 0)
    diag = np.diag(hist)
    
    acc = diag.sum() / hist.sum()
    #print(diag[1:].sum(), hist[1:,:].sum())
    acc_cls_c = diag / (gt_sum + EPS)
    acc_cls = np.mean(acc_cls_c[mask])
    iu = diag / (gt_sum + hist.sum(axis=0) - diag + EPS)
    mean_iu = np.mean(iu[mask])
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(self.n_classes), [iu[i] if m else "X" for i, m in enumerate(mask)]))
    cls_acc = dict(zip(range(self.n_classes), [acc_cls_c[i] if m else "X" for i, m in enumerate(mask)]))
    
    return {
            "Overall Acc": acc,
            "Mean Acc": acc_cls,
            "FreqW Acc": fwavacc,
            "Mean IoU": mean_iu,
            "Class IoU": cls_iu,
            "Class Acc": cls_acc,
        }
      
      
        
class Acc(Metrics):
  def __init__(self, **kwargs):
    self.init_values()

  def init_values(self):
    self.correct_pixels = 0
    self.total_pixels = 0

  def get_values(self):
    return np.sum(self.correct_pixels)/self.total_pixels

  def update(self, predictions, labels):
    c = ((predictions==labels) & (labels!=0)).float().cpu().numpy().sum()
    self.correct_pixels += c
    self.total_pixels += (labels!=0).sum().float().numpy()

  def print(self):
    #print(np.sum(self.correct_pixels), self.total_pixels)
    print("Foreground pixels accuracy : ", self.get_values())#np.sum(self.correct_pixels)/self.total_pixels)
    
    
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