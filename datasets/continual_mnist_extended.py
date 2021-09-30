import torch.utils.data as data
import numpy as np

class ContinualMnistExtended(data.Dataset):
  def __init__(self, X, y, tasks: dict, replace=True):
    self.X = X
    self.y = y
    self.classes = np.arange(self.y.shape[-1])
    self.tasks = tasks
    self.replace = replace
    self.curr_task_id = 0
    self.curr_classes = [0] + [t+1 for t in self.tasks[self.curr_task_id]]
    self.curr_X, self.curr_y = self.get_curr_Xy()
    self.seen_classes = []

  def __getitem__(self, index):
    return self.curr_X[index], self.curr_y[index]

  def __len__(self):
    return len(self.curr_X)

  def get_curr_Xy(self):
    """
    Process masks for the current task and return
     images, segmentation masks.
    """
    processed_y = self.process_groundtruth(self.y)
    idx = np.where(np.sum(processed_y, axis=(1,2))!=0)[0]
    processed_X = self.X[idx]
    processed_y = processed_y[idx]
    processed_X = np.transpose(processed_X, (0,3,1,2))

    return processed_X, processed_y

  def process_groundtruth(self, segmentation_masks):
    """
    Remove groundtruth mask annotations of classes
     that are not from the current set of classes.
    """
    processed_masks = np.copy(segmentation_masks)
    processed_masks[np.isin(processed_masks,
                            self.curr_classes, invert=True)] = 0 
    return processed_masks

  def next_task(self):
    """
    Switch to next task. If the task was the last one, 
     go back to first task.
    """
    self.seen_classes = list(set(self.seen_classes + self.curr_classes))

    self.curr_task_id += 1
    if self.curr_task_id >= len(self.tasks):
      self.curr_task_id = 0

    self.set_curr_classes()
    self.curr_X, self.curr_y = self.get_curr_Xy()

  def set_curr_classes(self):
    """Adjust current classes to current task and return them. """
    if self.replace:
      self.curr_classes = [0] + [t+1 for t in self.tasks[self.curr_task_id]]
    else:
      self.curr_classes = self.curr_classes + [t+1 for t in self.tasks[self.curr_task_id]]
    return self.curr_classes