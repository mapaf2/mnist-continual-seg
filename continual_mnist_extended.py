import sys
sys.path.append("/content/gdrive/MyDrive/Colab Notebooks/simple_deep_learning")
import torch.utils.data as data
import numpy as np
import simple_deep_learning
from simple_deep_learning.mnist_extended import semantic_segmentation

_tasks = {0: [0,1], 1: [2,3], 2: [4,5], 3: [6,7], 4: [8,9]}

class ContinualMnistDataset():
    def __init__(self, n_train = 10000, n_test = 2000):
        self.train_x, self.train_y, self.test_x, self.test_y = \
            semantic_segmentation.create_semantic_segmentation_dataset(
                num_train_samples=n_train,
                num_test_samples=n_test)
                
                
                
        self.train_dataset = continual_mnist_extended(X = self.train_x, 
                                                      y = self.train_y,
                                                      tasks = _tasks)
        self.test_dataset = continual_mnist_extended(X = self.test_x, 
                                                      y = self.test_y,
                                                      tasks = _tasks)
        

class continual_mnist_extended(data.Dataset):
  def __init__(self, X, y, tasks: dict):
    self.X = X
    self.y = y
    self.classes = np.arange(self.y.shape[-1])
    self.tasks = tasks
    self.curr_task_id = 0
    self.curr_classes = self.set_curr_classes()
    self.curr_X, self.curr_y = self.get_curr_Xy()

  def process_groundtruth(self, segmentation_masks):
    processed_masks = []
    num_classes = segmentation_masks.shape[-1]
    for i in range(len(segmentation_masks)):
      mask = np.copy(segmentation_masks[i])
      for j in range(num_classes):
        if j not in self.tasks[self.curr_task_id]:
          mask[:,:,j] = 0
      processed_masks.append(mask)

    return np.array(processed_masks)

  def set_curr_classes(self):
    curr_classes = self.tasks[self.curr_task_id]
    return curr_classes

  def next_task(self):
    self.curr_task_id += 1
    if self.curr_task_id >= len(self.tasks):
      self.curr_task_id = 0

    self.curr_classes = self.curr_classes + self.set_curr_classes()
    self.curr_X, self.curr_y = self.get_curr_Xy()

  def get_curr_Xy(self):
    processed_y = self.process_groundtruth(self.y)
    idx = np.where(np.sum(processed_y, axis=(1,2,3))!=0)[0]
    processed_X = self.X[idx]
    processed_y = processed_y[idx]
    processed_X = np.transpose(processed_X, (0,3,1,2))
    processed_y = np.transpose(processed_y, (0,3,1,2))

    return processed_X, processed_y


  def __getitem__(self, index):
    return self.curr_X[index], self.curr_y[index, self.curr_classes]

  def __len__(self):
    return len(self.curr_X)