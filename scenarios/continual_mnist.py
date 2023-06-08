import numpy as np
import torch.utils.data as data
from simple_deep_learning.mnist_extended import semantic_segmentation
from datasets.continual_mnist_extended import ContinualMnistExtended

class ContinualMnist:
  def __init__(self, 
               n_train,
               n_test,
               batch_size,
               tasks,
               overlap=True,
               return_im_level_label=False):
    self.n_train = n_train
    self.n_test = n_test
    self.batch_size = batch_size
    self.tasks = tasks
    self.overlap = overlap
    self.return_im_level_label = return_im_level_label

    self.curr_task_id = 0
    self.n_classes_per_task = [len(t) for _, t in self.tasks.items()]
    self._initiate_task()
    
  def _initiate_task(self):
    self.train_x, self.train_y, self.test_x, self.test_y = self._process_dataset(self.n_train, self.n_test)
    self.train_data = ContinualMnistExtended(self.train_x, self.train_y, self.tasks, curr_task_id=self.curr_task_id, return_im_level_label=self.return_im_level_label)
    self.train_stream = data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
    self.test_data = ContinualMnistExtended(self.test_x, self.test_y, self.tasks, replace=False)
    self.test_stream = data.DataLoader(self.test_data, batch_size=self.batch_size)

  def _process_dataset(self, n_train, n_test):
    # Transform groundtruth masks and add background channel
    train_x, train_y, test_x, test_y = self.create_semantic_segmentation_dataset(num_train_samples=n_train,
                                                                                 num_test_samples=n_test)
    train_y = np.concatenate([np.zeros_like(train_y[:,:,:,0:1]), train_y], axis=-1)
    train_y[:,:,:, 0] = np.sum(train_y, axis=-1) == 0
    test_y = np.concatenate([np.zeros_like(test_y[:,:,:,0:1]), test_y], axis=-1)
    test_y[:,:,:, 0] = np.sum(test_y, axis=-1) == 0
    train_y = np.argmax(train_y, axis=-1)
    test_y = np.argmax(test_y, axis=-1)
    return train_x, train_y, test_x, test_y
  
  @property
  def create_semantic_segmentation_dataset(self):
    if self.overlap:
      return lambda num_train_samples, num_test_samples : semantic_segmentation.create_semantic_segmentation_dataset(num_train_samples=num_train_samples,
                                                                                                                      num_test_samples=num_test_samples)
    else:
      classes = self.tasks[self.curr_task_id]
      return lambda num_train_samples, num_test_samples : semantic_segmentation.create_semantic_segmentation_dataset(num_train_samples=num_train_samples,
                                                                                                                      num_test_samples=num_test_samples,
                                                                                                                      classes=classes)
  def next_task(self):
    self.curr_task_id += 1
    self._initiate_task()
    


  
