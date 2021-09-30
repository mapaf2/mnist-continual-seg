import numpy as np
import simple_deep_learning
import torch.utils.data as data
from simple_deep_learning.mnist_extended import semantic_segmentation
from datasets.continual_mnist_extended import ContinualMnistExtended

class ContinualMnist:
  def __init__(self, 
               n_train,
               n_test,
               batch_size,
               tasks):
  
    self.tasks = tasks
    self.n_classes_per_task = [len(t) for _, t in self.tasks.items()]
    self.n_classes_per_task[0] +=1 # Include background
    self.batch_size = batch_size
    self.train_x, self.train_y, self.test_x, self.test_y = self._process_dataset(n_train, n_test)
    self.train_data = ContinualMnistExtended(self.train_x, self.train_y, self.tasks)
    self.train_stream = data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
    self.test_data = ContinualMnistExtended(self.test_x, self.test_y, self.tasks, replace=False)
    self.test_stream = data.DataLoader(self.test_data, batch_size=self.batch_size)

  def _process_dataset(self, n_train, n_test):
    # Transform groundtruth masks and add background channel
    train_x, train_y, test_x, test_y = semantic_segmentation.create_semantic_segmentation_dataset(num_train_samples=n_train,num_test_samples=n_test)
    train_y = np.concatenate([np.zeros_like(train_y[:,:,:,0:1]), train_y], axis=-1)
    train_y[:,:,:, 0] = np.sum(train_y, axis=-1) == 0
    test_y = np.concatenate([np.zeros_like(test_y[:,:,:,0:1]), test_y], axis=-1)
    test_y[:,:,:, 0] = np.sum(test_y, axis=-1) == 0
    train_y = np.argmax(train_y, axis=-1)
    test_y = np.argmax(test_y, axis=-1)
    return train_x, train_y, test_x, test_y


  def next_task(self):
    self.train_data.next_task()
    self.train_stream = data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
    self.test_data.next_task()
    self.test_stream = data.DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)


  
