import torch
from torch import nn
import numpy as np

class memory:
  def __init__(self,
               images_shape,
               masks_shape,
               n_classes,
               batch_size,
               memory_size):
                   
    self.images_shape = images_shape
    self.masks_shape = masks_shape
    self.n_classes = n_classes
    self.past_n_classes = 0
    self.batch_size = batch_size
    self.memory_size = memory_size
    self.slots_per_class = self.memory_size//self.n_classes
    self.memory_images = torch.zeros((self.n_classes,
                                        self.slots_per_class,
                                        self.images_shape[0],
                                        self.images_shape[1],
                                        self.images_shape[2]))
                                        
    self.memory_masks = torch.zeros((self.n_classes,
                                        self.slots_per_class,
                                       self.masks_shape[0],
                                       self.masks_shape[1]))
                                        
    self.empty_slots = [0]*self.n_classes
                                        
  def save_memory(self, image, mask, label):
    slot_class = self.empty_slots[label] 
    self.memory_images[label, slot_class] = torch.clone(image)
    self.memory_masks[label, slot_class] = torch.clone(mask)
    self._increment_empty_slots(label)
    
  def read_memory(self, label, i):
    assert i < self.slots_per_class
    return self.memory_images[label, i], self.memory_masks[label, i]
    
  def sample_batch(self, old_only=True, batch_size=None):
    if batch_size is None:
      batch_size = self.batch_size
      
    sample_n_class = self.past_n_classes if old_only else self.n_classes
    n_per_class = batch_size//sample_n_class
    
    images, masks = [], []
    for n in range(sample_n_class):
      idx = np.random.randint(0, self.slots_per_class, n_per_class)
      for i in idx:
        img, m = self.read_memory(label=n, i = i)
        images.append(img)
        masks.append(m)
    images = torch.cat(images, dim=0).reshape(n_per_class*sample_n_class,
                                              self.images_shape[0],
                                              self.images_shape[1],
                                              self.images_shape[2])
    masks = torch.cat(masks, dim=0).reshape(n_per_class*sample_n_class,
                                            self.masks_shape[0],
                                            self.masks_shape[1])
    return images, masks
    
  def _increment_empty_slots(self, label):
    if self.empty_slots[label] + 1 == self.slots_per_class:
      self.empty_slots[label] = 0
    else:
      self.empty_slots[label] += 1
      
  def refactor_memory(self, n_new_classes):
    new_n_classes = self.n_classes + n_new_classes
    new_slots_per_class = self.memory_size//new_n_classes
    
    new_memory_images = torch.zeros((new_n_classes,
                                        new_slots_per_class,
                                        self.images_shape[0],
                                        self.images_shape[1],
                                        self.images_shape[2]))
                                        
    new_memory_masks = torch.zeros((new_n_classes,
                                        new_slots_per_class,
                                       self.masks_shape[0],
                                       self.masks_shape[1]))
                                       
    new_memory_images[:self.n_classes] = self.memory_images[:, :new_slots_per_class]
    new_memory_masks[:self.n_classes] = self.memory_masks[:, :new_slots_per_class]
    new_empty_slots = [0]*new_n_classes
    
    self.memory_images = new_memory_images
    self.memory_masks = new_memory_masks
    self.past_n_classes = self.n_classes
    self.n_classes = new_n_classes
    self.slots_per_class = new_slots_per_class
    self.empty_slots = new_empty_slots