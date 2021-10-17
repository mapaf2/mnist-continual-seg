import torch
from torch import nn
import torch.nn.functional as F

class simple_backbone(torch.nn.Module):
    
  def __init__(self, conv_filters, input_dim):
    super(simple_backbone, self).__init__()
        
    self.block1 = nn.Sequential(
                     nn.Conv2d(input_dim, conv_filters, (3,3), padding=1),
                     nn.ReLU(),
                     nn.Conv2d(conv_filters, conv_filters, (3,3), padding=1),
                     nn.ReLU(),
                     nn.MaxPool2d((2,2))
                  )
    self.block2 = nn.Sequential(
                     nn.Conv2d(conv_filters, conv_filters, (3,3), padding=1),
                     nn.ReLU(),
                     nn.Conv2d(conv_filters, conv_filters, (3,3), padding=1),
                     nn.ReLU(),
                     nn.MaxPool2d((2,2))
                  )
    self.block3 = nn.Sequential(
                     nn.Conv2d(conv_filters, conv_filters, (3,3), padding=1),
                     nn.ReLU(),
                     nn.Conv2d(conv_filters, conv_filters, (3,3), padding=1),
                     nn.ReLU()
                  )
    
    self.encoder = nn.Sequential(self.block1,
                                 self.block2,
                                 self.block3)
                                 
                                
  def forward(self, x, return_blocks=False):
    if return_blocks:
      x = self.block1(x)
      x = self.block2(x)
      xx = self.block3(x)
      return x, xx
    else:
      return self.encoder(x)
      
      
      
