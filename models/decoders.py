import torch
from torch import nn

class simple_decoder(torch.nn.Module):
  def __init__(self, conv_filters):
    super(simple_decoder, self).__init__()
    
    self.block1 = nn.Sequential(
                     nn.Upsample(scale_factor=(2,2)),
                     nn.Conv2d(conv_filters, conv_filters, (3,3), padding=1),
                     nn.ReLU(),
                     nn.Conv2d(conv_filters, conv_filters, (3,3), padding=1),
                     nn.ReLU()
                  )
    
    self.block2 = nn.Sequential(
                     nn.Upsample(scale_factor=(2,2)),
                     nn.Conv2d(conv_filters, conv_filters, (3,3), padding=1),
                     nn.ReLU(),
                     nn.Conv2d(conv_filters, conv_filters, (3,3), padding=1),
                     nn.ReLU()
                  )

    self.decoder = nn.Sequential(self.block1,
                                 self.block2)
                                 
  def forward(self, x):
      return self.decoder(x)