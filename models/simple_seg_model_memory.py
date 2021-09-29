import torch
from torch import nn
from .backbones import simple_backbone
from .decoders import simple_decoder
class simple_seg_model_memory(torch.nn.Module):
  def __init__(self,
               conv_filters=32,
               n_classes_per_task=[2],
               input_shape=(1, 60,60),
               memory_size=100):
                   
    super(simple_seg_model, self).__init__()
    self.conv_filters = conv_filters
    self.n_classes_per_task = n_classes_per_task
    self.input_shape = input_shape
    self.memory_size = memory_size
    self.memory_X = torch.zeros((self.memory_size, input_size[0], input_size[1], input_size[2]))
    self.memory_y = torch.zeros((self.memory_size, input_shape[1], input_shape[2]))

    self.encoder = simple_backbone(conv_filters, input_shape[0])
    self.decoder = simple_decoder(conv_filters)
    
    self.cls = nn.ModuleList(
            [nn.Conv2d(self.conv_filters, c, 1) for c in n_classes_per_task]
        )

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)

    out = []
    for mod in self.cls:
        out.append(mod(x))
    x_o = torch.cat(out, dim=1)
    #x_o = torch.softmax(x_o)

    return x_o

model = simple_seg_model()