import torch
from torch import nn
from .backbones import simple_backbone
from .decoders import simple_decoder
class simple_seg_model(torch.nn.Module):
  def __init__(self,
               conv_filters=32,
               n_classes_per_task=[2],
               input_shape=(1, 60,60)):
                   
    super(simple_seg_model, self).__init__()
    self.conv_filters = conv_filters
    self.n_classes_per_task = n_classes_per_task
    self.input_shape = input_shape

    self.encoder = simple_backbone(conv_filters, input_shape[0])
    self.decoder = simple_decoder(conv_filters)
    
    #self.cls = nn.ModuleList(
    #        [nn.Conv2d(self.conv_filters, c, 1) for c in n_classes_per_task]
    #        )
    self.cls = nn.Conv2d(self.conv_filters, sum(n_classes_per_task), 1) 

  def forward(self, x, return_intermediate=False, return_blocks=False):
    if return_blocks:
      x_block2, x_enc = self.encoder(x, return_blocks=True)
    else:
      x_enc = self.encoder(x)
    x_dec = self.decoder(x_enc)

    #out = []
    #for mod in self.cls:
    #    out.append(mod(x_dec))
    #x_o = torch.cat(out, dim=1)
    x_o = self.cls(x_dec)
    if return_blocks:
      return x_block2, x_enc, _o
    if return_intermediate:
      return x_enc, x_dec, x_o

    return x_o
