import torch
from torch import nn
class simple_seg_model(torch.nn.Module):
  def __init__(self, conv_filters=32, n_classes_per_task=[2]):
    super(simple_seg_model, self).__init__()
    self.conv_filters = conv_filters
    self.n_classes_per_task = n_classes_per_task

    self.block1 = nn.Sequential(
                     nn.Conv2d(1, self.conv_filters, (3,3), padding=1),
                     nn.ReLU(),
                     nn.Conv2d(self.conv_filters, self.conv_filters, (3,3), padding=1),
                     nn.ReLU(),
                     nn.MaxPool2d((2,2))
                  )
    self.block2 = nn.Sequential(
                     nn.Conv2d(self.conv_filters, self.conv_filters, (3,3), padding=1),
                     nn.ReLU(),
                     nn.Conv2d(self.conv_filters, self.conv_filters, (3,3), padding=1),
                     nn.ReLU(),
                     nn.MaxPool2d((2,2))
                  )
    self.block3 = nn.Sequential(
                     nn.Conv2d(self.conv_filters, self.conv_filters, (3,3), padding=1),
                     nn.ReLU(),
                     nn.Conv2d(self.conv_filters, self.conv_filters, (3,3), padding=1),
                     nn.ReLU()
                  )
    
    self.encoder = nn.Sequential(self.block1,
                                 self.block2,
                                 self.block3)

    self.block4 = nn.Sequential(
                     nn.Upsample(scale_factor=(2,2)),
                     nn.Conv2d(self.conv_filters, self.conv_filters, (3,3), padding=1),
                     nn.ReLU(),
                     nn.Conv2d(self.conv_filters, self.conv_filters, (3,3), padding=1),
                     nn.ReLU()
                  )
    
    self.block5 = nn.Sequential(
                     nn.Upsample(scale_factor=(2,2)),
                     nn.Conv2d(self.conv_filters, self.conv_filters, (3,3), padding=1),
                     nn.ReLU(),
                     nn.Conv2d(self.conv_filters, self.conv_filters, (3,3), padding=1),
                     nn.ReLU()
                  )

    self.decoder = nn.Sequential(self.block4,
                                 self.block5)
    
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