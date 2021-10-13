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
                                 
                                
  def forward(self, x):
      return self.encoder(x)
      
      
      
class self_attention_cam(nn.Module):
    def __init__(self, 
                 num_class,
                 in_dim,
                 attention_dim=128,
                 linformer=False,
                 factor=8
                 ):
        super(self_attention_cam, self).__init__()
        self.num_class = num_class
        self.in_dim = in_dim
        self.attention_dim = attention_dim
        self.linformer = linformer
        self.factor = factor
        
        self.k_layer = nn.Conv2d(self.in_dim, attention_dim, 1)
        self.q_layer = nn.Conv2d(self.in_dim, attention_dim, 1)
        self.out_layer = nn.Conv2d(self.num_class, self.num_class, 1)
        self.bn = nn.BatchNorm2d(self.num_class)
        
    def forward(self, logits, hypercolumn):
        batch_size = hypercolumn.shape[0]
        h, w = hypercolumn.shape[2:]
        k = self.k_layer(hypercolumn)
        q = self.q_layer(hypercolumn)
        value = logits
        
        if self.linformer:
            reduced_h, reduced_w = (h // self.factor + 1), (w // self.factor + 1) 
            k = F.interpolate(k, (reduced_h, reduced_w), mode="bilinear", align_corners=True)
            k = torch.reshape(k,
                   [-1, reduced_h * reduced_w , self.attention_dim])
            
        attn = torch.matmul(q.reshape(batch_size, self.attention_dim, -1).permute(0,2,1), k.reshape(batch_size, self.attention_dim, -1))
        scaled_att_logits = attn / self.attention_dim**(1/2)
        att_weights = F.softmax(scaled_att_logits, dim=-1)

        if self.linformer:
            value = F.interpolate(value, ((h // self.factor + 1), (w // self.factor + 1)), mode="bilinear", align_corners=True)
        
        value = torch.reshape(value, [batch_size, value.shape[1], -1])
        value = value.permute(0,2,1)
        att_score = torch.matmul(att_weights, value)
        att_score = torch.reshape(att_score.permute(0,2,1), logits.shape)
        att_score += logits
        
        if self.linformer:
            att_score = F.interpolate(att_score, (h, w))
        
        out = self.bn(self.out_layer(att_score))
        
        return out