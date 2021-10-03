import torch
from .metrics import *

class EvaluaterCallback:
  __implemented_metrics = ["iou", "acc", "confusion_matrix"]

  def __init__(self, model, metrics, callback_frequency, **kwargs):
    self.model = model
    self.metrics = [self._convert_metrics(m, **kwargs) for m in metrics]
    self.callback_frequency = callback_frequency
    self.init_values()
    self.kwargs = kwargs

  def _convert_metrics(self, m, **kwargs):
    assert m in self.__implemented_metrics, f"Invalid metric, choose from {self.__implemented_metrics}"
    _c_m_dict = {"iou": IoU(**kwargs), 
                "acc": Acc(**kwargs),
                "confusion_matrix": ConfusionMatrix(**kwargs)
    }
    return _c_m_dict[m]

  def init_values(self):
    for m in self.metrics:
      m.init_values()     

  def test(self, test_loader, verbose=False):
    self.model.eval()
    self.init_values()
    with torch.no_grad():
      for images, labels in test_loader:
        images, labels = images.cuda().float(), labels.float()
        predictions = self.model(images)
        _, class_predictions = torch.max(predictions, dim=1)
        class_predictions = class_predictions.cpu().long()
        self.update(class_predictions, labels)
    
    if verbose:
      self.print_metrics()

  def callbacks(self, scenario, freq, verbose=False):
    if freq == self.callback_frequency:
      self.test(scenario.test_stream, verbose)
      for m in self.metrics:
        m.callbacks() 

  def create_animation(self, filepath, sample_freq=1):
    for m in self.metrics:
      m.create_animation(filepath, sample_freq)
  
  def update(self, predictions, labels):
    for m in self.metrics:
      m.update(predictions, labels)

  def print_metrics(self):
    for m in self.metrics:
      m.print()
      
  def show_metrics(self):
    for m in self.metrics:
      m.show()
      
  def get_metrics(self):
    results = []
    for m in self.metrics:
      results.append(m.get_values())
    return results    

  def set_model(self, model):
    self.model = model