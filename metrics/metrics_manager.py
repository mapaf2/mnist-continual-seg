from metrics import IoU, Acc, ConfusionMatrix

class MetricsManager:
  __implemented_metrics = ["iou", "acc", "confusion_matrix"]

  def __init__(self, metrics, **kwargs):
    self.metrics = [self._convert_metrics(m, **kwargs) for m in metrics]
    self.init_values()

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
    
  def callbacks(self):
    for m in self.metrics:
      m.callbacks()     