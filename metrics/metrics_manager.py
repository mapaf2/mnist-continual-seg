from metrics import IoU, Acc

class MetricsManager:
  __implemented_metrics = ["iou", "acc"]

  def __init__(self, metrics):
    self.metrics = [self._convert_metrics(m) for m in metrics]
    self.init_values()

  def _convert_metrics(self, m):
    assert m in self.__implemented_metrics, f"Invalid metric, choose from {self.__implemented_metrics}"
    _c_m_dict = {"iou": IoU(), 
                 "acc": Acc()}
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