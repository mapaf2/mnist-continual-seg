import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from loss import UnbiasedKnowledgeDistillationLoss, UnbiasedCrossEntropy

class Trainer:
  def __init__(self,
               model,
               n_classes,
               optim,
               curr_task=0,
               callbacks=[]):
    self.model = model
    self.criterion = nn.CrossEntropyLoss()
    self.n_classes = n_classes
    self.optim = optim
    self.curr_task = curr_task
    self.callbacks = callbacks

  def train(self, 
            cur_epoch,
            scenario,
            memory=None,
            sample_memory = False):
    """Train for 1 epoch."""
    epoch_loss = 0
    for cur_step, (images, labels) in enumerate(scenario.train_stream):
      images, labels = images.cuda().float(), labels.cuda().long()
      images_mem, labels_mem = self._enrich_with_memory(images, labels, memory, sample_memory)
      epoch_loss += self._train_step(images_mem, labels_mem)
      if memory is not None:
        memory.process_batch(images, labels)
      self._apply_callbacks(scenario, freq="step")
    self._apply_callbacks(scenario, freq="epoch")
    return epoch_loss / len(scenario.train_stream)

  def _enrich_with_memory(self,
                          images,
                          labels,
                          memory=None,
                          sample_memory=False):
    if memory is None or sample_memory==False:
      return images, labels
    else:
      new_images, new_labels = memory.sample_batch()
      new_images, new_labels = new_images.float().cuda(), new_labels.long().cuda()
      all_images = torch.cat([images, new_images])
      all_labels = torch.cat([labels, new_labels])
      return all_images, all_labels

  def _apply_callbacks(self, scenario, freq):
    for c in self.callbacks:
      c.callbacks(scenario, freq=freq)

  def _train_step(self, images, labels):
    """Perform 1 training iteration."""
    self.optim.zero_grad()
    outputs = self.model(images)
    loss = self._compute_loss(outputs, labels)

    loss.backward()
    self.optim.step()
    
    return loss

  def next_task(self, n_classes_per_task):
    """Switch to next task."""
    self.n_classes_per_task = n_classes_per_task
    new_model = self._load_new_model(self.n_classes_per_task)
    self.model = new_model.cuda()
    self.curr_task += 1

  def _load_new_model(self, n_classes_per_task):
    """
    Helper function to create new model and
     load weights of past training.
    """
    new_model = self._make_model(n_classes_per_task)
    path_weights = self._get_path_weights()
    step_checkpoint = torch.load(path_weights, map_location="cpu")
    new_model.load_state_dict(step_checkpoint, strict=False)
    return new_model

  def _make_model(self, n_classes_per_task):
    """Helper function to create new model."""
    m_constructor = type(self.model)
    new_model = m_constructor(n_classes_per_task=n_classes_per_task)
    return new_model

  def _get_path_weights(self):
    """Helper function to get path to previous model's weights."""
    path = f"checkpoints/task-{self.curr_task}.pth"
    return path

  def _compute_loss(self, outputs, labels):
    return self.criterion(outputs, labels)

  def set_optim(self, optim):
    self.optim = optim

  def set_callbacks(self, callbacks):
    self.callbacks = callbacks
    
    
class Trainer_MIB(Trainer):
  def __init__(self,
               new_model,
               n_classes,
               optim,
               from_new_class,
               old_model=None,
               lambda_distill=0,
               output_level_distill=False,
               encoder_level_distill=False,
               decoder_level_distill=False,
               curr_task=0,
               callbacks=[]):
                   
    super(Trainer_MIB, self).__init__(new_model, n_classes, optim, curr_task, callbacks)
    self.from_new_class = from_new_class
    self.old_model = old_model
    self.lambda_distill = lambda_distill
    self.output_level_distill = output_level_distill
    self.encoder_level_distill = encoder_level_distill
    self.decoder_level_distill = decoder_level_distill
    
  def _train_step(self, images, labels):
    if self.old_model is not None and self.lambda_distill != 0:
        return self._train_step_distillation(images, labels)
    else:
        return super(Trainer_MIB, self)._train_step(images, labels)
        
  def _train_step_distillation(self, images, labels):
    assert self.old_model is not None
    
    self.optim.zero_grad()
    
    new_enc, new_dec, new_outputs = self.model(images, return_intermediate=True)
    old_enc, old_dec, old_outputs = self.old_model(images, return_intermediate=True)
    
    ce_loss = self._compute_loss(new_outputs, labels)
    d_loss = self._distillation_loss(new_enc,
                                     old_enc,
                                     new_dec,
                                     old_dec,
                                     new_outputs,
                                     old_outputs,
                                     labels)
    loss = ce_loss + self.lambda_distill * d_loss
    loss.backward()
    self.optim.step()
    
    return ce_loss
    
  
    
  def _compute_loss(self, outputs, labels):
    if self.old_model is not None:
      #print(output.shape, labels.shape, self.from_new_class+1)
      un_ce = UnbiasedCrossEntropy(old_cl=self.from_new_class+1)
      #print(self.from_new_class-1, un_ce(outputs, labels), nn.CrossEntropyLoss()(outputs, labels))
      return un_ce(outputs, labels)
    else:
      return super(Trainer_MIB, self)._compute_loss(outputs, labels)
  def _distillation_loss(self,
                         new_enc,
                         old_enc,
                         new_dec,
                         old_dec,
                         new_outputs,
                         old_outputs,
                         labels):
    
    distill_loss = 0
    
    if self.output_level_distill:
      ukd_loss = UnbiasedKnowledgeDistillationLoss()
      distill_loss += ukd_loss(new_outputs, old_outputs)
    
    return distill_loss
    
  def next_task(self, n_classes_per_task):
    """Switch to next task."""
    self.from_new_class += n_classes_per_task[-1]
    self.n_classes_per_task = n_classes_per_task
    
    self.old_model = copy.deepcopy(self.model)
    new_model = self._load_new_model(self.n_classes_per_task)
    self.model = new_model.cuda()
    self.curr_task += 1
    
    
    
    
    
class Trainer_distillation(Trainer):
  def __init__(self,
               new_model,
               n_classes,
               optim,
               from_new_class,
               old_model=None,
               lambda_distill=0,
               output_level_distill=False,
               encoder_level_distill=False,
               decoder_level_distill=False,
               curr_task=0,
               callbacks=[]):
                   
    super(Trainer_distillation, self).__init__(new_model, n_classes, optim, curr_task, callbacks)
    self.from_new_class = from_new_class
    self.old_model = old_model
    self.lambda_distill = lambda_distill
    self.output_level_distill = output_level_distill
    self.encoder_level_distill = encoder_level_distill
    self.decoder_level_distill = decoder_level_distill
    
  def _train_step(self, images, labels):
    if self.old_model is not None and self.lambda_distill != 0:
        return self._train_step_distillation(images, labels)
    else:
        return super(Trainer_distillation, self)._train_step(images, labels)
        
  def _train_step_distillation(self, images, labels):
    assert self.old_model is not None
    
    self.optim.zero_grad()
    
    new_enc, new_dec, new_outputs = self.model(images, return_intermediate=True)
    old_enc, old_dec, old_outputs = self.old_model(images, return_intermediate=True)
    
    ce_loss = self._compute_loss(new_outputs, labels)
    d_loss = self._distillation_loss(new_enc,
                                     old_enc,
                                     new_dec,
                                     old_dec,
                                     new_outputs,
                                     old_outputs,
                                     labels)
    loss = ce_loss + self.lambda_distill * d_loss
    loss.backward()
    self.optim.step()
    
    return ce_loss
    
  def _distillation_loss(self,
                         new_enc,
                         old_enc,
                         new_dec,
                         old_dec,
                         new_outputs,
                         old_outputs,
                         labels):
    
    distill_loss = 0
    
    if self.output_level_distill:
        distill_loss += self._softXEnt(new_outputs[:, :old_outputs.shape[1]], old_outputs)
    if self.encoder_level_distill:
        distill_loss += self._l2_feature_loss(new_enc, old_enc)
    if self.decoder_level_distill:
        l = self._l2_feature_loss(new_dec, old_dec)
        distill_loss += 0.1*l
    
    return distill_loss
    
  def _l2_feature_loss(self, new_feats, old_feats):
    L2_LOSS = nn.MSELoss()
    feats_loss = L2_LOSS(new_feats, old_feats)
    return feats_loss
    
  def next_task(self, n_classes_per_task):
    """Switch to next task."""
    self.from_new_class += self.n_classes_per_task[-1]
    self.n_classes_per_task = n_classes_per_task
    
    self.old_model = copy.deepcopy(self.model)
    new_model = self._load_new_model(self.n_classes_per_task)
    self.model = new_model.cuda()
    self.curr_task += 1
    
  def _softXEnt (self, input, target):
    logprobs = torch.nn.functional.log_softmax (input, dim = 1)[:,1:]
    target = torch.nn.functional.softmax (target, dim = 1)[:,1:]
    return  -(target * logprobs).sum() / (input.shape[0] * input.shape[2] * input.shape[3])
    
    
    
    
    
class GeneticTrainer(Trainer):
  def __init__(self,
               model,
               n_classes,
               optim,
               curr_task=0,
               callbacks=[]):
    super(GeneticTrainer, self).__init__(model, n_classes, optim, curr_task, callbacks)
    
  def _train_step(self, images, labels):
    features, outputs = self.model(images)
    loss = self._compute_loss(outputs, labels)
    
    cls_in_images = torch.unique(labels)
    #pos_features = features[]
    
  #def crossovers