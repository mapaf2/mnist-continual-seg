import argparser
import os
import sys
sys.path.append("/content/gdrive/MyDrive/Colab Notebooks/simple_deep_learning")
from trainer import Trainer, Trainer_distillation, Trainer_MIB, Trainer_PseudoLabel, Trainer_PseudoLabel_ImageLabels
from metrics import EvaluaterCallback
from scenarios import ContinualMnist
from models import simple_seg_model
import torch
from utils import increment_task, meta_train
import contextlib
import numpy as np

_tasks = {"2-2": {0: [0,1], 1: [2,3], 2: [4,5], 3: [6,7], 4: [8,9]}}

def get_experiments_path(opts):
  path = os.path.join("experiments",
                       opts.tasks,
                       str(opts.n_train),
                       str(opts.batch_size),
                       str(opts.epochs),
                       #opts.encoder,
                       opts.method)
  if opts.distillation:
    path = os.path.join(path,
                        str(opts.lambda_distillation),
                        str(opts.encoder_level_distillation),
                        str(opts.decoder_level_distillation),
                        str(opts.output_level_distillation))
  else:
    path = os.path.join(path, "0") # no distillation
    
  return path
  
def init_trainer(opts, **kwargs):
  trainers = {"naive": Trainer,
              "distillation": Trainer_distillation,
              "mib": Trainer_MIB,
              "pseudo_label": Trainer_PseudoLabel,
              "pseudo_label_image": Trainer_PseudoLabel_ImageLabels
  }
  trainer_constructor = trainers[opts.method] 
  trainer = trainer_constructor(**kwargs)
  return trainer
  
if __name__ == '__main__':
  parser = argparser.get_argparser()
  opts = parser.parse_args()
  
  torch.manual_seed(opts.random_seed)
  torch.cuda.manual_seed(opts.random_seed)
  np.random.seed(opts.random_seed)

  opts.return_im_level_label = opts.method in ["pseudo_label_image"]
  opts.experiments_path = get_experiments_path(opts)
  os.makedirs(opts.experiments_path, exist_ok=opts.override_experiment)
  
  tasks = _tasks[opts.tasks]
  model = simple_seg_model(n_classes_per_task=[len(tasks[0])+1])
  model = model.cuda()
  
  optimizer = torch.optim.Adam(lr = 0.0005, params=model.parameters())

  continual_mnist = ContinualMnist(n_train=opts.n_train,
                                   n_test=opts.n_test,
                                   batch_size=opts.batch_size,
                                   tasks=tasks,
                                   return_im_level_label=opts.return_im_level_label)
  evaluater = EvaluaterCallback(model, ["confusion_matrix"], callback_frequency="task", n_classes=11, save_matrices=False)
  trainer = init_trainer(opts,
                         model=model,
                         n_classes=[3],
                         optim=optimizer,
                         from_new_class = 0,
                         lambda_distill=opts.lambda_distillation,
                         encoder_level_distill=opts.encoder_level_distillation,
                         decoder_level_distill=opts.decoder_level_distillation,
                         output_level_distill=opts.output_level_distillation,
                         callbacks=[evaluater])
  
  animation_path = os.path.join(opts.experiments_path, "conf_matrix_animation")
  
  file_path = os.path.join(opts.experiments_path, 'training_process.txt')
  with open(file_path, "w") as o:
    with contextlib.redirect_stdout(o):
      meta_train(n_tasks = len(tasks),
                 epochs = opts.epochs,
                 scenario = continual_mnist, 
                 trainer=trainer,
                 evaluater=evaluater,
                 memory=None,
                 animation_path=animation_path)
  