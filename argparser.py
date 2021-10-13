import argparse

def get_argparser():
  parser = argparse.ArgumentParser()

  # Hyperparameters
  parser.add_argument("--tasks", type=str, default="2-2")
  parser.add_argument("--n_train", type=int, default=1000)
  parser.add_argument("--n_test", type=int, default=2500)
  parser.add_argument("--batch_size", type=int, default=72)
  parser.add_argument("--epochs", type=int, default=200)
  parser.add_argument("--random_seed", type=int, default=42)
    
  # Methods  
  #parser.add_argument("--encoder", type=str,
  #                                 default = "simple_seg",
  #                                 choices=["simple_seg"])
  parser.add_argument("--method", type=str, choices=["naive", 
                                                     "distillation",
                                                     "mib",
                                                     "pseudo_label",
                                                     "pseudo_label_image"])
           
  # Distillation specific hyperparameters                                          
  parser.add_argument("--distillation", type=bool, default=False)    
  parser.add_argument("--lambda_distillation", type=int, default=1) 
  parser.add_argument("--encoder_level_distillation", type=bool, default=False)
  parser.add_argument("--output_level_distillation", type=bool, default=False)
  parser.add_argument("--decoder_level_distillation", type=bool, default=False) 
  
  parser.add_argument("--override_experiment", type=bool, default=False)
  
    
  return parser