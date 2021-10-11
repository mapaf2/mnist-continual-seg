import torch
import numpy as np
from tqdm import tqdm
import pandas as pd

def increment_task(scenario, trainer, evaluater, memory):
  scenario.next_task()
  task_id = scenario.train_data.curr_task_id
  trainer.next_task(scenario.n_classes_per_task[:task_id+1])
  optimizer = torch.optim.Adam(lr = 0.0005, params=trainer.model.parameters())
  trainer.set_optim(optimizer)
  evaluater.set_model(trainer.model)
  trainer.set_callbacks([evaluater])
  if memory is not None:
    memory.refactor_memory(scenario.n_classes_per_task[task_id])

def meta_train(n_tasks,
               epochs, 
               scenario,
               trainer,
               evaluater,
               memory=None, 
               pass_first_step=False,
               animation_path="animation"):
  
  def _print_header(scenario):
    print("*******")
    print(f"Task #{t}")
    print("*******")
    print("Classes to learn:")
    print(*[c-1 for c in scenario.train_data.curr_classes])
    print("*******")

  def _print_results():
    print()
    res = evaluater.metrics[-1].get_results()
    print("Overall stats")
    df = pd.DataFrame([res["Overall Acc"], res["Mean Acc"], res["Mean IoU"]]).T
    df.columns = ["Overall Acc", "Mean Acc", "Mean IoU"]
    print(df.to_markdown())

    print("Class IoU")
    df = pd.DataFrame(res["Class IoU"].values()).T
    df.columns = np.arange(-1,len(res["Class IoU"].values())-1)
    print(df.to_markdown())

    print("Class Acc")
    df = pd.DataFrame(res["Class Acc"].values()).T
    df.columns = np.arange(-1,len(res["Class IoU"].values())-1)
    print(df.to_markdown())

  def _print_new_task():
    print()
    print("####################################")
    print("Next Task")
    print("####################################")

  for t in range(n_tasks):
    _print_header(scenario)
    
    sample_memory = t > 0 # Only sample memory after the first task
    
    if t > 0 or not pass_first_step:
      for i in tqdm(range(epochs)):
        trainer.train(i, scenario, memory, sample_memory=sample_memory)

    _print_results()
    
    torch.save(trainer.model.state_dict(), f"checkpoints/task-{trainer.curr_task}.pth")

    if t < n_tasks - 1:
      _print_new_task()    
      increment_task(scenario = scenario, trainer = trainer, evaluater = evaluater, memory = memory)
  evaluater.create_animation(animation_path, sample_freq=4)

  