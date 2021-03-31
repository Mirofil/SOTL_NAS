from yacs.config import CfgNode as CN

C = CN()


C.model_type="max_deg"
C.a_optim = "SGD"
C.w_optim = "SGD" 
C.epochs =20 
C.batch_size=64
C.steps_per_epoch=1 
C.dataset="fourier" 
C.dry_run=True 
C.grad_outer_loop_order=-1 
C.grad_inner_loop_order=-1 
C.mode="bilevel" 
C.device="cuda" 
C.ihvp="exact" 
C.inv_hess="exact" 
C.hvp="exact"  
C.rand_seed =1 
C.initial_degree = 1
C.arch_train_data ="sotl" 
C.optimizer_mode="autograd" 
C.T=25
C.recurrent =True 
C.w_lr=1e-2 
C.a_lr=1e-2
C.a_weight_decay=0.1
C.w_weight_decay=0.001
C.adaptive_a_lr=True
C.grad_clip = None

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return C.clone()

cfg = C