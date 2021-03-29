from yacs.config import CfgNode as CN

C = CN()

C.epochs = 75
C.steps_per_epoch=None
C.batch_size = 128
C.n_features = 18
C.n_samples = 5000
C.w_optim='SGD'
C.w_decay_order=2
C.w_lr = 1e-3
C.w_momentum=0.0
C.w_weight_decay=1e-4
C.a_optim="SGD"
C.a_decay_order=2
C.a_lr = 1e-2
C.a_momentum = 0.0
C.a_weight_decay = 0.
C.T = 2
C.grad_clip = 10
C.grad_clip_bilevel=1000
C.logging_freq = 200
C.w_checkpoint_freq = 1
C.n_informative=7
C.noise=0.25
C.featurize_type="fourier"
C.initial_degree=1
C.hvp="exact"
C.ihvp ="exact"
C.inv_hess="exact"
C.normalize_a_lr=True
C.w_warm_start=0
C.log_grad_norm=True
C.log_alphas=True
C.extra_weight_decay=0.
C.grad_inner_loop_order=-1
C.grad_outer_loop_order=-1
C.arch_train_data="sotl"
C.model_type="MLP2"
C.dataset="MNIST"
C.device = 'cuda'
C.train_arch=True
C.dry_run=False
C.mode="bilevel"
C.hessian_tracking=False
C.smoke_test=True
C.rand_seed = 1
C.decay_scheduler=None
C.w_scheduler=None
C.a_scheduler=None
C.features=None
C.loss='ce'
C.log_suffix = ""
C.optimizer_mode = "autograd"
C.bilevel_w_steps=None
C.debug=False
C.recurrent=True
C.rand_seed=1
C.adaptive_a_lr = False
C.softplus_alpha_lr = False
C.alpha_lr=1e-3
C.softplus_beta=1
C.arch_update_frequency=1
C.loss_threshold=None
C.log_suffix=""

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return C.clone()

cfg = C