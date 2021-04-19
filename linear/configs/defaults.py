from yacs.config import CfgNode as CN

C = CN()

C.epochs = 50
C.steps_per_epoch=None
C.batch_size = 64
C.n_features = 18
C.n_samples = None
C.w_optim='SGD'
C.a_optim='SGD'
C.w_decay_order=2
C.w_lr = 1e-2
C.w_momentum=0.0
C.w_weight_decay=0.001
C.a_decay_order=2
C.a_lr = 1e-4
C.a_momentum = 0.0
C.a_weight_decay = 0.
C.T = 10
C.grad_clip = None
C.grad_clip_bilevel = 50
C.logging_freq = 200
C.w_checkpoint_freq = 1
C.n_informative=7
C.noise=0.25
C.featurize_type="fourier"
C.initial_degree=1
C.hvp="exact"
C.ihvp="exact"
C.inv_hess="exact"
C.arch_train_data="sotl"
C.normalize_a_lr=False
C.log_grad_norm=True
C.log_alphas=True
C.w_warm_start=0
C.alpha_weight_decay=0.
C.grad_inner_loop_order=-1
C.grad_outer_loop_order=-1
C.model_type="max_deg"
C.dataset="fourier"
C.device= 'cuda'
C.train_arch=True
C.dry_run=False
C.hinge_loss=0.25
C.mode = "bilevel"
C.hessian_tracking=False
C.smoke_test:bool = False
C.rand_seed = None
C.a_scheduler:str = 'step'
C.w_scheduler:str = 'step'
C.decay_scheduler:str=None
C.loss:str = None
C.optimizer_mode="manual"
C.bilevel_w_steps=None
C.debug=False
C.recurrent=True
C.l=1e5
C.adaptive_a_lr=False
C.alpha_lr = None
C.softplus_alpha_lr=True
C.softplus_beta=100
C.arch_update_frequency=1
C.loss_threshold=None
C.features=None
C.log_suffix=""
C.val_split = 0.01
C.alpha_lr_reject_strategy="half" # half or zero
C.sotl_agg = "sum"

def cfg_defaults():
    return C.clone()

cfg = C