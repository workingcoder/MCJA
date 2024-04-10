"""MCJA/configs/default/strategy.py
   It outlines the default strategy configurations for the framework.
"""

from yacs.config import CfgNode

strategy_cfg = CfgNode()

strategy_cfg.prefix = "SYSU"

# Settings for Data
strategy_cfg.dataset = "sysu"
strategy_cfg.image_size = (384, 128)
strategy_cfg.sample_method = "norm_triplet"
strategy_cfg.p_size = 8
strategy_cfg.k_size = 16
strategy_cfg.batch_size = 128

# Settings for Augmentation
strategy_cfg.random_flip = True
strategy_cfg.random_crop = True
strategy_cfg.random_erase = True
strategy_cfg.color_jitter = True
strategy_cfg.padding = 10
strategy_cfg.vimc_wg = True
strategy_cfg.vimc_cc = True
strategy_cfg.vimc_sj = True

# Settings for Model
strategy_cfg.drop_last_stride = False
strategy_cfg.mda_ratio = 2
strategy_cfg.mda_m = 2

# Setting for Loss
strategy_cfg.loss_id = True
strategy_cfg.loss_cmr = True

# Settings for Training
strategy_cfg.lr = 0.00035
strategy_cfg.wd = 0.0005
strategy_cfg.num_epoch = 140
strategy_cfg.lr_step = [80, 120]
strategy_cfg.fp16 = True

# Settings for Logging
strategy_cfg.start_eval = 0
strategy_cfg.log_period = 10
strategy_cfg.eval_interval = 1

# Settings for Testing
strategy_cfg.resume = ''
strategy_cfg.mser = True
strategy_cfg.test_only = False
