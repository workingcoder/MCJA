"""MCJA/configs/default/dataset.py
   It defines the default configuration settings for customized datasets.
"""

from yacs.config import CfgNode

dataset_cfg = CfgNode()

dataset_cfg.sysu = CfgNode()
dataset_cfg.sysu.num_id = 395
dataset_cfg.sysu.num_cam = 6
dataset_cfg.sysu.data_root = "../../Datasets/SYSU-MM01"

dataset_cfg.regdb = CfgNode()
dataset_cfg.regdb.num_id = 206
dataset_cfg.regdb.num_cam = 2
dataset_cfg.regdb.data_root = "../../Datasets/RegDB"
