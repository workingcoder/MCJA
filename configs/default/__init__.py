"""MCJA/configs/default/__init__.py
   It serves as an entry point for the default configuration settings.
"""

from configs.default.dataset import dataset_cfg
from configs.default.strategy import strategy_cfg

__all__ = ["dataset_cfg", "strategy_cfg"]
