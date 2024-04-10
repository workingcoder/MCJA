"""MCJA/utils/tools.py
   This utility file provides some helper functions.
"""

import random
import datetime
import numpy as np
import torch


def set_seed(seed, cuda=True):
    """
    A utility function for setting the random seed across various libraries commonly used in deep learning projects to
    ensure reproducibility of results. By fixing the random seed, this function makes experiments deterministic, meaning
    that running the same code with the same inputs and the same seed (on the same experimental platform) will produce
    the same outputs every time, which is crucial for debugging and comparing different models and configurations.

    Args:
    - seed (int): The random seed value to be set across all libraries.
    - cuda (bool, optional): A flag indicating whether to apply the seed to CUDA operations as well. Default is True.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def time_str(fmt=None):
    """
    A utility function for generating a formatted string representing the current date and time. This function is
    particularly useful for creating timestamps for logging, file naming, or any other task that requires capturing
    the exact moment when an event occurs. By default, the function produces a string formatted as "YYYY-MM-DD_hh-mm-ss",
    but it allows for customization of the format according to the user's needs.

    Args:
    - fmt (str, optional): A format string defining how the date and time should be represented.
    This string should follow the formatting rules used by Python's `strftime` method. If no format is specified, the
    default format "%Y-%m-%d_%H-%M-%S" is used, which corresponds to the "year-month-day_hour-minute-second" format.

    Returns:
    - str: A string representation of the current date and time,
      formatted according to the provided or default format specification.
    """

    if fmt is None:
        fmt = '%Y-%m-%d_%H-%M-%S'
    return datetime.datetime.today().strftime(fmt)
