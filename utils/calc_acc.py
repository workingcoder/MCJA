"""MCJA/utils/calc_acc.py
   This utility file defines a function `calc_acc` for calculating classification accuracy.
"""

import torch


def calc_acc(logits, label, ignore_index=-100, mode="multiclass"):
    """
    A utility function for calculating the accuracy of model predictions given the logits and corresponding labels.
    It supports both binary and multiclass classification tasks by interpreting the logits according to the specified
    mode and comparing them against the ground truth labels to determine the number of correct predictions.
    The function also accommodates scenarios where certain examples should be ignored in the accuracy calculation,
    based on a designated ignore_index or the structure of the labels.

    Args:
    - logits (Tensor): The output logits from a model. For binary classification, logits should be a 1D tensor of
      probabilities. For multiclass classification, logits should be a 2D tensor with shape [batch_size, num_classes].
    - label (Tensor): The ground truth labels for the predictions. For multiclass classification, labels should be a
      1D tensor of class indices. For binary classification, labels should be a tensor with the same shape as logits.
    - ignore_index (int, optional): Specifies a label value that should be ignored when calculating accuracy.
      Examples with this label are not considered in the denominator of the accuracy calculation. Default is -100.
    - mode (str, optional): Determines how logits are interpreted. Can be "binary" for binary classification tasks,
      where logits are rounded to 0 or 1, or "multiclass" for tasks with more than two classes, where the class with
      the highest logit is selected. Default is "multiclass".

    Returns:
    - Tensor: The calculated accuracy as a float value, representing the proportion of correct predictions out
      of the total number of examples considered (excluding ignored examples).
    """

    if mode == "binary":
        indices = torch.round(logits).type(label.type())
    elif mode == "multiclass":
        indices = torch.max(logits, dim=1)[1]

    if label.size() == logits.size():
        ignore = 1 - torch.round(label.sum(dim=1))
        label = torch.max(label, dim=1)[1]
    else:
        ignore = torch.eq(label, ignore_index).view(-1)

    correct = torch.eq(indices, label).view(-1)
    num_correct = torch.sum(correct)
    num_examples = logits.shape[0] - ignore.sum()

    return num_correct.float() / num_examples.float()
