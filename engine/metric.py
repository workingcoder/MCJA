"""MCJA/engine/metric.py
   It provides a flexible mechanism for aggregating and computing metrics of cross-modality person re-identification.
"""

from collections import defaultdict

import torch
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric, Accuracy


class ScalarMetric(Metric):
    """
    A simple, generic implementation of an Ignite Metric for aggregating scalar values over iterations or epochs. This
    class provides a framework for tracking and computing the average of any scalar metric (e.g., loss, accuracy) during
    the training or evaluation process of a machine learning model. It accumulates the sum of the scalar values and the
    count of instances (batches) it has seen, allowing for the calculation of average scalar metric over all instances.

    Methods:
    - update(value): Adds a new scalar value to the running sum and increments the instance count.
      This method is called at each iteration with the scalar metric value for that iteration.
    - reset(): Resets the running sum and instance count to zero.
      Typically called at the start of each epoch or evaluation run to prepare for new calculations.
    - compute(): Calculates and returns the average of all scalar values added since the last reset.
      If no instances have been added, it raises a NotComputableError, indicating that there is not enough data.
    """

    def update(self, value):
        self.sum_metric += value
        self.sum_inst += 1

    def reset(self):
        self.sum_inst = 0
        self.sum_metric = 0

    def compute(self):
        if self.sum_inst == 0:
            raise NotComputableError('Accuracy must have at least one example before it can be computed')
        return self.sum_metric / self.sum_inst


class IgnoreAccuracy(Accuracy):
    """
    An extension of the Ignite Accuracy metric that incorporates the ability to ignore certain target labels during the
    computation of accuracy. This class is particularly useful in scenarios where some target labels in the dataset
    should not contribute to the accuracy calculation, such as padding tokens in sequence models or background classes
    in segmentation tasks. By specifying an ignore index, instances with this target label are excluded from both the
    numerator and denominator of the accuracy calculation.

    Args:
    - ignore_index (int): The target label that should be ignored in the accuracy computation. Instances with this
      label are not considered correct or incorrect predictions, effectively being excluded from the metric.

    Methods:
    - reset(): Resets the internal counters for correct predictions and total examples,
      preparing the metric for a new set of calculations.
    - update(output): Processes a batch of predictions and targets,
      updating the internal counters by counting correct predictions that do not correspond to the ignore index.
    - compute(): Calculates and returns the accuracy over all batches processed since the last reset,
      excluding instances with the ignore index from the calculation.
    """

    def __init__(self, ignore_index=-1):
        super(IgnoreAccuracy, self).__init__()

        self.ignore_index = ignore_index

    def reset(self):
        self._num_correct = 0
        self._num_examples = 0

    def update(self, output):

        y_pred, y = self._check_shape(output)
        self._check_type((y_pred, y))

        if self._type == "binary":
            indices = torch.round(y_pred).type(y.type())
        elif self._type == "multiclass":
            indices = torch.max(y_pred, dim=1)[1]

        correct = torch.eq(indices, y).view(-1)
        ignore = torch.eq(y, self.ignore_index).view(-1)
        self._num_correct += torch.sum(correct).item()
        self._num_examples += correct.shape[0] - ignore.sum().item()

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('Accuracy must have at least one example before it can be computed')
        return self._num_correct / self._num_examples


class AutoKVMetric(Metric):
    """
    A flexible metric class in the Ignite framework that computes and stores key-value (KV) pair metrics for each
    output of a model during training or evaluation. The AutoKVMetric class is designed to handle outputs in the
    form of dictionaries where each key corresponds to a specific metric name, and its value represents the metric
    value for that batch. This class allows for the automatic aggregation of multiple metrics over all batches,
    providing a convenient way to track a variety of performance indicators within a single metric class.

    Methods:
    - update(output): Updates the running sum of each metric based on the current batch's output. The output is expected
      to be a dictionary where each key-value pair represents a metric name and its corresponding value.
    - reset(): Resets all internal counters and sums for each metric, preparing metric for a new round of calculations.
      This method is typically called at the start of each epoch or evaluation run.
    - compute(): Calculates and returns the average value of each metric over all processed batches since last reset.
      The return value is a dictionary mirroring the structure of the input to `update`, with each key corresponding to
      a metric name and each value being the average metric value.
    """

    def __init__(self):
        self.kv_sum_metric = defaultdict(lambda: torch.tensor(0., device="cuda"))
        self.kv_sum_inst = defaultdict(lambda: torch.tensor(0., device="cuda"))

        self.kv_metric = defaultdict(lambda: 0)

        super(AutoKVMetric, self).__init__()

    def update(self, output):
        if not isinstance(output, dict):
            raise TypeError('The output must be a key-value dict.')

        for k in output.keys():
            self.kv_sum_metric[k].add_(output[k])
            self.kv_sum_inst[k].add_(1)

    def reset(self):
        for k in self.kv_sum_metric.keys():
            self.kv_sum_metric[k].zero_()
            self.kv_sum_inst[k].zero_()
            self.kv_metric[k] = 0

    def compute(self):
        for k in self.kv_sum_metric.keys():
            if self.kv_sum_inst[k] == 0:
                continue
                # raise NotComputableError('Accuracy must have at least one example before it can be computed')

            metric_value = self.kv_sum_metric[k] / self.kv_sum_inst[k]
            self.kv_metric[k] = metric_value.item()

        return self.kv_metric
