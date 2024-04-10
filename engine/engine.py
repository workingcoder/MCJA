"""MCJA/engine/engine.py
   It defines the creation of training and evaluation engines using the Ignite library.
"""

import numpy as np
import torch

from torch.autograd import no_grad
from ignite.engine import Engine
from ignite.engine import Events
from apex import amp


def create_train_engine(model, optimizer, non_blocking=False):
    """
    A factory function that creates and returns an Ignite Engine configured for training a VI-ReID model. This engine
    orchestrates the training process, managing the data flow, loss calculation, parameter updates, and any additional
    computations needed per iteration. The function encapsulates the core training loop, including data loading to the
    device, executing model's forward pass, computing the loss, performing backpropagation, and updating model weights.

    Args:
    - model (nn.Module): The model to be trained. The model should accept input data, labels, camera IDs, and
      potentially other information like image paths and epoch number, returning computed loss and additional metrics.
    - optimizer (Optimizer): The optimizer used for updating the model parameters based on the computed gradients.
    - non_blocking (bool): If set to True, allows asynchronous data transfers to the GPU for improved efficiency.

    Returns:
    - Engine: An Ignite Engine object that processes batches of data using the provided model and optimizer.
    """

    device = torch.device("cuda", torch.cuda.current_device())

    def _process_func(engine, batch):
        model.train()

        data, labels, cam_ids, img_paths, img_idxes = batch
        epoch = engine.state.epoch
        data = data.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)
        cam_ids = cam_ids.to(device, non_blocking=non_blocking)

        optimizer.zero_grad(set_to_none=True)

        loss, metric = model(data, labels,
                             cam_ids=cam_ids,
                             img_paths=img_paths,
                             epoch=epoch)

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        optimizer.step()

        return metric

    return Engine(_process_func)


def create_eval_engine(model, non_blocking=False):
    """
    A factory function that creates and returns an Ignite Engine configured for evaluating a VI-ReID model. This engine
    manages evaluation process, facilitating the flow of data through the model and the collection of output features
    for later analysis. It operates in evaluation mode, ensuring that the model's behavior is consistent with inference
    conditions, such as disabled dropout layers.

    Args:
    - model (nn.Module): The model to be evaluated. The model should accept input data and potentially other
      information like camera IDs, returning feature representations.
    - non_blocking (bool): If set to True, allows asynchronous data transfers to the GPU to improve efficiency.

    Returns:
    - Engine: An Ignite Engine object that processes batches of data through the provided model in evaluation mode.
    """

    device = torch.device("cuda", torch.cuda.current_device())

    def _process_func(engine, batch):
        model.eval()
        data, labels, cam_ids, img_paths = batch[:4]
        data = data.to(device, non_blocking=non_blocking)

        with no_grad():
            feat = model(data, cam_ids=cam_ids.to(device, non_blocking=non_blocking))

        return feat.data.float().cpu(), labels, cam_ids, np.array(img_paths)

    engine = Engine(_process_func)

    @engine.on(Events.EPOCH_STARTED)
    def clear_data(engine):
        if not hasattr(engine.state, "feat_list"):
            setattr(engine.state, "feat_list", [])
        else:
            engine.state.feat_list.clear()

        if not hasattr(engine.state, "id_list"):
            setattr(engine.state, "id_list", [])
        else:
            engine.state.id_list.clear()

        if not hasattr(engine.state, "cam_list"):
            setattr(engine.state, "cam_list", [])
        else:
            engine.state.cam_list.clear()

        if not hasattr(engine.state, "img_path_list"):
            setattr(engine.state, "img_path_list", [])
        else:
            engine.state.img_path_list.clear()

    @engine.on(Events.ITERATION_COMPLETED)
    def store_data(engine):
        engine.state.feat_list.append(engine.state.output[0])
        engine.state.id_list.append(engine.state.output[1])
        engine.state.cam_list.append(engine.state.output[2])
        engine.state.img_path_list.append(engine.state.output[3])

    return engine
