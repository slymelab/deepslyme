import math
import logging
from typing import Any, list
import torch
from slyme.context import Context, Ref
from slyme.node import Node, node, sequential_exec, Auto

logger = logging.getLogger(__name__)


@node
def init_training_state(
    ctx: Context,
    /,
    *,
    arg_max_steps: Auto[int] = -1,
    arg_num_train_epochs: Auto[float],
    grad_acc_steps: Auto[int],
    train_dataloader: Auto[Any],
    drop_last: Auto[bool] = False,
    state_num_train_epochs: Ref[int],
    state_max_steps: Ref[int],
    state_log_history: Ref[list[dict]],
    state_global_step: Ref[int],
    state_epoch_idx: Ref[int],
    state_epoch: Ref[float],
    control_should_stop_epoch: Ref[bool],
    control_should_stop_training: Ref[bool],
) -> Context:
    """Initialize training state, including max_steps and num_train_epochs."""
    try:
        len_dataloader = len(train_dataloader)
    except (AttributeError, TypeError):
        len_dataloader = None

    if arg_max_steps > 0:
        max_steps = arg_max_steps
        if len_dataloader:
            num_update_steps_per_epoch = max(
                len_dataloader // grad_acc_steps
                + int(len_dataloader % grad_acc_steps > 0 and not drop_last),
                1,
            )
            num_train_epochs = math.ceil(max_steps / num_update_steps_per_epoch)
        else:
            num_train_epochs = (
                math.ceil(arg_num_train_epochs) if arg_num_train_epochs > 0 else 1
            )
    else:
        if not len_dataloader:
            raise ValueError(
                "Cannot infer max_steps from dataloader when arg_max_steps <= 0 and "
                "dataloader does not support `len()`"
            )

        num_update_steps_per_epoch = max(
            len_dataloader // grad_acc_steps
            + int(len_dataloader % grad_acc_steps > 0 and not drop_last),
            1,
        )
        max_steps = math.ceil(arg_num_train_epochs * num_update_steps_per_epoch)
        num_train_epochs = math.ceil(arg_num_train_epochs)

    return ctx.update(
        {
            state_max_steps: max_steps,
            state_num_train_epochs: num_train_epochs,
            state_log_history: [],
            state_global_step: 0,
            state_epoch_idx: 0,
            state_epoch: 0.0,
            control_should_stop_epoch: False,
            control_should_stop_training: False,
        }
    )


@node
def epoch_loop(
    ctx: Context,
    /,
    *,
    state_num_train_epochs: Auto[int],
    state_epoch_idx: Ref[int],
    control_should_stop_training: Ref[bool],
    nodes: list[Node],
) -> Context:
    """Epoch loop."""
    for epoch in range(ctx.get(state_epoch_idx), state_num_train_epochs):
        ctx = ctx.set(state_epoch_idx, epoch)
        ctx = sequential_exec(ctx, nodes)

        if ctx.get(control_should_stop_training):
            break
    return ctx


@node
def dataloader_set_epoch(
    ctx: Context,
    /,
    *,
    dataloader: Auto[Any],
    state_epoch_idx: Auto[int],
) -> Context:
    """Set epoch for dataloader sampler."""
    if hasattr(dataloader, "sampler") and hasattr(dataloader.sampler, "set_epoch"):
        dataloader.sampler.set_epoch(state_epoch_idx)
    return ctx


@node
def compute_loss(
    ctx: Context,
    /,
    *,
    step_inputs: Auto[Any],
    model_for_training: Auto[Any],
    step_loss: Ref[torch.Tensor],
) -> Context:
    """Compute loss for SFT."""
    output = model_for_training(**step_inputs)
    return ctx.set(step_loss, output["loss"])
