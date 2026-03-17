import os
import json
import logging
from datetime import timedelta
from typing import Any, Optional
from itertools import islice
from collections.abc import Iterable, Iterator
import torch
from torch.utils.data import DataLoader
import deepspeed
from slyme.context import Context, Ref
from slyme.node import Node, node, sequential_exec, Auto
from slyme.utils.pytree import P, PyTreeEngine
import deepslyme.utils.accelerator as accelerator
from deepslyme.node.distributed import DistributedState

logger = logging.getLogger(__name__)


@node
def deepspeed_init_distributed(
    ctx: Context,
    /,
    *,
    ddp_timeout: Auto[int] = 1800,
    distributed_state: Ref[DistributedState],
) -> Context:
    """Initialize the DeepSpeed distributed environment."""
    deepspeed.init_distributed(
        dist_backend=accelerator.current_comm_backend_name(),
        timeout=timedelta(seconds=ddp_timeout),
    )

    distributed_state_ = DistributedState()
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        accelerator.set_device_index(local_rank)
    else:
        accelerator.set_device_index(distributed_state_.device.index)

    return ctx.set(distributed_state, distributed_state_)


@node
def deepspeed_config_init(
    ctx: Context,
    /,
    *,
    deepspeed_path: Auto[str],
    deepspeed_config: Ref[dict],
    bsz: Auto[int],
    grad_acc_steps: Auto[int],
    num_processes: Auto[int],
    fp16: Auto[bool] = False,
    bf16: Auto[bool] = True,
    max_grad_norm: Auto[float] = 1.0,
    hidden_size: Auto[Optional[int]] = None,
    must_match: Auto[bool] = True,
) -> Context:
    """
    Advanced PyTree-based DeepSpeed Config parser.
    Treats target configurations as leaf nodes in a tree, achieving safe,
    side-effect-free "auto" injection and conflict validation.
    """
    with open(deepspeed_path, "r") as f:
        ds_config = json.load(f)

    # Declare expected paths and values using KeyPathExpr (P)
    mappings = {
        tuple(P["train_micro_batch_size_per_gpu"]): bsz,
        tuple(P["gradient_accumulation_steps"]): grad_acc_steps,
        tuple(P["train_batch_size"]): bsz * grad_acc_steps * num_processes,
        tuple(P["gradient_clipping"]): max_grad_norm,
        tuple(P["fp16"]["enabled"]): fp16,
        tuple(P["bf16"]["enabled"]): bf16,
    }

    if hidden_size is not None:
        mappings[tuple(P["zero_optimization"]["reduce_bucket_size"])] = (
            hidden_size * hidden_size
        )
        mappings[tuple(P["zero_optimization"]["stage3_prefetch_bucket_size"])] = int(
            0.9 * hidden_size * hidden_size
        )
        mappings[
            tuple(P["zero_optimization"]["stage3_param_persistence_threshold"])
        ] = 10 * hidden_size

    # Interceptor: Stop flattening if the path is declared in mappings
    def _is_target_leaf(element, traverse_aux):
        return traverse_aux.key_path in mappings

    # Flatten, validate, and inject
    engine = PyTreeEngine(name="ds_config_engine")
    leaves_with_path, treedef = engine.flatten_with_key_path(
        ds_config, is_leaf=_is_target_leaf
    )

    mismatches = []
    new_leaves = []

    for path, val in leaves_with_path:
        if path in mappings:
            target_val = mappings[path]
            if val == "auto":
                new_leaves.append(target_val)
            elif val is not None and val != target_val:
                mismatches.append(
                    f"- {engine.codify_key_path(path)}: Config has hardcoded {val}, but runtime args specify {target_val}"
                )
                new_leaves.append(val)
            else:
                new_leaves.append(val)
        else:
            new_leaves.append(val)

    if must_match and mismatches:
        mismatch_str = "\n".join(mismatches)
        raise ValueError(
            "DeepSpeed configuration conflicts with runtime arguments:\n"
            f"{mismatch_str}\n"
            "If you want arguments to override these values, set the corresponding fields to 'auto' in your JSON config."
        )

    # Reconstruct as a new dictionary (functional approach)
    new_ds_config = engine.unflatten(treedef, new_leaves)
    return ctx.set(deepspeed_config, new_ds_config)


def batched(iterable: Iterable, n: int) -> Iterator[tuple]:
    """Safe batched iterator."""
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while chunk := tuple(islice(it, n)):
        yield chunk


@node
def deepspeed_dataloader_loop(
    ctx: Context,
    /,
    *,
    dataloader: Auto[DataLoader],
    step: Ref[Any],
    step_inputs: Ref[Any],
    step_current_gas: Ref[int],
    step_should_sync_grad: Ref[bool],
    state_global_step: Ref[int],
    mini_step_nodes: list[Node],
    global_step_nodes: list[Node],
    control_should_stop_epoch: Ref[bool],
    control_should_stop_training: Ref[bool],
    state_max_steps: Auto[int],
    model_for_training: Auto[Any],
    grad_acc_steps: Auto[int],
) -> Context:
    """
    DeepSpeed Dataloader loop.
    Dynamically handles the final tail batch to prevent DDP hangs and scale mismatch
    when the remaining steps are less than grad_acc_steps.
    """
    for chunk in batched(dataloader, grad_acc_steps):
        current_gas = len(chunk)
        ctx = ctx.set(step_current_gas, current_gas)

        for i, inputs in enumerate(chunk):
            should_sync = i == current_gas - 1
            ctx = ctx.update(
                {
                    step_inputs: inputs,
                    step_should_sync_grad: should_sync,
                }
            )
            model_for_training.set_gradient_accumulation_boundary(should_sync)
            ctx = sequential_exec(ctx, mini_step_nodes)

        # NOTE: We set the boundary to True to force engine.step to work.
        model_for_training.set_gradient_accumulation_boundary(True)
        try:
            ctx = ctx.set(state_global_step, ctx.get(state_global_step) + 1)
            ctx = sequential_exec(ctx, global_step_nodes)
        finally:
            model_for_training.set_gradient_accumulation_boundary(False)

        ctx = ctx.delete(step)
        if ctx.get(state_global_step) >= state_max_steps:
            ctx = ctx.set(control_should_stop_training, True)
        if ctx.get(control_should_stop_training) or ctx.get(control_should_stop_epoch):
            break
    return ctx


@node
def deepspeed_backward(
    ctx: Context,
    /,
    *,
    step_loss: Auto[torch.Tensor],
    model_for_training: Auto[Any],
    step_current_gas: Auto[int],
) -> Context:
    """
    Execute backward pass.
    Disables engine-native scale_wrt_gas to allow manual dynamic scaling using step_current_gas.
    """
    model_for_training.backward(step_loss / step_current_gas, scale_wrt_gas=False)
    return ctx


@node
def deepspeed_step(
    ctx: Context,
    /,
    *,
    model_for_training: Auto[Any],
) -> Context:
    """Execute optimizer step via DeepSpeed Engine."""
    model_for_training.step()
    return ctx


@node
def deepspeed_initialize(
    ctx: Context,
    /,
    *,
    model: Auto[torch.nn.Module],
    optimizer: Ref[torch.optim.Optimizer],
    model_for_training: Ref[Any],
    deepspeed_config: Auto[dict],
    lr_scheduler: Ref[Any],
) -> Context:
    """Initialize the DeepSpeed Engine with model, optimizer, and scheduler."""
    optimizer_ = ctx.get(optimizer, None)
    scheduler_ = ctx.get(lr_scheduler, None)

    model_engine, optimizer_, _, scheduler_ = deepspeed.initialize(
        model=model,
        optimizer=optimizer_,
        config=deepspeed_config,
        lr_scheduler=scheduler_,
    )

    updates = {model_for_training: model_engine}
    if optimizer_ is not None:
        updates[optimizer] = optimizer_
    if scheduler_ is not None:
        updates[lr_scheduler] = scheduler_

    return ctx.update(updates)
