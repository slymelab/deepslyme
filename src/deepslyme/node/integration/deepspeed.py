# Copyright 2026 The SlymeLab Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
from datetime import timedelta
from typing import Any

import deepspeed
import torch
from slyme.context import Context, Ref, RefLike
from slyme.node import Auto, Node, node, wrapper
from slyme.utils.pytree import KeyPath, MappingKey, PyTreeEngine

from deepslyme.node.distributed import DistributedState
from deepslyme.utils import accelerator

logger = logging.getLogger(__name__)


def _mapping_path(*keys: str) -> KeyPath:
    return tuple(MappingKey(key) for key in keys)


@node
def deepspeed_init_distributed(
    ctx: Context,
    /,
    *,
    ddp_timeout: Auto[int] = 1800,
    distributed_state: Ref[DistributedState],
) -> None:
    """Initialize the DeepSpeed distributed environment."""
    deepspeed.init_distributed(
        dist_backend=accelerator.current_comm_backend_name(),
        timeout=timedelta(seconds=ddp_timeout),
    )

    distributed_state_ = DistributedState()
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank != -1:
        accelerator.set_device_index(local_rank)
    else:
        accelerator.set_device_index(distributed_state_.device.index)

    ctx.set(distributed_state, distributed_state_)


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
    hidden_size: Auto[int | None] = None,
    must_match: Auto[bool] = True,
) -> None:
    """
    Advanced PyTree-based DeepSpeed Config parser.
    Treats target configurations as leaf nodes in a tree, achieving safe,
    side-effect-free "auto" injection and conflict validation.
    """
    with open(deepspeed_path, "r") as f:
        ds_config = json.load(f)

    # Declare the expected PyTree mapping paths and their runtime values.
    mappings = {
        _mapping_path("train_micro_batch_size_per_gpu"): bsz,
        _mapping_path("gradient_accumulation_steps"): grad_acc_steps,
        _mapping_path("train_batch_size"): bsz * grad_acc_steps * num_processes,
        _mapping_path("gradient_clipping"): max_grad_norm,
        _mapping_path("fp16", "enabled"): fp16,
        _mapping_path("bf16", "enabled"): bf16,
    }

    if hidden_size is not None:
        mappings[_mapping_path("zero_optimization", "reduce_bucket_size")] = (
            hidden_size * hidden_size
        )
        mappings[_mapping_path("zero_optimization", "stage3_prefetch_bucket_size")] = (
            int(0.9 * hidden_size * hidden_size)
        )
        mappings[
            _mapping_path("zero_optimization", "stage3_param_persistence_threshold")
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
    ctx.set(deepspeed_config, new_ds_config)


@node
def deepspeed_set_grad_acc_boundary(
    ctx: Context,
    /,
    *,
    model_for_training: Auto[Any],
    step_should_sync_grad: Auto[bool],
) -> None:
    """Set the gradient accumulation boundary for DeepSpeed."""
    model_for_training.set_gradient_accumulation_boundary(step_should_sync_grad)


@wrapper
def deepspeed_with_grad_acc_boundary(
    ctx: Context,
    wrapped: Node,
    call_next,
    /,
    *,
    model_for_training: Auto[Any],
    step_should_sync_grad: Auto[bool],
    reset_boundary_to: Auto[bool | None] = None,
) -> Any:
    reset_val = (
        reset_boundary_to
        if reset_boundary_to is not None
        else model_for_training.is_gradient_accumulation_boundary()
    )
    model_for_training.set_gradient_accumulation_boundary(step_should_sync_grad)
    try:
        return call_next(ctx)
    finally:
        model_for_training.set_gradient_accumulation_boundary(reset_val)


@node
def deepspeed_backward(
    ctx: Context,
    /,
    *,
    step_loss: Auto[torch.Tensor],
    model_for_training: Auto[Any],
    step_current_gas: Auto[int],
) -> None:
    """
    Execute backward pass.
    Disables engine-native scale_wrt_gas to allow manual dynamic scaling using step_current_gas.
    """
    model_for_training.backward(step_loss / step_current_gas, scale_wrt_gas=False)


@node
def deepspeed_step(
    ctx: Context,
    /,
    *,
    model_for_training: Auto[Any],
) -> None:
    """Execute optimizer step via DeepSpeed Engine."""
    model_for_training.step()


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
) -> None:
    """Initialize the DeepSpeed Engine with model, optimizer, and scheduler."""
    optimizer_ = ctx.get(optimizer, None)
    scheduler_ = ctx.get(lr_scheduler, None)

    model_engine, optimizer_, _, scheduler_ = deepspeed.initialize(
        model=model,
        optimizer=optimizer_,
        config=deepspeed_config,
        lr_scheduler=scheduler_,
    )

    updates: dict[RefLike, Any] = {model_for_training: model_engine}
    if optimizer_ is not None:
        updates[optimizer] = optimizer_
    if scheduler_ is not None:
        updates[lr_scheduler] = scheduler_

    ctx.update(updates)
