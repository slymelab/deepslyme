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

import re
import math
import logging
from functools import partial
from itertools import islice
from collections.abc import Iterable, Iterator
from typing import Any, Optional, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from slyme.context import Context, Ref
from slyme.node import Node, node, Auto, sequential_exec
from slyme.utils.pytree import PyTreeEngine
import deepslyme.utils.accelerator as accelerator
from deepslyme.utils.optimizer import OPTIMIZER_REGISTRY
from deepslyme.utils.scheduler import SCHEDULER_REGISTRY
from .distributed import DistributedState

logger = logging.getLogger(__name__)


@node
def set_seed(
    ctx: Context,
    /,
    *,
    seed: Auto[int] = 42,
) -> Context:
    """Set the random seed for reproducibility."""
    accelerator.set_device_seed(seed)
    return ctx


@node
def free_memory(ctx: Context, /) -> Context:
    """Clear cache to avoid memory fragmentation."""
    accelerator.empty_cache()
    return ctx


@node
def empty_cache_by_step(
    ctx: Context,
    /,
    *,
    empty_steps: Auto[int] = 1,
    state_global_step: Auto[int],
) -> Context:
    """Clear cache to avoid memory fragmentation."""
    if state_global_step > 0 and state_global_step % empty_steps == 0:
        accelerator.empty_cache()
    return ctx


def _seed_worker(worker_id: int, num_workers: int, rank: int):
    init_seed = torch.initial_seed() % (2**32)
    worker_seed = (num_workers * rank + init_seed) % (2**32)
    accelerator.set_device_seed(worker_seed)


@node
def prepare_distributed_dataloader(
    ctx: Context,
    /,
    *,
    dataset: Auto[Any],
    data_collator: Auto[Any],
    bsz: Auto[int],
    num_workers: Auto[int],
    pin_memory: Auto[bool],
    persistent_workers: Auto[bool],
    drop_last: Auto[bool] = False,
    prefetch_factor: Auto[Optional[int]],
    process_index: Auto[int],
    dataloader: Ref[DataLoader],
    distributed_state: Auto[DistributedState],
    seed: Auto[int] = 42,
    is_training: Auto[bool] = True,
) -> Context:
    """Prepare a PyTorch DataLoader mapped with a DistributedSampler."""
    sampler = DistributedSampler(
        dataset,
        num_replicas=distributed_state.num_processes,
        rank=distributed_state.process_index,
        shuffle=is_training,
        seed=seed,
        drop_last=drop_last,
    )

    dataloader_params = {
        "batch_size": bsz,
        "collate_fn": data_collator,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "sampler": sampler,
        "drop_last": drop_last,
    }

    if prefetch_factor is not None and num_workers > 0:
        dataloader_params["prefetch_factor"] = prefetch_factor

    if is_training:
        dataloader_params["worker_init_fn"] = partial(
            _seed_worker, num_workers=num_workers, rank=process_index
        )

    return ctx.set(dataloader, DataLoader(dataset, **dataloader_params))


_inputs_pytree_engine = PyTreeEngine("prepare_inputs")


@node
def prepare_inputs(
    ctx: Context,
    /,
    *,
    step_inputs: Ref[Any],
    device: Auto[torch.device],
    dtype: Auto[Optional[torch.dtype]] = None,
) -> Context:
    """Recursively convert dtype and device of step_inputs."""

    def _prepare(element):
        if isinstance(element, torch.Tensor):
            kwargs = {"device": device}
            if dtype is not None and (
                torch.is_floating_point(element) or torch.is_complex(element)
            ):
                kwargs["dtype"] = dtype
            return element.to(**kwargs, non_blocking=True)
        return element

    return ctx.set(
        step_inputs,
        _inputs_pytree_engine.map(_prepare, ctx.get(step_inputs)),
    )


@node
def clean_step_inputs(
    ctx: Context,
    /,
    *,
    step_inputs: Ref[Any],
) -> Context:
    """Delete step_inputs to free memory."""
    return ctx.delete(step_inputs)


@node
def setup_dtype(
    ctx: Context,
    /,
    *,
    dtype: Ref[torch.dtype],
    bf16: Auto[bool] = True,
    fp16: Auto[bool] = False,
):
    """Setup dtype."""
    if bf16 and fp16:
        raise ValueError("bf16 and fp16 cannot be True at the same time.")
    if bf16:
        ctx = ctx.set(dtype, torch.bfloat16)
    elif fp16:
        ctx = ctx.set(dtype, torch.float16)
    else:
        ctx = ctx.set(dtype, torch.float32)
    return ctx


def _get_parameter_names(
    model: nn.Module,
    forbidden_layer_types: list[type],
    forbidden_layer_names: Optional[list[str]] = None,
):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    forbidden_layer_patterns = (
        [re.compile(pattern) for pattern in forbidden_layer_names]
        if forbidden_layer_names is not None
        else []
    )
    result = []
    for name, child in model.named_children():
        child_params = _get_parameter_names(
            child, forbidden_layer_types, forbidden_layer_names
        )
        result += [
            f"{name}.{n}"
            for n in child_params
            if not isinstance(child, tuple(forbidden_layer_types))
            and not any(
                pattern.search(f"{name}.{n}".lower())
                for pattern in forbidden_layer_patterns
            )
        ]
    # Add model specific parameters that are not in any child
    result += [
        k
        for k in model._parameters
        if not any(pattern.search(k.lower()) for pattern in forbidden_layer_patterns)
    ]

    return result


def _get_decay_parameter_names(model) -> list[str]:
    """
    Get all parameter names that weight decay will be applied to.

    This function filters out parameters in two ways:
    1. By layer type (instances of layers specified in ALL_LAYERNORM_LAYERS)
    2. By parameter name patterns (containing 'bias', or variation of 'norm')
    """
    forbidden_name_patterns = [
        r"bias",
        r"layernorm",
        r"rmsnorm",
        r"(?:^|\.)norm(?:$|\.)",
        r"_norm(?:$|\.)",
    ]
    decay_parameters = _get_parameter_names(
        model, [nn.LayerNorm], forbidden_name_patterns
    )
    return decay_parameters


@node
def create_optimizer(
    ctx: Context,
    /,
    *,
    model: Auto[nn.Module],
    weight_decay: Auto[float],
    optimizer_cls: Auto[str],
    optimizer_kwargs: Auto[dict],
    optimizer: Ref[torch.optim.Optimizer],
) -> Context:
    """"""
    decay_parameters = _get_decay_parameter_names(model)
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if n in decay_parameters and p.requires_grad
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if n not in decay_parameters and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]

    return ctx.set(
        optimizer,
        OPTIMIZER_REGISTRY.get(optimizer_cls)(
            optimizer_grouped_parameters, **optimizer_kwargs
        ),
    )


@node
def create_scheduler(
    ctx: Context,
    /,
    *,
    lr_scheduler_type: Auto[str],
    lr_scheduler_kwargs: Auto[dict],
    optimizer: Auto[torch.optim.Optimizer],
    state_max_steps: Auto[int],
    warmup_ratio: Auto[float],
    lr_scheduler: Ref[Any],
) -> Context:
    """
    Node for building learning rate schedulers using pure PyTorch.
    Eliminates the transformers library dependency.
    """
    if lr_scheduler_type not in SCHEDULER_REGISTRY:
        raise ValueError(
            f"Unsupported lr_scheduler_type: '{lr_scheduler_type}'. "
            f"Available options are: {list(SCHEDULER_REGISTRY.keys())}"
        )

    num_warmup_steps = math.ceil(warmup_ratio * state_max_steps)
    scheduler_factory = SCHEDULER_REGISTRY.get(lr_scheduler_type)

    # Dynamically instantiate the scheduler
    scheduler = scheduler_factory(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=state_max_steps,
        **lr_scheduler_kwargs,
    )

    return ctx.set(lr_scheduler, scheduler)


@node
def init_progress(
    ctx: Context,
    /,
    *,
    progress: Ref[Any],
    state_max_steps: Auto[int],
    process_index: Auto[int] = 0,
    task_desc: str = "",
) -> Context:
    """Initialize tqdm progress bar, only on the main process (rank 0)."""
    if process_index == 0:
        return ctx.set(progress, tqdm(total=state_max_steps, desc=task_desc))
    else:
        # NOTE: Avoid auto injection error in other processes
        return ctx.set(progress, None)


@node
def update_progress(
    ctx: Context,
    /,
    *,
    progress: Auto[Union[tqdm, None]],
    process_index: Auto[int] = 0,
    state_global_step: Auto[int],
) -> Context:
    """Update tqdm progress bar, only on the main process (rank 0)."""
    if process_index == 0 and progress is not None:
        progress.update(state_global_step - progress.n)
    return ctx


@node
def destroy_progress(
    ctx: Context,
    /,
    *,
    progress: Auto[Union[tqdm, None]],
    process_index: Auto[int] = 0,
) -> Context:
    """Close tqdm progress bar, only on the main process (rank 0)."""
    if process_index == 0 and progress is not None:
        progress.close()
    return ctx


def batched(iterable: Iterable, n: int) -> Iterator[tuple]:
    """Safe batched iterator."""
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while chunk := tuple(islice(it, n)):
        yield chunk


@node
def dataloader_loop(
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
    grad_acc_steps: Auto[int],
) -> Context:
    """
    Generic Dataloader loop.
    Dynamically handles the final tail batch to prevent hangs and scale mismatch
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
            ctx = sequential_exec(ctx, mini_step_nodes)

        ctx = ctx.set(state_global_step, ctx.get(state_global_step) + 1)
        ctx = sequential_exec(ctx, global_step_nodes)

        ctx = ctx.delete(step)
        if ctx.get(state_global_step) >= state_max_steps:
            ctx = ctx.set(control_should_stop_training, True)
        if ctx.get(control_should_stop_training) or ctx.get(control_should_stop_epoch):
            break
    return ctx


@node
def dataloader_loop_with_micro_steps(
    ctx: Context,
    /,
    *,
    dataloader: Auto[DataLoader],
    step: Ref[Any],
    step_inputs: Ref[Any],
    step_current_gas: Ref[int],
    step_should_sync_grad: Ref[bool],
    state_global_step: Ref[int],
    step_micro_batches: Ref[list[dict[Ref, Any]]],
    mini_step_nodes: list[Node],
    micro_step_nodes: list[Node],
    global_step_nodes: list[Node],
    control_should_stop_epoch: Ref[bool],
    control_should_stop_training: Ref[bool],
    state_max_steps: Auto[int],
    grad_acc_steps: Auto[int],
) -> Context:
    """
    Generic Data-Driven Dataloader loop with nested micro-steps.
    It expects `mini_step_nodes` to produce a list of dictionary updates (`step_micro_batches`),
    where each dictionary contains the specific context mutations for that micro-batch.
    """
    for chunk in batched(dataloader, grad_acc_steps):
        current_gas = len(chunk)

        for i, inputs in enumerate(chunk):
            is_last_mini = i == current_gas - 1
            ctx = ctx.update(
                {
                    step_inputs: inputs,
                }
            )
            ctx = sequential_exec(ctx, mini_step_nodes)
            micro_batches = ctx.get(step_micro_batches)
            num_micro_batches = len(micro_batches)
            final_gas = current_gas * num_micro_batches

            for mb_idx, mb_updates in enumerate(micro_batches):
                is_last_micro = mb_idx == num_micro_batches - 1
                should_sync = is_last_mini and is_last_micro
                ctx = ctx.update(mb_updates)
                ctx = ctx.update(
                    {
                        step_should_sync_grad: should_sync,
                        step_current_gas: final_gas,
                    }
                )
                ctx = sequential_exec(ctx, micro_step_nodes)

        ctx = ctx.set(state_global_step, ctx.get(state_global_step) + 1)
        ctx = sequential_exec(ctx, global_step_nodes)

        ctx = ctx.delete(step)
        if ctx.get(state_global_step) >= state_max_steps:
            ctx = ctx.set(control_should_stop_training, True)
        if ctx.get(control_should_stop_training) or ctx.get(control_should_stop_epoch):
            break

    return ctx
