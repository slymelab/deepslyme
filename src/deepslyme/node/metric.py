# Copyright 2026 Slymer-Tech
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

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Literal, Union

import torch
import torch.distributed as dist

from slyme.context import Context, Ref
from slyme.node import node, Auto
from slyme.utils.pytree import PyTreeEngine


@dataclass(frozen=True)
class MetricRecord:
    value: Any


BuiltinStrategy = Literal["mean", "sum", "max", "min", "latest"]
ReducerCallable = Callable[
    [list[MetricRecord], torch.device], Union[Any, dict[str, Any]]
]
ReducerType = Union[BuiltinStrategy, ReducerCallable]


@dataclass(frozen=True)
class MetricDef:
    reduce: ReducerType = "mean"


_metrics_pytree_engine = PyTreeEngine("sanitize_metrics")


@node
def collect_metrics(
    ctx: Context,
    /,
    *,
    current_metrics: Auto[dict[str, Any]],
    step_metrics_history: Ref[dict[str, list[MetricRecord]]],
) -> Context:
    """Micro-step node: Collect metrics into columnar history."""
    history = ctx.get(step_metrics_history, None)
    if history is None:
        history = defaultdict(list)

    def _sanitize(element):
        if isinstance(element, torch.Tensor):
            if element.numel() == 1:
                return element.detach().float().item()
            return element.detach().cpu()
        return element

    for name, val in current_metrics.items():
        cleaned_val = _metrics_pytree_engine.map(_sanitize, val)
        history[name].append(MetricRecord(value=cleaned_val))

    return ctx.set(step_metrics_history, history)


def _sync_tensors(vals: list[float], op: Any, device: torch.device) -> list[float]:
    """Helper to sync tensors across distributed processes; acts as no-op if single process."""
    if not vals or not dist.is_initialized():
        return vals
    buf = torch.tensor(vals, device=device, dtype=torch.float32)
    dist.all_reduce(buf, op=op)
    return buf.tolist()


@node
def reduce_and_log_metrics(
    ctx: Context,
    /,
    *,
    metric_defs: Auto[dict[str, Union[ReducerType, MetricDef]]],
    step_metrics_history: Ref[dict[str, list[MetricRecord]]],
    state_log_history: Ref[list],
    state_global_step: Auto[int],
    process_index: Auto[int],
    device: Auto[torch.device],
) -> Context:
    """Global-step node: Reduce history temporally and spatially, then log."""
    history = ctx.get(step_metrics_history)
    if not history:
        return ctx

    # 1. Normalize metric definitions
    normalized_defs: dict[str, MetricDef] = {}
    for name in history.keys():
        mdef = metric_defs.get(name, MetricDef(reduce="mean"))
        normalized_defs[name] = (
            mdef if isinstance(mdef, MetricDef) else MetricDef(reduce=mdef)
        )

    final_log_values = {}

    # 2. Sort names to ensure deterministic order across all distributed processes
    builtin_names = sorted(
        k for k, v in normalized_defs.items() if isinstance(v.reduce, str)
    )
    custom_names = sorted(k for k, v in normalized_defs.items() if callable(v.reduce))

    # 3. Setup sync groups by ReduceOp to minimize all_reduce calls
    # Format: op_type -> {"op": dist.ReduceOp, "keys": [(name, type), ...], "vals": [float, ...]}
    sync_groups = {
        "SUM": {
            "op": dist.ReduceOp.SUM if dist.is_initialized() else None,
            "keys": [],
            "vals": [],
        },
        "MAX": {
            "op": dist.ReduceOp.MAX if dist.is_initialized() else None,
            "keys": [],
            "vals": [],
        },
        "MIN": {
            "op": dist.ReduceOp.MIN if dist.is_initialized() else None,
            "keys": [],
            "vals": [],
        },
    }

    # 4. Local Temporal Reduction
    for name in builtin_names:
        records = history.get(name)
        if not records:
            continue

        strategy = normalized_defs[name].reduce
        if strategy == "mean":
            v_sum = sum(r.value for r in records)
            w_sum = len(records)
            sync_groups["SUM"]["keys"].extend([(name, "mean_v"), (name, "mean_w")])
            sync_groups["SUM"]["vals"].extend([v_sum, w_sum])
        elif strategy == "sum":
            sync_groups["SUM"]["keys"].append((name, "direct"))
            sync_groups["SUM"]["vals"].append(sum(r.value for r in records))
        elif strategy == "max":
            sync_groups["MAX"]["keys"].append((name, "direct"))
            sync_groups["MAX"]["vals"].append(max(r.value for r in records))
        elif strategy == "min":
            sync_groups["MIN"]["keys"].append((name, "direct"))
            sync_groups["MIN"]["vals"].append(min(r.value for r in records))
        elif strategy == "latest":
            final_log_values[name] = records[-1].value

    # 5. Global Spatial Reduction (Distributed Sync) & Reconstruction
    mean_accum = defaultdict(dict)
    for group in sync_groups.values():
        synced_vals = _sync_tensors(group["vals"], group["op"], device)

        # Unpack the synced 1D tensor back to individual metrics
        for (name, ktype), val in zip(group["keys"], synced_vals):
            if ktype == "direct":
                final_log_values[name] = val
            elif ktype == "mean_v":
                mean_accum[name]["v"] = val
            elif ktype == "mean_w":
                mean_accum[name]["w"] = val

    # 6. Finalize 'mean' division post-sync
    for name, d in mean_accum.items():
        final_log_values[name] = (d["v"] / d["w"]) if d.get("w", 0) > 0 else 0.0

    # 7. Process Custom Reducers
    for name in custom_names:
        records = history.get(name)
        if not records:
            continue
        result = normalized_defs[name].reduce(records, device)
        if isinstance(result, dict):
            final_log_values.update(result)
        else:
            final_log_values[name] = result

    # 8. Logging (Main process only)
    if process_index == 0:
        log_entry = {"step": state_global_step, **final_log_values}

        log_history = ctx.get(state_log_history, None)
        if log_history is None:
            log_history = []
        log_history.append(log_entry)
        ctx = ctx.set(state_log_history, log_history)

        log_strs = [
            f"{k}: {v}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in final_log_values.items()
        ]
        print(f"Step {state_global_step} | " + " | ".join(log_strs))

    # 9. Reset metrics history for the next step
    return ctx.set(step_metrics_history, defaultdict(list))
