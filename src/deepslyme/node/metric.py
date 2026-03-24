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
    gas: float


BuiltinStrategy = Literal["weighted_mean", "mean", "sum", "max", "min", "latest"]
ReducerCallable = Callable[
    [list[MetricRecord], torch.device], Union[Any, dict[str, Any]]
]
ReducerType = Union[BuiltinStrategy, ReducerCallable]


@dataclass(frozen=True)
class MetricDef:
    reduce: ReducerType = "weighted_mean"


_metrics_pytree_engine = PyTreeEngine("sanitize_metrics")


@node
def collect_metrics(
    ctx: Context,
    /,
    *,
    current_metrics: Auto[dict[str, Any]],
    step_current_gas: Auto[float],
    step_metrics_history: Ref[dict[str, list[MetricRecord]]],
) -> Context:
    """Micro-step node: Collect metrics into columnar history."""
    history = ctx.get(step_metrics_history) or defaultdict(list)

    def _sanitize(element):
        if isinstance(element, torch.Tensor):
            if element.numel() == 1:
                return element.detach().float().item()
            return element.detach().cpu()
        return element

    for name, val in current_metrics.items():
        cleaned_val = _metrics_pytree_engine.map(_sanitize, val)
        history[name].append(MetricRecord(value=cleaned_val, gas=step_current_gas))

    return ctx.set(step_metrics_history, history)


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

    normalized_defs: dict[str, MetricDef] = {}
    for name in history.keys():
        mdef = metric_defs.get(name)
        if mdef is None:
            normalized_defs[name] = MetricDef(reduce="weighted_mean")
        elif isinstance(mdef, MetricDef):
            normalized_defs[name] = mdef
        else:
            normalized_defs[name] = MetricDef(reduce=mdef)

    final_log_values = {}
    builtin_local_vals = {}
    builtin_defs = {
        k: v for k, v in normalized_defs.items() if isinstance(v.reduce, str)
    }

    for name, mdef in builtin_defs.items():
        records = history.get(name)
        if not records:
            continue

        strategy = mdef.reduce
        if strategy == "weighted_mean":
            tot_w = sum(r.gas for r in records)
            builtin_local_vals[name] = (
                sum(r.gas * r.value for r in records) / tot_w if tot_w > 0 else 0.0
            )
        elif strategy == "mean":
            builtin_local_vals[name] = sum(r.value for r in records) / len(records)
        elif strategy in ("sum", "max", "min"):
            op = {"sum": sum, "max": max, "min": min}[strategy]
            builtin_local_vals[name] = op(r.value for r in records)
        elif strategy == "latest":
            builtin_local_vals[name] = records[-1].value

    if dist.is_initialized() and builtin_local_vals:
        world_size = dist.get_world_size()
        groups = {"sum": [], "max": [], "min": []}

        for name in builtin_local_vals:
            strategy = builtin_defs[name].reduce
            if strategy in ("mean", "weighted_mean", "sum"):
                groups["sum"].append(name)
            else:
                groups[strategy].append(name)

        def _reduce_group(names: list[str], op: Any):
            if not names:
                return
            buf = torch.tensor(
                [builtin_local_vals[n] for n in names],
                device=device,
                dtype=torch.float32,
            )
            dist.all_reduce(buf, op=op)
            for idx, n in enumerate(names):
                val = buf[idx].item()
                if builtin_defs[n].reduce in ("mean", "weighted_mean"):
                    val /= world_size
                final_log_values[n] = val

        _reduce_group(groups["sum"], dist.ReduceOp.SUM)
        _reduce_group(groups["max"], dist.ReduceOp.MAX)
        _reduce_group(groups["min"], dist.ReduceOp.MIN)
    else:
        final_log_values.update(builtin_local_vals)

    custom_defs = {k: v for k, v in normalized_defs.items() if callable(v.reduce)}
    for name, mdef in custom_defs.items():
        records = history.get(name)
        if not records:
            continue

        result = mdef.reduce(records, device)
        if isinstance(result, dict):
            final_log_values.update(result)
        else:
            final_log_values[name] = result

    if process_index == 0:
        log_entry = {"step": state_global_step, **final_log_values}
        ctx.get(state_log_history).append(log_entry)

        log_strs = []
        for k, v in final_log_values.items():
            fmt_v = f"{v:.4f}" if isinstance(v, float) else str(v)
            log_strs.append(f"{k}: {fmt_v}")

        print(f"Step {state_global_step} | " + " | ".join(log_strs))

    return ctx.set(step_metrics_history, defaultdict(list))
