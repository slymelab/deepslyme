import torch
from slyme.context import Context, Ref
from slyme.node import node, Auto


@node
def calc_grpo_advantages(
    ctx: Context,
    /,
    *,
    rewards: Auto[torch.Tensor],
    group_size: Auto[int],
    advantages: Ref[torch.Tensor],
    grpo_adv_eps: Auto[float] = 1e-5,
) -> Context:
    rewards_view = rewards.view(-1, group_size)
    mean = rewards_view.mean(dim=1, keepdim=True)
    std = rewards_view.std(dim=1, keepdim=True)
    adv_val = (rewards_view - mean) / (std + grpo_adv_eps)
    adv_val = torch.where(std > grpo_adv_eps, adv_val, torch.zeros_like(adv_val))
    return ctx.set(advantages, adv_val.view(-1))
