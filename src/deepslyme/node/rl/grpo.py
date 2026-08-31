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

import torch
from slyme.context import Context, Ref
from slyme.node import Auto, node


@node
def calc_grpo_advantages(
    ctx: Context,
    /,
    *,
    rewards: Auto[torch.Tensor],
    group_size: Auto[int],
    advantages: Ref[torch.Tensor],
    grpo_adv_eps: Auto[float] = 1e-5,
) -> None:
    rewards_view = rewards.view(-1, group_size)
    mean = rewards_view.mean(dim=1, keepdim=True)
    std = rewards_view.std(dim=1, keepdim=True)
    adv_val = (rewards_view - mean) / (std + grpo_adv_eps)
    adv_val = torch.where(std > grpo_adv_eps, adv_val, torch.zeros_like(adv_val))
    ctx.set(advantages, adv_val.view(-1))
