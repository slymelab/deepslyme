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

from collections.abc import Mapping
from typing import Any

import ray
from slyme.context import Context, RefLike
from slyme.node import Auto, Node, node


@ray.remote
class SlymeRayActor:
    def __init__(
        self,
        ctx: Context,
        init_node: Node | None,
        exec_node: Node,
    ):
        self.ctx = ctx

        if init_node is not None:
            init_node(self.ctx)

        self.exec_node = exec_node

    def exec(
        self,
        inputs: Mapping[RefLike, Any],
        output_refs: list[RefLike],
    ) -> Any:
        self.ctx.update(inputs)
        self.exec_node(self.ctx)
        return self.ctx.extract(output_refs)


@node
def ray_node(
    ctx: Context,
    /,
    *,
    actor_handle: Any,
    inputs: Auto[dict],
    outputs: Mapping[RefLike, RefLike],
    resolve_outputs: bool = False,
) -> None:
    num_outs = len(outputs)

    if num_outs == 0:
        actor_handle.exec.remote(inputs, [])
        return

    output_refs_list = list(outputs.values())
    result_refs = actor_handle.exec.options(num_returns=num_outs).remote(
        inputs, output_refs_list
    )

    if num_outs == 1:
        result_refs = [result_refs]
    else:
        result_refs = list(result_refs)

    if resolve_outputs:
        result_values = ray.get(result_refs)
    else:
        result_values = result_refs

    update_dict: dict[RefLike, Any] = {
        local_key: val for local_key, val in zip(outputs.keys(), result_values)
    }
    ctx.update(update_dict)
