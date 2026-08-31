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

from slyme.context import Arg as SlymeArg
from slyme.context import Context, Ref, RefFactory

from deepslyme.context.metadata import ARG, Arg
from deepslyme.utils.config import parse_and_inject, resolve_args_from_refs


def test_metadata_and_argument_resolution_reuse_slyme() -> None:
    assert Arg is SlymeArg

    refs = RefFactory(
        {
            "model": {
                "lr": Ref(metadata={ARG: Arg(default=0.1, type=float)}),
            }
        }
    )

    assert resolve_args_from_refs([refs.model.lr]) == {
        "model.lr": Arg(default=0.1, type=float)
    }


def test_parse_and_inject_supports_schema_refs_and_inplace_context() -> None:
    refs = RefFactory(
        {
            "model": {
                "lr": Ref(metadata={ARG: Arg(default=0.1, type=float)}),
            }
        }
    )
    ctx = Context()

    result = parse_and_inject(
        context=ctx,
        cli_args=["--model.lr", "0.25"],
        extra_refs=[refs.model.lr],
    )

    assert result is ctx
    assert ctx.get(refs.model.lr) == 0.25
