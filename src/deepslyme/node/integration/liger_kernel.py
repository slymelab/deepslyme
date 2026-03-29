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

import logging
from typing import Any, Optional
from transformers import PreTrainedModel
from slyme.context import Context
from slyme.node import node, Auto

logger = logging.getLogger(__name__)


def _apply_liger_kernel(model: Any, liger_kwargs: Optional[dict] = None) -> None:
    """
    Core logic to apply Liger Kernel patches to the model.
    Extracted to separate purely functional logic from framework context management.
    """
    # Lazy import to avoid hard dependency
    try:
        from liger_kernel.transformers import _apply_liger_kernel_to_instance
    except ImportError:
        raise ImportError(
            "Liger Kernel is not installed. Please install it using "
            "`pip install liger-kernel` to use the `apply_liger_kernel` node."
        )

    # Resolve kwargs (empty dict if None)
    kwargs = liger_kwargs or {}

    # Target the base model if it's wrapped by PEFT
    if isinstance(model, PreTrainedModel):
        target_model = model
    elif hasattr(model, "get_base_model") and isinstance(
        model.get_base_model(), PreTrainedModel
    ):
        target_model = model.get_base_model()
    else:
        logger.warning(
            "The model is not an instance of PreTrainedModel (nor a PEFT wrapper). "
            "No Liger Kernels will be applied."
        )
        return

    # Apply the kernel patch
    _apply_liger_kernel_to_instance(model=target_model, **kwargs)

    logger.info(
        f"Successfully applied Liger Kernel to {target_model.__class__.__name__} "
        f"with configs: {kwargs}"
    )


@node
def apply_liger_kernel(
    ctx: Context,
    /,
    *,
    model: Auto[Any],
    liger_kwargs: Auto[Optional[dict]] = None,
) -> Context:
    """
    Apply Liger Kernel patches to the model for optimized operator performance
    (e.g., fused LayerNorm, fused CrossEntropy).

    This node uses lazy importing to ensure that the framework does not crash
    if the `liger-kernel` package is not installed in the user's environment.
    Supports both standard HuggingFace PreTrainedModels and PEFT models.
    """
    _apply_liger_kernel(model=model, liger_kwargs=liger_kwargs)
    return ctx
