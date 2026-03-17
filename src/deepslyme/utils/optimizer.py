import torch
from slyme.utils.registry import Registry

OPTIMIZER_REGISTRY = Registry[type[torch.optim.Optimizer]]("optimizer")
OPTIMIZER_REGISTRY.register(torch.optim.AdamW, key="adamw")
