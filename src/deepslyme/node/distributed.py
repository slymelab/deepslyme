import torch
import torch.distributed as dist
from deepslyme.utils.accelerator import current_device_name, device_count


class DistributedState:
    """Metadata cache for the distributed environment."""

    def __init__(self):
        self.process_index = dist.get_rank() if dist.is_initialized() else 0
        self.num_processes = dist.get_world_size() if dist.is_initialized() else 1
        self.is_main_process = self.process_index == 0
        self.device = torch.device(
            current_device_name(), self.process_index % max(1, device_count())
        )
