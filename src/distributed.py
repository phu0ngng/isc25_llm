import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from config import TrainingConfig


class DistributedSetup:
    @staticmethod
    def setup_distributed(hw_config):
        """Initialize distributed training based on hardware type"""
        if hw_config.device_type == ["cpu", "hip"]:
            backend = "gloo"  # CPU-based backend
        elif hw_config.device_type in ["cuda"]:
            backend = "nccl"  # GPU-optimized backend
        elif hw_config.device_type == "xpu":
            backend = "ccl"  # Intel-optimized backend

        if hw_config.num_devices > 1:
            dist.init_process_group(backend=backend, init_method="env://")

        return dist.get_rank() if dist.is_initialized() else 0

    @staticmethod
    def get_device(hw_config, local_rank=0):
        """Get appropriate device based on hardware configuration"""
        if hw_config.device_type == "cpu":
            device = torch.device("cpu")
        elif hw_config.device_type in ["cuda", "hip", "xpu"]:
            device = torch.device(f"{hw_config.device_type}:{local_rank}")
        else:
            raise Exception("Invalid device_type - {device_type}")
        return device

    @staticmethod
    def cleanup_distributed():
        """
        Clean up the distributed training environment
        """
        if dist.is_initialized():
            dist.destroy_process_group()

    @staticmethod
    def is_main_process(rank):
        """
        Check if current process is the main process
        """
        return rank == 0

    @staticmethod
    def barrier():
        """
        Synchronization point for all processes
        """
        if dist.is_initialized():
            dist.barrier()

    @staticmethod
    def reduce_dict(input_dict, average=True):
        """
        Reduce dictionary values across all processes
        """
        if not dist.is_initialized():
            return input_dict

        world_size = dist.get_world_size()
        if world_size < 2:
            return input_dict

        with torch.no_grad():
            names = []
            values = []
            for k in sorted(input_dict.keys()):
                names.append(k)
                values.append(input_dict[k])

            values = torch.stack(values, dim=0)
            dist.all_reduce(values)
            if average:
                values /= world_size

            reduced_dict = dict(zip(names, values))
            return reduced_dict

    @staticmethod
    def all_gather(data):
        """
        Gather data from all processes
        """
        if not dist.is_initialized():
            return [data]

        world_size = dist.get_world_size()
        if world_size < 2:
            return [data]

        tensor = torch.tensor(data).cuda()
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor)
        return [t.item() for t in gathered]

    @staticmethod
    def wrap_model_for_distributed(
        model: nn.Module, config: TrainingConfig, local_rank: int
    ):
        """
        Wrap model based on specified parallel strategy
        """
        if dist.is_initialized():
            model = model.to(DistributedSetup.get_device(config.hardware, local_rank))
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
            )
        return model
