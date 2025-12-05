import torch.distributed as dist
import torch


def get_global_step(local_step: int) -> int:
    """
    Compute global step from local step across all ranks.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    global_step = local_step * world_size + rank
    return global_step

def is_available() -> bool:
    """
    Check if distributed is available

    Returns:
        bool:
    """
    return dist.is_available()


def is_initialized() -> bool:
    """
    Check if distributed is initialized

    Returns:
        bool:
    """
    return dist.is_initialized()


def is_dist_avail_and_initialized() -> bool:
    """
    Check distributed status

    Returns:
        bool: 
    """
    return is_available() and is_initialized()


def get_rank() -> int:
    """
    Get current rank

    Returns:
        int:
    """
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """
    Get worl size

    Returns:
        int:
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def is_main_process() -> bool:
    """
    Check whether is main process

    Returns:
        bool:
    """
    return get_rank() == 0


def barrier() -> None:
    """Synchronize all processes."""
    if is_dist_avail_and_initialized():
        dist.barrier()


def all_reduce_tensor(tensor, op=dist.ReduceOp.SUM) -> torch.tensor:
    """
    All-reduce a tensor across processes.
    Default = SUM. Divide by world_size manually if you want an average.
    """
    if not is_dist_avail_and_initialized():
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=op)
    return tensor
