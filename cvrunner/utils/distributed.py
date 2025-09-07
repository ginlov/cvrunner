import torch.distributed as dist

def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def is_main_process():
    return get_rank() == 0


def barrier():
    """Synchronize all processes."""
    if is_dist_avail_and_initialized():
        dist.barrier()


def all_reduce_tensor(tensor, op=dist.ReduceOp.SUM):
    """
    All-reduce a tensor across processes.
    Default = SUM. Divide by world_size manually if you want an average.
    """
    if not is_dist_avail_and_initialized():
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=op)
    return tensor
