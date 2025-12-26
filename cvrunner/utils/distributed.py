import os
import torch
import torch.distributed as dist
import datetime

def setup_distributed():
    """
    Initialize the distributed process group using environment variables 
    injected by 'torchrun'.
    
    Returns:
        tuple: (local_rank (int), device (torch.device))
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # We are running inside torchrun
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl", 
                init_method="env://", 
                world_size=world_size, 
                rank=rank,
                timeout=datetime.timedelta(seconds=3600)
            )
        
        # CRITICAL: Set the specific GPU for this process
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        return local_rank, device
    
    else:
        # Fallback for debugging (running script directly without torchrun)
        print("[cvrunner] No DDP environment detected. Running in standard mode.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, device

def cleanup_distributed():
    """Clean up the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()

# --- Existing Helper Functions (Unchanged) ---

def get_global_step(local_step: int) -> int:
    """
    Compute global step from local step across all ranks.
    """
    if not is_dist_avail_and_initialized():
        return local_step
        
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    global_step = local_step * world_size + rank
    return global_step

def is_available() -> bool:
    return dist.is_available()

def is_initialized() -> bool:
    return dist.is_initialized()

def is_dist_avail_and_initialized() -> bool:
    return is_available() and is_initialized()

def get_rank() -> int:
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def get_world_size() -> int:
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def is_main_process() -> bool:
    return get_rank() == 0

def barrier() -> None:
    if is_dist_avail_and_initialized():
        dist.barrier()

def reduce_dict(input_dict, average=True):
    """
    Reduces a dictionary of tensors/numbers across all processes.
    Useful for averaging loss across GPUs before logging.
    """
    if not is_dist_avail_and_initialized():
        return input_dict

    world_size = dist.get_world_size()
    if world_size < 2:
        return input_dict

    with torch.no_grad():
        names = []
        values = []
        # Sort keys to ensure order is identical across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        
        values = torch.stack([torch.tensor(v, dtype=torch.float32, device="cuda") for v in values])
        dist.all_reduce(values)
        
        if average:
            values /= world_size
            
        reduced_dict = {k: v.item() for k, v in zip(names, values)}
    
    return reduced_dict
