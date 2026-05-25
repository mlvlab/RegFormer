"""
Distributed Data Parallel (DDP) utilities for weak HOI training
"""

import os
import torch
import torch.distributed as dist
import numpy as np
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_ddp(args):
    """
    Initialize distributed training
    
    Args:
        args: Arguments containing DDP configuration
    
    Returns:
        Updated args with proper rank/world_size settings
    """
    if not args.use_ddp:
        return args
    
    # Get environment variables set by torchrun
    if 'RANK' in os.environ:
        args.rank = int(os.environ['RANK'])
    if 'WORLD_SIZE' in os.environ:
        args.world_size = int(os.environ['WORLD_SIZE'])
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    
    # Initialize the process group
    if not dist.is_initialized():
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank
        )
    
    # Set the device for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
        args.device = f'cuda:{args.local_rank}'
    else:
        args.device = 'cpu'
    
    print(f"[Rank {args.rank}/{args.world_size}] Initialized DDP on device {args.device}")
    
    return args


def cleanup_ddp():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(args):
    """Check if this is the main process (rank 0)"""
    if not args.use_ddp:
        return True
    return args.rank == 0


def wrap_model_ddp(model, args):
    """
    Wrap model with DistributedDataParallel
    
    Args:
        model: Model to wrap
        args: Arguments containing DDP configuration
    
    Returns:
        Wrapped model
    """
    if not args.use_ddp:
        return model
    
    # Ensure model is on the correct device before wrapping
    model = model.to(args.device)
    
    # Wrap with DDP
    model = DDP(
        model,
        device_ids=[args.local_rank] if torch.cuda.is_available() else None,
        output_device=args.local_rank if torch.cuda.is_available() else None,
        find_unused_parameters=False  # Set to True if you have unused parameters
    )
    
    print(f"[Rank {args.rank}] Model wrapped with DDP")
    
    return model


def create_distributed_sampler(dataset, args, shuffle=True):
    """
    Create distributed sampler for dataset
    
    Args:
        dataset: Dataset to sample from
        args: Arguments containing DDP configuration
        shuffle: Whether to shuffle the dataset
    
    Returns:
        Sampler for the dataset
    """
    if not args.use_ddp:
        return None
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=shuffle,
        drop_last=True  # Ensure all processes have the same number of batches
    )
    
    return sampler


def reduce_tensor(tensor, args, average=True):
    """
    Reduce tensor across all processes
    
    Args:
        tensor: Tensor to reduce
        args: Arguments containing DDP configuration
        average: Whether to average the tensor across processes
    
    Returns:
        Reduced tensor
    """
    if not args.use_ddp:
        return tensor
    
    if not dist.is_initialized():
        return tensor
    
    # Clone tensor to avoid modifying the original
    reduced_tensor = tensor.clone()
    
    # All-reduce the tensor
    dist.all_reduce(reduced_tensor, op=dist.ReduceOp.SUM)
    
    # Average if requested
    if average:
        reduced_tensor /= args.world_size
    
    return reduced_tensor


def reduce_dict(input_dict, args, average=True):
    """
    Reduce dictionary of tensors across all processes
    
    Args:
        input_dict: Dictionary of tensors to reduce
        args: Arguments containing DDP configuration
        average: Whether to average the tensors across processes
    
    Returns:
        Dictionary with reduced tensors
    """
    if not args.use_ddp:
        return input_dict
    
    reduced_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, torch.Tensor):
            reduced_dict[key] = reduce_tensor(value, args, average)
        elif isinstance(value, (int, float)):
            # Convert to tensor, reduce, then convert back
            tensor_value = torch.tensor(value, dtype=torch.float32, device=args.device)
            reduced_tensor = reduce_tensor(tensor_value, args, average)
            reduced_dict[key] = reduced_tensor.item()
        else:
            # Keep non-tensor values as is
            reduced_dict[key] = value
    
    return reduced_dict


def gather_tensors(tensor, args):
    """
    Gather tensors from all processes
    
    Args:
        tensor: Tensor to gather from current process
        args: Arguments containing DDP configuration
    
    Returns:
        gathered_tensors: List of tensors from all processes (only on rank 0)
                         None on other processes
    """
    if not args.use_ddp or not dist.is_initialized():
        return [tensor]
    
    # Only gather on main process to save memory
    if is_main_process(args):
        # Prepare list to collect tensors from all processes
        gathered_tensors = [torch.zeros_like(tensor) for _ in range(args.world_size)]
        dist.gather(tensor, gathered_tensors, dst=0)
        return gathered_tensors
    else:
        # Other processes just send their tensor
        dist.gather(tensor, dst=0)
        return None


def gather_numpy_arrays(array, args):
    """
    Gather numpy arrays from all processes
    
    Args:
        array: Numpy array to gather from current process
        args: Arguments containing DDP configuration
    
    Returns:
        gathered_arrays: Concatenated numpy array from all processes (only on rank 0)
                        None on other processes
    """
    if not args.use_ddp or not dist.is_initialized():
        return array
    
    # Convert to tensor
    device = torch.device(args.device if hasattr(args, 'device') else 'cuda' if torch.cuda.is_available() else 'cpu')
    tensor = torch.from_numpy(array).to(device)
    
    # Gather tensors
    gathered_tensors = gather_tensors(tensor, args)
    
    if gathered_tensors is not None:
        # Convert back to numpy and concatenate
        gathered_arrays = [t.cpu().numpy() for t in gathered_tensors]
        return np.concatenate(gathered_arrays, axis=0)
    else:
        return None


def barrier(args):
    """
    Synchronization barrier for all processes
    
    Args:
        args: Arguments containing DDP configuration
    """
    if args.use_ddp and dist.is_initialized():
        dist.barrier()


def save_checkpoint_ddp(model, optimizer, scheduler, epoch, loss, args, filename, metrics=None):
    """
    Save checkpoint only from main process
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        scheduler: Scheduler to save
        epoch: Current epoch
        loss: Current loss
        args: Arguments
        filename: Filename to save to
        metrics: Optional metrics dictionary to save
    """
    if not is_main_process(args):
        return
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create full path for checkpoint
    checkpoint_path = os.path.join(args.output_dir, filename)
    
    # Get model state dict (unwrap DDP if necessary) - only trainable parameters
    model_to_save = model.module if hasattr(model, 'module') else model
    
    # Filter to only include parameters that require gradients (trainable parameters)
    trainable_state_dict = {
        name: param for name, param in model_to_save.state_dict().items()
        if any(p.requires_grad for n,p in model_to_save.named_parameters() 
               if n == name)
    }
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': trainable_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        # 'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'metrics': metrics,
        'args': args
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"[Main Process] Checkpoint saved to {checkpoint_path} (trainable parameters only)")


def save_results_json(results, args, filename='results.json'):
    """
    Save results dictionary to JSON file
    
    Args:
        results: Dictionary containing results to save
        args: Arguments
        filename: JSON filename to save to
    """
    if not is_main_process(args):
        return
    
    import json
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create full path for results
    results_path = os.path.join(args.output_dir, filename)
    
    # Convert any tensor values to float for JSON serialization
    def convert_tensors(obj):
        import torch
        import numpy as np
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
            return float(obj) if isinstance(obj, (np.float32, np.float64)) else int(obj)
        elif isinstance(obj, dict):
            return {k: convert_tensors(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_tensors(v) for v in obj]
        else:
            return obj
    
    # Convert results to JSON-serializable format
    json_results = convert_tensors(results)
    
    # Save results to JSON
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"[Main Process] Results saved to {results_path}")


def load_checkpoint_ddp(model, optimizer, scheduler, args, filename):
    """
    Load checkpoint for distributed training
    
    Args:
        model: Model to load
        optimizer: Optimizer to load
        scheduler: Scheduler to load
        args: Arguments
        filename: Filename to load from
    
    Returns:
        Loaded epoch and loss
    """
    if not os.path.exists(filename):
        print(f"Checkpoint {filename} not found")
        return 0, float('inf')
    
    # Load checkpoint
    checkpoint = torch.load(filename, map_location=args.device)
    
    # Load model state dict (handle DDP wrapping)
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer and scheduler
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"[Rank {args.rank}] Checkpoint loaded from {filename} (epoch {epoch})")
    
    return epoch, loss


def print_ddp(message, args):
    """
    Print message only from main process
    
    Args:
        message: Message to print
        args: Arguments containing DDP configuration
    """
    if is_main_process(args):
        print(message)


def get_model_for_inference(model):
    """
    Get model for inference (unwrap DDP if necessary)
    
    Args:
        model: Model (potentially wrapped with DDP)
    
    Returns:
        Unwrapped model
    """
    if hasattr(model, 'module'):
        return model.module
    return model