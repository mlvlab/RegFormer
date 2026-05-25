import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from PIL import Image

import numpy as np
# Wandb import (optional)
import wandb
WANDB_AVAILABLE = True

from core.arguments import weak_hoi_args
from core.engine import train_one_epoch, evaluate
from core.loss import get_loss_function
from core.ddp_utils import (
    setup_ddp, cleanup_ddp, is_main_process, wrap_model_ddp,
    create_distributed_sampler, save_checkpoint_ddp, save_results_json, print_ddp
)
from models.model import VisionTextAlignmentModel
from utils.data_utils import WeakDATA
from utils.label_utils import (
    get_class_labels, print_label_info, verb_to_hoi_dict, hoi_to_verb_list,
    get_verb_object_indices
)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import math
import random


def set_seed(seed):
    """
    Set random seed for reproducibility across all libraries
    
    Args:
        seed (int): Random seed value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    
    print(f"Random seed set to: {seed}")


class WarmupCosineAnnealingLR:
    """
    Custom learning rate scheduler with warmup and cosine annealing
    Supports iteration-based or epoch-based stepping
    """
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.01, warmup_start_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.warmup_start_lr = warmup_start_lr
        self.current_step = 0
        
        # Store initial learning rates for each parameter group
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
    def step(self):
        """Update learning rates for all parameter groups"""
        self.current_step += 1
        
        for i, param_group in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[i]
            
            if self.current_step <= self.warmup_steps:
                # Warmup phase: linear increase from warmup_start_lr to base_lr
                lr = self.warmup_start_lr + (base_lr - self.warmup_start_lr) * (self.current_step / self.warmup_steps)
            else:
                # Cosine annealing phase
                cosine_steps = self.current_step - self.warmup_steps
                cosine_total_steps = self.total_steps - self.warmup_steps
                min_lr = base_lr * self.min_lr_ratio
                
                lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * cosine_steps / cosine_total_steps))
            
            param_group['lr'] = lr
    
    def get_last_lr(self):
        """Return current learning rates for all parameter groups"""
        return [group['lr'] for group in self.optimizer.param_groups]


def create_scheduler(optimizer, args, total_steps_per_epoch):
    """
    Create learning rate scheduler with warmup and cosine annealing
    
    Args:
        optimizer: The optimizer
        args: Arguments containing scheduler settings
        total_steps_per_epoch: Number of optimizer steps per epoch (considering gradient accumulation)
        
    Returns:
        scheduler: Configured scheduler or None if not using scheduler
    """
    if not args.use_cosine_scheduler:
        return None
    
    # Calculate total steps based on scheduler_step_type
    if args.scheduler_step_type == 'iteration':
        warmup_steps = int(args.warmup_epochs * total_steps_per_epoch)
        total_steps = args.num_epochs * total_steps_per_epoch
    else:  # epoch
        warmup_steps = int(args.warmup_epochs)
        total_steps = args.num_epochs
    
    scheduler = WarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=args.min_lr_ratio,
        warmup_start_lr=args.warmup_start_lr
    )
    
    print_ddp(f"Created {args.scheduler_step_type}-based scheduler:", args)
    print_ddp(f"  Warmup steps: {warmup_steps}", args)
    print_ddp(f"  Total steps: {total_steps}", args)
    print_ddp(f"  Min LR ratio: {args.min_lr_ratio}", args)
    print_ddp(f"  Warmup start LR: {args.warmup_start_lr}", args)
    
    return scheduler


def create_optimizer_with_different_lrs(model, args):
    """
    Create optimizer with different learning rates for different model components
    
    Args:
        model: The model (potentially wrapped with DDP)
        args: Arguments containing learning rate settings
    
    Returns:
        optimizer: AdamW optimizer with parameter groups
    """
    # Get base model (unwrap DDP if necessary)
    base_model = model.module if hasattr(model, 'module') else model
    
    # Set learning rates (use default lr if specific lr not provided)
    vision_lr = args.vision_lr if args.vision_lr is not None else args.lr
    text_lr = args.text_lr if args.text_lr is not None else args.lr
    projector_lr = args.projector_lr if args.projector_lr is not None else args.lr
    
    # Collect parameters for different components
    param_groups = []    
    
    # Vision encoder parameters (only trainable ones)
    if hasattr(base_model, 'vision_model'):
        vision_params = [p for p in base_model.vision_model.parameters() if p.requires_grad]
        if vision_params:
            param_groups.append({
                'params': vision_params,
                'lr': vision_lr,
                'name': 'vision_encoder'
            })
            total_vision = len(list(base_model.vision_model.parameters()))
            print_ddp(f"Vision encoder: {len(vision_params)}/{total_vision} trainable parameter tensors, lr={vision_lr}", args)
        else:
            print_ddp(f"Vision encoder: 0 trainable parameters (all frozen)", args)
    
    # Text encoder parameters (only trainable ones)
    if hasattr(base_model, 'text_model'):
        text_params = [p for p in base_model.text_model.parameters() if p.requires_grad]
        if text_params:
            param_groups.append({
                'params': text_params,
                'lr': text_lr,
                'name': 'text_encoder'
            })
            total_text = len(list(base_model.text_model.parameters()))
            print_ddp(f"Text encoder: {len(text_params)}/{total_text} trainable parameter tensors, lr={text_lr}", args)
        else:
            print_ddp(f"Text encoder: 0 trainable parameters (all frozen)", args)
    
    # Projector parameters (only trainable ones)
    projector_params = []
    if hasattr(base_model, 'vision_projector'):
        projector_params.extend([p for p in base_model.vision_projector.parameters() if p.requires_grad])
    if hasattr(base_model, 'text_projector'):
        projector_params.extend([p for p in base_model.text_projector.parameters() if p.requires_grad])
    
    if projector_params:
        param_groups.append({
            'params': projector_params,
            'lr': projector_lr,
            'name': 'projectors'
        })
        print_ddp(f"Projectors: {len(projector_params)} trainable parameter tensors, lr={projector_lr}", args)
    
    # Other parameters (scale_logit, scale_bias, etc.) - only trainable ones
    other_params = []
    for name, param in base_model.named_parameters():
        # Skip parameters already included in other groups
        is_vision = 'vision_model' in name
        is_text = 'text_model' in name
        is_projector = 'vision_projector' in name or 'text_projector' in name
        if args.zero_wd_for_scaler:
            if 'scale_logit' in name or 'scale_bias' in name:
                continue
        
        if not (is_vision or is_text or is_projector) and param.requires_grad:
            other_params.append(param)
    
    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': args.lr,
            'name': 'other'
        })
        print_ddp(f"Other parameters: {len(other_params)} trainable parameter tensors, lr={args.lr}", args)
    
    if args.zero_wd_for_scaler:
        print_ddp(f"Zero weight decay for logit_scale and bias", args)
        scaler_params = []
        for name, param in base_model.named_parameters():
            if ('scale_logit' in name or 'scale_bias' in name) and param.requires_grad:
                scaler_params.append(param)
        if scaler_params:
            param_groups.append({
                'params': scaler_params,
                'lr': args.lr,
                'weight_decay': 0.0,
                'name': 'scaler'
            })
            print_ddp(f"Scaler parameters: {len(scaler_params)} trainable parameter tensors, lr={args.lr}, weight_decay=0.0", args)
    
    # Create optimizer
    optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)
    
    # Print parameter group summary
    total_trainable_params = sum(len(group['params']) for group in param_groups)
    total_model_params = len(list(base_model.parameters()))
    print_ddp(f"Created optimizer with {len(param_groups)} parameter groups", args)
    print_ddp(f"Total trainable parameters: {total_trainable_params}/{total_model_params} parameter tensors", args)
    
    return optimizer


def setup_wandb(args):
    """
    Initialize Weights & Biases tracking
    
    Args:
        args: Arguments containing wandb configuration
    
    Returns:
        bool: True if wandb is initialized successfully, False otherwise
    """
    if not args.use_wandb or not WANDB_AVAILABLE:
        if args.use_wandb and not WANDB_AVAILABLE:
            print("Warning: wandb requested but not available. Install with: pip install wandb")
        return False
    
    # Only initialize wandb on main process in DDP
    if not is_main_process(args):
        return False
    
    try:
        # Create run name if not provided
        if args.wandb_name is None:
            run_name = f"{args.dataset_name}_{args.target_type}_{args.vision_encoder.split('/')[-1]}"
            if args.vision_projector_type != args.projector_type:
                run_name += f"_v{args.vision_projector_type}_t{args.text_projector_type}"
            else:
                run_name += f"_{args.projector_type}"
        else:
            run_name = args.wandb_name
        
        # Initialize wandb
        wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            id=args.wandb_id,
            entity=args.wandb_entity,
            name=run_name,
            tags=args.wandb_tags,
            notes=args.wandb_notes,
            config=args
        )
        
        print_ddp(f"Wandb initialized: {wandb.run.url}", args)
        return True
        
    except Exception as e:
        print(f"Failed to initialize wandb: {e}")
        return False


def get_transforms_from_model(model, is_train=True, args=None):
    """Extract transforms from the model's vision processor"""
    # try:
    # Get vision processor from model
    vision_processor = model.vision_processor
    
    # Create transforms based on processor configuration
    if hasattr(vision_processor, 'image_processor'):
        # For newer CLIP versions
        processor = vision_processor.image_processor
    else:
        # For older CLIP versions or other processors
        processor = vision_processor
    
    # Get image size and normalization parameters
    # if hasattr(processor, 'size'):
    #     if isinstance(processor.size, dict):
    #         image_size = (processor.size.get('height', 224), processor.size.get('width', 224))
    #     else:
    #         image_size = (processor.size, processor.size)
    # else:
    #     image_size = (224, 224)
    image_size = (args.input_resolution, args.input_resolution)
    
    # Get normalization parameters
    if hasattr(processor, 'image_mean') and hasattr(processor, 'image_std'):
        mean = processor.image_mean
        std = processor.image_std
    elif isinstance(processor, T.Compose):
        # Extract normalization from existing transforms
        for transform in processor.transforms:
            if isinstance(transform, T.Normalize):
                mean = list(transform.mean)
                std = list(transform.std)
                # break
    else:
        # Default CLIP normalization
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
    
    # Create transforms
    if is_train:
        transforms = T.Compose([
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.4, 0.4, 0.4),
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
    else:
        transforms = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
    
    print(f"Created transforms with image_size={image_size}, mean={mean}, std={std}")
    return transforms
        
    # except Exception as e:
    #     print(f"Error extracting transforms from model: {e}")
    #     print("Using default CLIP transforms")
    #     # Default CLIP transforms
    #     return T.Compose([
    #         T.Resize((224, 224)),
    #         T.ToTensor(),
    #         T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
    #                    std=[0.26862954, 0.26130258, 0.27577711])
    #     ])


def create_data_loaders(args, train_transforms, eval_transforms):
    """Create train and validation data loaders with distributed support"""
    
    # Handle backward compatibility for legacy arguments
    train_data_path = args.train_data_path or args.data_path
    train_image_root = args.train_image_root or args.image_root
    val_data_path = args.val_data_path or args.data_path
    val_image_root = args.val_image_root or args.image_root
    
    if not train_data_path or not train_image_root:
        raise ValueError("Must provide either --train_data_path/--train_image_root or legacy --data_path/--image_root")
    
    # Create train dataset arguments
    train_args = type('Args', (), {})()
    for attr in dir(args):
        if not attr.startswith('_'):
            setattr(train_args, attr, getattr(args, attr))
    train_args.data_path = train_data_path
    train_args.image_root = train_image_root
    
    # Create validation dataset arguments
    val_args = type('Args', (), {})()
    for attr in dir(args):
        if not attr.startswith('_'):
            setattr(val_args, attr, getattr(args, attr))
    val_args.data_path = val_data_path
    val_args.image_root = val_image_root
    
    # Create separate train and validation datasets
    print_ddp(f"Creating train dataset from: {train_data_path}", args)
    train_dataset = WeakDATA(train_args, transforms=train_transforms, is_train=True)
    
    print_ddp(f"Creating validation dataset from: {val_data_path}", args)
    val_dataset = WeakDATA(val_args, transforms=eval_transforms, is_train=False)
    
    # Create distributed samplers if using DDP
    train_sampler = create_distributed_sampler(train_dataset, args, shuffle=True)
    val_sampler = create_distributed_sampler(val_dataset, args, shuffle=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),  # Don't shuffle if using distributed sampler
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True if 'cuda' in args.device else False,
        drop_last=True  # Ensure consistent batch sizes across processes
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True if 'cuda' in args.device else False,
        drop_last=False
    )
    
    print_ddp(f"Created train loader with {len(train_loader)} batches ({len(train_dataset)} samples)", args)
    print_ddp(f"Created val loader with {len(val_loader)} batches ({len(val_dataset)} samples)", args)
    
    return train_loader, val_loader, train_sampler, val_sampler


def main():
    # Parse arguments
    parser = weak_hoi_args()
    args = parser.parse_args()
    
    if args.use_seperate_classifier:
        assert args.target_type == 'hoi', "use_seperate_classifier only support for target type with 'hoi'"
        assert args.use_text_embeddings, "currently only support for pre-extracted text embeddings"
    
    if args.eval_only:
        finetuned_ckpt = args.finetuned_ckpt
        compute_latency = args.compute_latency
        use_union_cropped_image = args.use_union_cropped_image
        ckpt = torch.load(finetuned_ckpt, map_location='cpu')
        return_attention_weight = args.return_attention_weight
        vis_label_weight = args.vis_label_weight
        saved_args = ckpt['args']
        for key, value in vars(saved_args).items():
            setattr(args, key, value)
        args.eval_only = True
        args.finetuned_ckpt = finetuned_ckpt
        args.return_attention_weight = return_attention_weight
        args.vis_label_weight = vis_label_weight
        args.use_ddp = False # DDP not supported for evaluation
        args.use_wandb = False # wandb not supported for evaluation
        args.compute_latency = compute_latency
        args.use_union_cropped_image = use_union_cropped_image
    
    # Setup distributed training
    args = setup_ddp(args)
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Calculate per-GPU batch size from global batch size
    world_size = getattr(args, 'world_size', 1)
    gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
    
    if args.batch_size is None:
        # Auto-calculate per-GPU batch size
        total_devices = world_size * gradient_accumulation_steps
        if args.global_batch_size % total_devices != 0:
            print_ddp(f"Warning: global_batch_size ({args.global_batch_size}) is not divisible by "
                     f"(world_size * gradient_accumulation_steps) = {total_devices}", args)
            print_ddp(f"Using floor division: {args.global_batch_size // total_devices}", args)
        
        args.batch_size = args.global_batch_size // total_devices
        
        # Ensure minimum batch size of 1
        if args.batch_size < 1:
            args.batch_size = 1
            print_ddp(f"Warning: Calculated batch_size < 1. Setting to 1.", args)
    else:
        # Calculate actual global batch size based on provided batch_size
        actual_global_batch_size = args.batch_size * world_size * gradient_accumulation_steps
        if actual_global_batch_size != args.global_batch_size:
            print_ddp(f"Warning: Provided batch_size ({args.batch_size}) results in "
                     f"actual global batch size of {actual_global_batch_size}, not {args.global_batch_size}", args)
    
    # Print batch size information
    effective_batch_size = args.batch_size * gradient_accumulation_steps
    actual_global_batch_size = effective_batch_size * world_size
    
    print_ddp(f"Batch size configuration:", args)
    print_ddp(f"  Global batch size: {args.global_batch_size} (target) / {actual_global_batch_size} (actual)", args)
    print_ddp(f"  Per-GPU batch size: {args.batch_size}", args)
    print_ddp(f"  Gradient accumulation steps: {gradient_accumulation_steps}", args)
    print_ddp(f"  Effective batch size per GPU: {effective_batch_size}", args)
    print_ddp(f"  World size (num GPUs): {world_size}", args)
    
    try:
        # Load class labels and auto-detect num_classes if not specified
        print_ddp(f"Loading {args.dataset_name} {args.target_type} labels...", args)
        class_labels, auto_verb_num_classes = get_class_labels(args.dataset_name, 'verb')
        class_labels, auto_hoi_num_classes = get_class_labels(args.dataset_name, 'hoi')
        verb_object_indices = get_verb_object_indices(args.dataset_name)
        verb_to_hoi = verb_to_hoi_dict(args.dataset_name)
        hoi_to_verb = hoi_to_verb_list(args.dataset_name)
        args.verb_to_hoi = verb_to_hoi
        args.hoi_to_verb = hoi_to_verb
        
        # Set num_classes if not specified
        # if args.num_classes is None:
        #     args.num_classes = auto_num_classes
        #     print_ddp(f"Auto-detected num_classes: {args.num_classes}", args)
        # elif args.num_classes != auto_num_classes:
        #     print_ddp(f"Warning: Specified num_classes ({args.num_classes}) differs from dataset ({auto_num_classes})", args)
        #     print_ddp(f"Using dataset num_classes: {auto_num_classes}", args)
        #     args.num_classes = auto_num_classes
        
        args.verb_num_classes = auto_verb_num_classes
        print_ddp(f"Auto-detected verb num_classes for {args.dataset_name}: {args.verb_num_classes}", args)
        args.hoi_num_classes = auto_hoi_num_classes
        print_ddp(f"Auto-detected hoi num_classes for {args.dataset_name}: {args.hoi_num_classes}", args)
        
        # Print label information (only from main process)
        if is_main_process(args):
            print_label_info(args.dataset_name, args.target_type)
        
        # Set device
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        print_ddp(f"Using device: {device}", args)
        
        # Create model
        print_ddp("Creating vision-text alignment model...", args)
        model = VisionTextAlignmentModel(verb_object_indices, args)
        
                # Load checkpoint for initialization if provided (for training)
        if not args.eval_only and args.finetuned_ckpt:
            print_ddp(f"Loading checkpoint for initialization from {args.finetuned_ckpt}...", args)
            ckpt = torch.load(args.finetuned_ckpt, map_location='cpu')
            model_state_dict = ckpt['model_state_dict']
            model_dict = model.state_dict()
            
            # Filter out keys that don't exist in the model
            for key, value in model_dict.items():
                if key in model_state_dict:
                    print(f"{key}")
                    model_dict[key] = model_state_dict[key]
            
            # Load the filtered state dict
            model.load_state_dict(model_dict)
            print_ddp(f"Loaded checkpoint from {args.finetuned_ckpt}", args)
        
        model = model.to(device)
        
        # Wrap model with DDP if using distributed training
        model = wrap_model_ddp(model, args)
        
        # Initialize wandb and track model
        wandb_active = setup_wandb(args)
        if wandb_active:
            # Get base model for wandb tracking (unwrapped DDP)
            base_model_for_wandb = model.module if hasattr(model, 'module') else model
            wandb.watch(base_model_for_wandb, log_freq=args.wandb_watch_freq, log_graph=True)
            print_ddp("Model tracking with wandb initialized", args)
    
        # Get transforms from model's vision processor
        print_ddp("Extracting transforms from model...", args)
        # Get base model for transform extraction (unwrap DDP if necessary)
        base_model_for_transforms = model
        if hasattr(model, 'module'):
            base_model_for_transforms = model.module
        
        train_transforms = get_transforms_from_model(base_model_for_transforms, is_train=True, args=args)
        eval_transforms = get_transforms_from_model(base_model_for_transforms, is_train=False, args=args)
        
        # Create data loaders
        print_ddp("Creating data loaders...", args)
        train_loader, val_loader, train_sampler, val_sampler = create_data_loaders(args, train_transforms, eval_transforms)
    
        # Create loss function
        print_ddp(f"Creating loss function: {args.loss_type}", args)
        class_stats = train_loader.dataset.get_class_statistics()
        # label_mask = train_loader.dataset.label_mask
        criterion = get_loss_function(class_stats['class_counts'], args)
        
        # Create optimizer with different learning rates
        print_ddp("Creating optimizer with component-specific learning rates...", args)
        optimizer = create_optimizer_with_different_lrs(model, args)
        
        # Create learning rate scheduler with warmup and cosine annealing
        # Calculate effective steps per epoch considering gradient accumulation
        raw_steps_per_epoch = len(train_loader)
        gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
        
        # Effective steps per epoch = how many times optimizer.step() is called per epoch
        effective_steps_per_epoch = math.ceil(raw_steps_per_epoch / gradient_accumulation_steps)
        
        print_ddp(f"Scheduler steps calculation:", args)
        print_ddp(f"  Raw batches per epoch: {raw_steps_per_epoch}", args)
        print_ddp(f"  Gradient accumulation steps: {gradient_accumulation_steps}", args)
        print_ddp(f"  Effective optimizer steps per epoch: {effective_steps_per_epoch}", args)
        
        scheduler = create_scheduler(optimizer, args, effective_steps_per_epoch)
        
        # Print model info (only from main process)
        if is_main_process(args):
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
        
        # Run initial evaluation if checkpoint was loaded
        if not args.eval_only and args.finetuned_ckpt:
            print_ddp("Running initial evaluation to verify checkpoint loading...", args)
            val_loss, val_metrics = evaluate(
                model, val_loader, criterion, device, args, class_labels
            )
            print_ddp("Initial evaluation results:", args)
            if is_main_process(args):
                print(f"Initial Val Loss: {val_loss:.4f}")
                if val_metrics:
                    for metric_name, value in val_metrics.items():
                        if not isinstance(value, list):
                            print(f"Initial Val {metric_name}: {value:.4f}")
            print_ddp("-" * 50, args)
        
        # Training loop
        print_ddp(f"Starting training for {args.num_epochs} epochs...", args)
        best_val_loss = float('inf')
        best_val_metrics = None
        all_results = []  # Store all epoch results
    
        if args.eval_only:
            model_state_dict = ckpt['model_state_dict']
            model_dict = model.state_dict()
            
            # Filter out keys that don't exist in the model
            for key, value in model_dict.items():
                if key in model_state_dict:
                    model_dict[key] = model_state_dict[key]
            
            # Load the filtered state dict
            model.load_state_dict(model_dict)
            print_ddp(f"Loaded finetuned model from {finetuned_ckpt}", args)
            
            # compute latency
            # for image-level hoi reasoning
            if args.compute_latency:
                from benchmark import benchmark
                from line_profiler import LineProfiler
                throughput_dict = {'benchmarks': [],
                                'n_objects': [],
                                'n_interactions': []}
                all_latency = []
                num_object_list = [80]
                num_interaction_list = [50, 100, 150, 200, 250, 300, 350, 400]
                # num_interaction_list=[]
                for n_o in num_object_list:
                    for n_i in num_interaction_list:
                        benchmark_results = benchmark(
                            model,
                            batch_size=1,
                            num_objects=n_o,
                            num_interactions=n_i,
                            runs=100,
                            throw_out=0.2,
                            args=args
                        )
                        all_latency.append(benchmark_results['throughput_im_per_sec'])
                        throughput_dict['benchmarks'].append(benchmark_results)
                        throughput_dict['n_objects'].append(n_o)
                        throughput_dict['n_interactions'].append(n_i)
                        print_ddp(f"Throughput for {n_o} objects and {n_i} interactions: {benchmark_results['throughput_im_per_sec']:.2f} im/s", args)
                        
                # for instance-level hoi reasoning        
                num_instance_pairs_list = [5,10, 20, 50, 100, 150, 200,300, 400]
                instance_latency_dict = {'benchmarks': [],
                                        'n_instance_pairs': []}
                for n_i in num_instance_pairs_list:
                    benchmark_results = benchmark(
                        model,
                        batch_size=1,
                        num_instance_pairs=n_i,
                        runs=100,
                        throw_out=0.2,
                        reasoning_level='instance',
                        args=args
                    )
                    instance_latency_dict['benchmarks'].append(benchmark_results)
                    instance_latency_dict['n_instance_pairs'].append(n_i)
                    print_ddp(f"Instance-level latency for {n_i} instance pairs: {benchmark_results['throughput_im_per_sec']:.2f} im/s", args)
                    
                # Save throughput results to output directory
                import json
                import os
                
                # Ensure output directory exists
                os.makedirs(args.output_dir, exist_ok=True)
                
                all_dict = {'image_level': throughput_dict,
                            'instance_level': instance_latency_dict}
                # Save throughput results
                throughput_file = os.path.join(args.output_dir, 'throughput_results.json')
                with open(throughput_file, 'w') as f:
                    json.dump(all_dict, f, indent=2)
                
                print_ddp(f"Throughput results saved to {throughput_file}", args)
                return
                
            # evaluate
            val_loss, val_metrics = evaluate(
                model, val_loader, criterion, device, args, class_labels
            )
            for metric_name, value in val_metrics.items():
                if not isinstance(value,list):
                    print(f"Evaluation {metric_name}: {value:.4f}")
            return
        
        for epoch in range(args.num_epochs):
            print_ddp(f"\nEpoch {epoch + 1}/{args.num_epochs}", args)
            print_ddp("-" * 50, args)
            
            # Set epoch for distributed sampler
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            
            # Train one epoch
            train_loss, train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, device, epoch, args, class_labels, scheduler
            )
            
            # Validate
            val_loss, val_metrics = evaluate(
                model, val_loader, criterion, device, args, class_labels
            )
            
            # Store epoch results
            epoch_results = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_metrics': train_metrics if train_metrics else {},
                'val_loss': val_loss,
                'val_metrics': val_metrics if val_metrics else {},
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            all_results.append(epoch_results)
            
            # Update learning rate (only for epoch-based scheduler)
            if scheduler and args.scheduler_step_type == 'epoch':
                scheduler.step()
            
            # Print epoch results (only from main process)
            if is_main_process(args):
                print(f"Train Loss: {train_loss:.4f}")
                print(f"Val Loss: {val_loss:.4f}")
                
                if train_metrics:
                    for metric_name, value in train_metrics.items():
                        if not isinstance(value,list):
                            print(f"Train {metric_name}: {value:.4f}")
                        # else:
                        #     print(f"Train {metric_name}: {value}")
                
                if val_metrics:
                    for metric_name, value in val_metrics.items():
                        if not isinstance(value,list):
                            print(f"Val {metric_name}: {value:.4f}")
                
                # Print learning rates for each parameter group
                lr_info = []
                for i, group in enumerate(optimizer.param_groups):
                    group_name = group.get('name', f'group_{i}')
                    lr_info.append(f"{group_name}: {group['lr']:.6f}")
                print(f"Learning Rates - {', '.join(lr_info)}")
            
            # Log to wandb (only from main process)
            if wandb_active and is_main_process(args):
                log_dict = {
                    'epoch': epoch + 1,
                    'train/loss': train_loss,
                    'val/loss': val_loss,
                }
                
                # Log learning rates for each parameter group
                for group in optimizer.param_groups:
                    group_name = group.get('name', 'default')
                    log_dict[f'train/lr_{group_name}'] = group['lr']
                
                # Add train metrics
                if train_metrics:
                    for metric_name, value in train_metrics.items():
                        # Handle list metrics (like AP)
                        if isinstance(value, list):
                            # Log mean of valid values (excluding -1)
                            valid_values = [v for v in value if v != -1]
                            if valid_values:
                                log_dict[f'train/{metric_name}_mean'] = sum(valid_values) / len(valid_values)
                        else:
                            log_dict[f'train/{metric_name}'] = value
                
                # Add validation metrics
                if val_metrics:
                    for metric_name, value in val_metrics.items():
                        # Handle list metrics (like AP)
                        if isinstance(value, list):
                            # Log mean of valid values (excluding -1)
                            valid_values = [v for v in value if v != -1]
                            if valid_values:
                                log_dict[f'val/{metric_name}_mean'] = sum(valid_values) / len(valid_values)
                        else:
                            log_dict[f'val/{metric_name}'] = value
                
                wandb.log(log_dict)
            
            # Save best model (only from main process)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_metrics = val_metrics
                print_ddp(f"New best validation loss: {best_val_loss:.4f}", args)
                
                # Save best model checkpoint with metrics
                save_checkpoint_ddp(
                    model, optimizer, scheduler, epoch, val_loss, args, 
                    'best_model.pth', metrics=val_metrics
                )
        
        # Save final model checkpoint
        print_ddp("Saving final model checkpoint...", args)
        save_checkpoint_ddp(
            model, optimizer, scheduler, args.num_epochs - 1, val_loss, args,
            'final_model.pth', metrics=val_metrics
        )
        
        # Save all results to JSON
        final_results = {
            'training_summary': {
                'total_epochs': args.num_epochs,
                'best_val_loss': best_val_loss,
                'best_val_metrics': best_val_metrics if best_val_metrics else {},
                'final_val_loss': val_loss,
                'final_val_metrics': val_metrics if val_metrics else {}
            },
            'epoch_results': all_results,
            'model_config': {
                'vision_encoder': args.vision_encoder,
                'text_encoder': args.text_encoder,
                'vision_projector_type': args.vision_projector_type or args.projector_type,
                'text_projector_type': args.text_projector_type or args.projector_type,
                'vision_hidden_dim': args.vision_hidden_dim or args.hidden_dim,
                'text_hidden_dim': args.text_hidden_dim or args.hidden_dim,
                'projection_dim': args.projection_dim,
                'scale_logit': args.scale_logit,
                'scale_bias': args.scale_bias,
                'dataset_name': args.dataset_name,
                'target_type': args.target_type,
                'num_classes': args.num_classes
            },
            'training_config': {
                'global_batch_size': args.global_batch_size,
                'per_gpu_batch_size': args.batch_size,
                'gradient_accumulation_steps': gradient_accumulation_steps,
                'world_size': world_size,
                'effective_global_batch_size': args.batch_size * gradient_accumulation_steps * world_size,
                'learning_rate': args.lr,
                'weight_decay': args.weight_decay,
                'loss_type': args.loss_type,
                'num_epochs': args.num_epochs
            }
        }
        
        save_results_json(final_results, args, 'results.json')
        
        print_ddp("\nTraining completed!", args)
        print_ddp(f"Best validation loss: {best_val_loss:.4f}", args)
        if best_val_metrics:
            print_ddp(f"Best validation mAP: {best_val_metrics.get('mAP', 'N/A'):.4f}", args)
        
        # Log final results to wandb
        if wandb_active and is_main_process(args):
            wandb.summary.update({
                'final/best_val_loss': best_val_loss,
                'final/final_val_loss': val_loss,
                'final/best_val_mAP': best_val_metrics.get('mAP', 0) if best_val_metrics else 0,
                'final/final_val_mAP': val_metrics.get('mAP', 0) if val_metrics else 0
            })
        
    finally:
        # Finish wandb run
        if wandb_active and is_main_process(args):
            wandb.finish()
            print_ddp("Wandb run finished", args)
        
        # Clean up distributed training
        cleanup_ddp()


if __name__ == "__main__":
    main()
