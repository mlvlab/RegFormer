import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
import numpy as np
from tqdm import tqdm
import time

from core.ddp_utils import reduce_tensor, reduce_dict, is_main_process, gather_numpy_arrays, print_ddp
from eval.metric import _average_precision

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, args, class_labels, scheduler=None):
    """
    Train the model for one epoch
    
    Args:
        model: The model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
        epoch: Current epoch number
        args: Arguments containing DDP configuration
        class_labels: List of text labels for each class
        scheduler: Learning rate scheduler (optional)
    
    Returns:
        avg_loss: Average training loss
        metrics: Dictionary of training metrics
    """
    model.train()
    criterion.train()
    running_loss = 0.0
    all_predictions = []
    all_targets = []
    
    # Gradient accumulation setup
    gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
    max_grad_norm = getattr(args, 'max_grad_norm', 1.0)
    world_size = getattr(args, 'world_size', 1)
    
    # Calculate batch size metrics
    per_gpu_batch_size = args.batch_size
    effective_batch_size_per_gpu = per_gpu_batch_size * gradient_accumulation_steps
    global_effective_batch_size = effective_batch_size_per_gpu * world_size
    
    # Print gradient accumulation info (only on main process and only once per epoch)
    if is_main_process(args) and epoch == 0:
        print_ddp(f"Training batch size configuration:", args)
        print_ddp(f"  Per-GPU batch size: {per_gpu_batch_size}", args)
        if gradient_accumulation_steps > 1:
            print_ddp(f"  Gradient accumulation steps: {gradient_accumulation_steps}", args)
            print_ddp(f"  Effective batch size per GPU: {effective_batch_size_per_gpu}", args)
        print_ddp(f"  Global effective batch size: {global_effective_batch_size}", args)
        if max_grad_norm > 0:
            print_ddp(f"  Gradient clipping: max_norm={max_grad_norm}", args)
    
    # Use tqdm for progress bar (only on main process)
    if is_main_process(args):
        desc = f"Training Epoch {epoch + 1}"
        if gradient_accumulation_steps > 1:
            desc += f" (global_bs={global_effective_batch_size})"
        else:
            desc += f" (global_bs={global_effective_batch_size})"
        pbar = tqdm(train_loader, desc=desc)
        data_loader = pbar
    else:
        data_loader = train_loader
    
    # Zero gradients at the beginning
    optimizer.zero_grad()
    
    for batch_idx, (images, targets) in enumerate(data_loader):
        # Move data to device
        images = images.to(device)
        targets = { k: v.to(device) for k, v in targets.items()}
        train_type = args.target_type
        interaction_targets = targets[train_type]
        test_targets = targets['hoi']
        # Forward pass
        # Get predictions from vision-text alignment model using class labels
        # For multi-label classification, we compute similarity with all class labels
        out_dict = model(images, class_labels)
        logits = out_dict['logits']
        test_logits = out_dict['test_logits']
        # # Handle different logit shapes based on model output
        # if logits.dim() > 1 and logits.shape != targets.shape:
        #     # If model returns similarity matrix, we need to adapt it
        #     if logits.shape[1] == targets.shape[1]:
        #         # Already the right shape for multi-label classification
        #         pass
        #     elif logits.shape[0] == logits.shape[1] == images.shape[0]:
        #         # Square similarity matrix - take diagonal for self-similarity
        #         logits = torch.diag(logits).unsqueeze(0) if logits.dim() == 2 else logits
        #     else:
        #         raise ValueError(f"Cannot adapt logits shape {logits.shape} to targets shape {targets.shape}")
        
        # Ensure logits have the correct shape
        if logits.shape != interaction_targets.shape:
            raise ValueError(f"Final logits shape {logits.shape} doesn't match interaction targets shape {interaction_targets.shape}")
        
        # Calculate loss
        loss = 0.0
        label_mask = train_loader.dataset.label_mask
        interaction_loss = criterion(logits, interaction_targets, label_mask)* args.interaction_classifcation_loss_weight
        loss += interaction_loss 
        if 'instance_scores' in out_dict:
            instance_scores = out_dict['instance_scores']
            interactiveness_targets = targets['interactiveness']
            interactiveness_loss = criterion(instance_scores, interactiveness_targets)* args.interactiveness_loss_weight
            loss += interactiveness_loss 
        
        # Scale loss by gradient accumulation steps
        loss = loss / gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Accumulate loss (unscaled for logging)
        running_loss += loss.item() * gradient_accumulation_steps
        
        # Store predictions and targets for metrics
        with torch.no_grad():
            predictions = torch.sigmoid(test_logits).cpu().numpy()
            targets_np = test_targets.cpu().numpy()
            all_predictions.append(predictions)
            all_targets.append(targets_np)
        
        # Update weights every gradient_accumulation_steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()
            
            # Update learning rate (for iteration-based scheduler)
            if scheduler and hasattr(args, 'scheduler_step_type') and args.scheduler_step_type == 'iteration':
                scheduler.step()
        
        # Update progress bar (only on main process)
        if is_main_process(args) and hasattr(data_loader, 'set_postfix'):
            # Get current learning rates for display
            lr_info = {}
            for param_group in optimizer.param_groups:
                group_name = param_group.get('name', 'default')
                lr_info[f'{group_name}_lr'] = f"{param_group['lr']:.2e}"
            
            # Add scheduler status if available
            if args.use_seperate_interactiveness_loss:
                postfix = {'Loss': f'{loss.item() * gradient_accumulation_steps:.4f}, Interaction Loss: {interaction_loss.item() * gradient_accumulation_steps:.4f}, Interactiveness Loss: {interactiveness_loss.item() * gradient_accumulation_steps:.4f}'}  # Show unscaled loss
            else:
                postfix = {'Loss': f'{loss.item() * gradient_accumulation_steps:.4f}'}
            if gradient_accumulation_steps > 1:
                postfix['GradAccum'] = f"{(batch_idx + 1) % gradient_accumulation_steps}/{gradient_accumulation_steps}"
            if scheduler and hasattr(scheduler, 'current_step') and hasattr(scheduler, 'warmup_steps'):
                is_warmup = scheduler.current_step <= scheduler.warmup_steps
                phase = "W" if is_warmup else "C"  # W for warmup, C for cosine
                postfix['Phase'] = phase
                postfix['Step'] = f"{scheduler.current_step}"
            
            postfix.update(lr_info)
            data_loader.set_postfix(postfix)
    
    # Handle remaining gradients if not evenly divisible
    if len(train_loader) % gradient_accumulation_steps != 0:
        # Gradient clipping
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        # Final optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        # Update learning rate (for iteration-based scheduler)
        if scheduler and hasattr(args, 'scheduler_step_type') and args.scheduler_step_type == 'iteration':
            scheduler.step()
    
    # Calculate average loss
    avg_loss = running_loss / len(train_loader)
    
    # Reduce loss across all processes
    avg_loss_tensor = torch.tensor(avg_loss, device=device)
    avg_loss_tensor = reduce_tensor(avg_loss_tensor, args, average=True)
    avg_loss = avg_loss_tensor.item()
    
    # Calculate metrics on gathered data
    if all_predictions and all_targets:
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        
        # Gather predictions and targets from all processes
        gathered_predictions = gather_numpy_arrays(all_predictions, args)
        gathered_targets = gather_numpy_arrays(all_targets, args)
        
        # Calculate metrics only on main process with all data
        if is_main_process(args) and gathered_predictions is not None and gathered_targets is not None:
            metrics = calculate_metrics(gathered_predictions, gathered_targets)
            print_ddp(f"[Training] Calculated metrics on {gathered_predictions.shape[0]} total samples", args)
            
            # Log parameter group learning rates
            if scheduler:
                current_step = scheduler.current_step if hasattr(scheduler, 'current_step') else 'N/A'
                warmup_steps = scheduler.warmup_steps if hasattr(scheduler, 'warmup_steps') else 'N/A'
                is_warmup = current_step <= warmup_steps if isinstance(current_step, int) and isinstance(warmup_steps, int) else False
                warmup_status = "WARMUP" if is_warmup else "COSINE"
                print_ddp(f"[Training] LR Schedule - Step: {current_step}/{scheduler.total_steps if hasattr(scheduler, 'total_steps') else 'N/A'} ({warmup_status})", args)
            
            print_ddp(f"[Training] Parameter group learning rates:", args)
            for i, param_group in enumerate(optimizer.param_groups):
                group_name = param_group.get('name', f'group_{i}')
                lr = param_group['lr']
                print_ddp(f"  {group_name}: {lr:.2e}", args)
        else:
            metrics = {}
        
        # Return metrics directly without broadcasting to avoid DDP synchronization issues
        reduced_metrics = metrics
    else:
        reduced_metrics = {}
    
    return avg_loss, reduced_metrics


def evaluate(model, val_loader, criterion, device, args, class_labels):
    """
    Evaluate the model
    
    Args:
        model: The model to evaluate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to use
        args: Arguments containing DDP configuration
        class_labels: List of text labels for each class
    
    Returns:
        avg_loss: Average validation loss
        metrics: Dictionary of validation metrics
    """
    model.eval()
    criterion.eval()
    running_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        # Use tqdm for progress bar (only on main process)
        if is_main_process(args):
            pbar = tqdm(val_loader, desc="Evaluating")
            data_loader = pbar
        else:
            data_loader = val_loader
        
        for batch_idx, (images, targets) in enumerate(data_loader):
            # Move data to device
            images = images.to(device)
            targets = { k: v.to(device) for k, v in targets.items()}
            train_type = args.target_type
            interaction_targets = targets[train_type]
            test_targets = targets['hoi']
            
            # Forward pass
            
            # Get predictions from vision-text alignment model using class labels
            # For multi-label classification, we compute similarity with all class labels
            logit_dict = model(images, class_labels, meta_data=targets if args.vis_label_weight else None)
            logits = logit_dict['logits']
            test_logits = logit_dict['test_logits']
            
            # # Handle different logit shapes based on model output
            # if logits.dim() > 1 and logits.shape != targets.shape:
            #     # If model returns similarity matrix, we need to adapt it
            #     if logits.shape[1] == targets.shape[1]:
            #         # Already the right shape for multi-label classification
            #         pass
            #     elif logits.shape[0] == logits.shape[1] == images.shape[0]:
            #         # Square similarity matrix - take diagonal for self-similarity
            #         logits = torch.diag(logits).unsqueeze(0) if logits.dim() == 2 else logits
            #     else:
            #         raise ValueError(f"Cannot adapt logits shape {logits.shape} to targets shape {targets.shape}")
            
            # Ensure logits have the correct shape
            if logits.shape != interaction_targets.shape:
                raise ValueError(f"Final logits shape {logits.shape} doesn't match interaction targets shape {interaction_targets.shape}")
            
            # Calculate loss
            loss = criterion(logits, interaction_targets)
            running_loss += loss.item()
            
            # Store predictions and targets for metrics
            predictions = torch.sigmoid(test_logits).cpu().numpy()
            targets_np = test_targets.cpu().numpy()
            all_predictions.append(predictions)
            all_targets.append(targets_np)
            
            # Update progress bar (only on main process)
            if is_main_process(args) and hasattr(data_loader, 'set_postfix'):
                data_loader.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    # Calculate average loss
    avg_loss = running_loss / len(val_loader)
    
    # Reduce loss across all processes
    avg_loss_tensor = torch.tensor(avg_loss, device=device)
    avg_loss_tensor = reduce_tensor(avg_loss_tensor, args, average=True)
    avg_loss = avg_loss_tensor.item()
    
    # Calculate metrics on gathered data
    if all_predictions and all_targets:
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        
        # Gather predictions and targets from all processes
        gathered_predictions = gather_numpy_arrays(all_predictions, args)
        gathered_targets = gather_numpy_arrays(all_targets, args)
        
        # Calculate metrics only on main process with all data
        if is_main_process(args) and gathered_predictions is not None and gathered_targets is not None:
            metrics = calculate_metrics(gathered_predictions, gathered_targets, val_loader.dataset.unseen_index)
            print_ddp(f"[Validation] Calculated metrics on {gathered_predictions.shape[0]} total samples", args)
        else:
            metrics = {}
        
        # Return metrics directly without broadcasting to avoid DDP synchronization issues
        reduced_metrics = metrics
    else:
        reduced_metrics = {}
    
    return avg_loss, reduced_metrics


def calculate_metrics(predictions, targets, unseen_index=None):
    """
    Calculate various metrics for multi-label classification using score-based evaluation
    
    Args:
        predictions: Predicted probabilities [batch_size, num_classes]
        targets: Ground truth labels [batch_size, num_classes]
    
    Returns:
        metrics: Dictionary of calculated metrics
    """
    metrics = {}
    
    # Class-wise Average Precision (mAP)
    interp_aps, aps = [], []
    valid_classes = []
    
    for i in range(targets.shape[1]):
        y_true = targets[:, i]
        y_scores = predictions[:, i]
        
        if y_true.sum() > 0:  # If class has positive samples
            try:
                # Sort by scores (descending)
                sorted_indices = np.argsort(y_scores)[::-1]
                sorted_true = y_true[sorted_indices]
                
                # Calculate precision at each threshold
                tp = np.cumsum(sorted_true)
                fp = np.cumsum(1 - sorted_true)
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (y_true.sum() + 1e-8)
                
                # Calculate AP using trapezoidal rule
                # Add endpoints for proper integration
                recall_extended = np.concatenate([[0], recall, [1]])
                precision_extended = np.concatenate([[0], precision, [0]])
                
                # Make precision monotonically decreasing
                for j in range(len(precision_extended) - 2, -1, -1):
                    precision_extended[j] = max(precision_extended[j], precision_extended[j + 1])
                
                # Calculate AP
                interp_ap = np.trapz(precision_extended, recall_extended)
                interp_aps.append(interp_ap)
                valid_classes.append(i)
                
                ap = _average_precision(y_scores, y_true)
                aps.append(ap)
            except Exception as e:
                print(f"Warning: Error calculating AP for class {i}: {e}")
                interp_aps.append(-1)
                aps.append(-1)  # Add -1 for error cases
                continue
        else:
            # No positive samples for this class
            interp_aps.append(-1)
            aps.append(-1)
    
    # Store all APs (including -1 for classes without positive samples)
    metrics['AP'] = aps
    metrics['interp_AP'] = interp_aps
    # Mean Average Precision (exclude -1 values)
    valid_aps = [ap for ap in aps if ap != -1]
    valid_interp_aps = [ap for ap in interp_aps if ap != -1]
    if valid_aps:
        metrics['mAP'] = np.mean(valid_aps)
        metrics['num_valid_classes'] = len(valid_aps)
        metrics['interp_mAP'] = np.mean(valid_interp_aps)
        if unseen_index is not None:
            unseen_aps = []
            seen_aps = []
            unseen_interp_aps = []
            seen_interp_aps = []
            for i in range(len(aps)):
                if aps[i] != -1:
                    if i in unseen_index:
                        unseen_aps.append(aps[i])
                        unseen_interp_aps.append(interp_aps[i])
                    else:
                        seen_aps.append(aps[i])
                        seen_interp_aps.append(interp_aps[i])
            metrics['unseen_mAP'] = np.mean(unseen_aps)
            metrics['seen_mAP'] = np.mean(seen_aps)
            metrics['unseen_interp_mAP'] = np.mean(unseen_interp_aps)
            metrics['seen_interp_mAP'] = np.mean(seen_interp_aps)
            metrics['num_unseen_classes'] = len(unseen_aps)
            metrics['num_seen_classes'] = len(seen_aps)
    else:
        metrics['mAP'] = 0.0
        metrics['interp_mAP'] = 0.0
        metrics['num_valid_classes'] = 0
    
    metrics['num_classes_with_no_positives'] = aps.count(-1)
    
    # Top-k accuracy (ranking-based)
    for k in [1, 3, 5]:
        if k <= predictions.shape[1]:
            metrics[f'top{k}_accuracy'] = top_k_accuracy(predictions, targets, k)
    
    # Score-based ranking metrics
    metrics['mean_rank'] = calculate_mean_rank(predictions, targets)
    metrics['mrr'] = calculate_mean_reciprocal_rank(predictions, targets)  # Mean Reciprocal Rank
    
    # Class distribution info
    metrics['avg_positive_per_sample'] = np.mean(np.sum(targets, axis=1))
    metrics['class_balance'] = np.sum(targets, axis=0).std() / (np.sum(targets, axis=0).mean() + 1e-8)

    return metrics


def calculate_mean_rank(predictions, targets):
    """
    Calculate mean rank of true positive labels
    
    Args:
        predictions: Predicted probabilities [batch_size, num_classes]
        targets: Ground truth labels [batch_size, num_classes]
    
    Returns:
        mean_rank: Average rank of true positive labels
    """
    ranks = []
    
    for i in range(len(targets)):
        # Get indices of true positive labels
        true_labels = np.where(targets[i] == 1)[0]
        
        if len(true_labels) > 0:
            # Get ranks (1-indexed) of predictions for this sample
            sorted_indices = np.argsort(predictions[i])[::-1]  # Descending order
            ranks_dict = {idx: rank + 1 for rank, idx in enumerate(sorted_indices)}
            
            # Get ranks of true labels
            sample_ranks = [ranks_dict[label] for label in true_labels]
            ranks.extend(sample_ranks)
    
    return np.mean(ranks) if ranks else float('inf')


def calculate_mean_reciprocal_rank(predictions, targets):
    """
    Calculate Mean Reciprocal Rank (MRR)
    
    Args:
        predictions: Predicted probabilities [batch_size, num_classes]
        targets: Ground truth labels [batch_size, num_classes]
    
    Returns:
        mrr: Mean Reciprocal Rank
    """
    reciprocal_ranks = []
    
    for i in range(len(targets)):
        # Get indices of true positive labels
        true_labels = np.where(targets[i] == 1)[0]
        
        if len(true_labels) > 0:
            # Get ranks (1-indexed) of predictions for this sample
            sorted_indices = np.argsort(predictions[i])[::-1]  # Descending order
            ranks_dict = {idx: rank + 1 for rank, idx in enumerate(sorted_indices)}
            
            # Get minimum rank (best rank) of true labels
            min_rank = min(ranks_dict[label] for label in true_labels)
            reciprocal_ranks.append(1.0 / min_rank)
    
    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0


def top_k_accuracy(predictions, targets, k):
    """
    Calculate top-k accuracy for multi-label classification
    
    Args:
        predictions: Predicted probabilities [batch_size, num_classes]
        targets: Ground truth labels [batch_size, num_classes]
        k: Number of top predictions to consider
    
    Returns:
        accuracy: Top-k accuracy
    """
    # Get top-k predictions for each sample
    top_k_indices = np.argsort(predictions, axis=1)[:, -k:]
    
    correct = 0
    total = 0
    
    for i in range(len(targets)):
        # Get true positive labels for this sample
        true_labels = np.where(targets[i] == 1)[0]
        
        if len(true_labels) > 0:  # If sample has positive labels
            # Check if any true label is in top-k predictions
            if any(label in top_k_indices[i] for label in true_labels):
                correct += 1
            total += 1
    
    return correct / total if total > 0 else 0.0



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
