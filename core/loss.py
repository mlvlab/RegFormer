import torch
import torch.nn as nn
import torch.nn.functional as F
import random

def get_fed_loss_inds(targets, num_sample_cats, weight=None, verb_to_hoi=None, hoi_to_verb=None, verb_centric_negative_sampling=False, label_mask=None):
    """
    Get indices for federated loss
    code from https://github.com/facebookresearch/Detic/blob/main/detic/modeling/utils.py
    """
    B,C = targets.shape
    weight = weight.to(targets.device)
    gt_classes = targets.any(dim=0) # [C]
    appeared = gt_classes.nonzero().squeeze(1) # C'
    not_used_labels = torch.where(label_mask == 0)[0].to(targets.device)
    # Verb-centric negative sampling
    if verb_centric_negative_sampling and verb_to_hoi is not None and hoi_to_verb is not None:
        verb_related_hois = set()
        for hoi_idx in appeared.cpu().numpy():
            verb_idx = hoi_to_verb[hoi_idx]
            verb_related_hois.update(verb_to_hoi[verb_idx])
        
        # Add verb-related HOIs that are not already in appeared
        new_hois = list(verb_related_hois - set(appeared.cpu().numpy()))
        if new_hois:
            new_hois_tensor = torch.tensor(new_hois, device=targets.device)
            appeared = torch.cat([appeared, new_hois_tensor])
    
    # Remove indices that are in not_used_labels
    if label_mask is not None:
        appeared = appeared[~torch.isin(appeared, not_used_labels)]
    
    prob = appeared.new_ones(C).float()
    if len(appeared) < num_sample_cats:
        if weight is not None:
            prob = weight.float().clone()
        prob[appeared] = 0
        more_appeared = torch.multinomial(
            prob, num_sample_cats - len(appeared),
            replacement=False)
        appeared = torch.cat([appeared, more_appeared])
    appeared_mask = appeared.new_zeros(C, dtype=torch.bool, device=targets.device)
    appeared_mask[appeared] = 1
    fed_w = appeared_mask.view(1,C).expand(B, C)
    loss_weight = fed_w.float()
    return loss_weight

class BCELoss(nn.Module):
    """Standard Binary Cross Entropy Loss for multi-label classification"""
    
    def __init__(self, class_counts, reduction='mean', args=None):
        super().__init__()
        self.class_counts = class_counts
        self.reduction = reduction
        self.use_fed_loss = args.use_federated_loss
        self.target_type = args.target_type
        if self.use_fed_loss:
            self.fed_num_samples = args.fed_num_samples
            self.freq_weight = self.class_counts.float()*args.fed_loss_freq_weight
            self.verb_centric_negative_sampling = args.verb_centric_negative_sampling
            self.verb_to_hoi = args.verb_to_hoi
            self.hoi_to_verb = args.hoi_to_verb
        self.label_smoothing = args.label_smoothing
        if self.label_smoothing > 0.0:
            print(f"Label smoothing: {self.label_smoothing}")
            
    def forward(self, logits, targets, label_mask=None):
        """
        Args:
            logits: [batch_size, num_classes] - model predictions
            targets: [batch_size, num_classes] - ground truth labels (0 or 1)
        """            
        if self.label_smoothing > 0.0:
            smoothed_targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
            bce_loss = F.binary_cross_entropy_with_logits(logits, smoothed_targets, reduction='none')
        else:
            bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        if self.training and self.use_fed_loss:
            if len(self.class_counts) != logits.shape[1]:
                raise ValueError(f"Class counts length {len(self.class_counts)} does not match logits shape {logits.shape[1]}")
            loss_weight = get_fed_loss_inds(targets, 
                                            self.fed_num_samples, 
                                            self.freq_weight, 
                                            self.verb_to_hoi, 
                                            self.hoi_to_verb, 
                                            self.verb_centric_negative_sampling and self.target_type == 'hoi', 
                                            label_mask)
            if loss_weight[...,~label_mask].sum()!=0:
                raise ValueError(f"Loss weight for unused labels is not 0: {loss_weight[:,~label_mask]}")
            
            bce_loss = bce_loss * loss_weight
        bce_loss = bce_loss[..., label_mask]
        if self.reduction == 'mean':
            return bce_loss.mean()
        elif self.reduction == 'sum':
            return bce_loss.sum()
        elif self.reduction == 'target_mean':
            return bce_loss.sum() / targets.sum()
        else:
            return bce_loss


class FocalBCELoss(nn.Module):
    """Focal Binary Cross Entropy Loss for handling class imbalance"""
    
    def __init__(self, class_counts, alpha=1.0, gamma=2.0, reduction='mean', args=None):
        super().__init__()
        self.class_counts = class_counts
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.use_fed_loss = args.use_federated_loss
        if self.use_fed_loss:
            self.fed_num_samples = args.fed_num_samples
            self.freq_weight = self.class_counts.float()*args.fed_loss_freq_weight
            self.verb_centric_negative_sampling = args.verb_centric_negative_sampling
            self.verb_to_hoi = args.verb_to_hoi
            self.hoi_to_verb = args.hoi_to_verb
        self.label_smoothing = args.label_smoothing
        if self.label_smoothing > 0.0:
            print(f"Label smoothing: {self.label_smoothing}")
    
    def forward(self, logits, targets, label_mask=None):
        """
        Args:
            logits: [batch_size, num_classes] - model predictions
            targets: [batch_size, num_classes] - ground truth labels (0 or 1)
        """
        # Get probabilities
        probs = torch.sigmoid(logits)
        
        # Calculate BCE loss
        if self.label_smoothing > 0.0 and self.training:
            smoothed_targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
            bce_loss = F.binary_cross_entropy_with_logits(logits, smoothed_targets, reduction='none')
        else:
            bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        if self.training and self.use_fed_loss:
            if len(self.class_counts) != logits.shape[1]:
                raise ValueError(f"Class counts length {len(self.class_counts)} does not match logits shape {logits.shape}")
            loss_weight = get_fed_loss_inds(targets, 
                                            self.fed_num_samples, 
                                            self.freq_weight, 
                                            self.verb_to_hoi, 
                                            self.hoi_to_verb, 
                                            self.verb_centric_negative_sampling and self.target_type == 'hoi', 
                                            label_mask)
            if loss_weight[...,~label_mask].sum()!=0:
                raise ValueError(f"Loss weight for unused labels is not 0: {loss_weight[...,~label_mask]}")
            
            bce_loss = bce_loss * loss_weight
        # Calculate focal weight
        # For positive samples (targets = 1): (1 - p)^gamma
        # For negative samples (targets = 0): p^gamma
        if self.label_smoothing > 0.0 and self.training:
            pt = smoothed_targets * probs + (1 - smoothed_targets) * (1 - probs)
        else:
            pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting (optional)
        if self.label_smoothing > 0.0 and self.training:
            alpha_weight = smoothed_targets * self.alpha + (1 - smoothed_targets) * (1 - self.alpha)
        else:
            alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # Calculate focal loss
        focal_loss = alpha_weight * focal_weight * bce_loss
        # focal_loss = focal_loss[:, label_mask]
        focal_loss = focal_loss[..., label_mask]
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        elif self.reduction == 'target_mean':
            return focal_loss.sum() / targets.sum()
        else:
            return focal_loss


class FederatedBCELoss(nn.Module):
    """Federated BCE Loss with negative label sampling based on positive label count"""
    
    def __init__(self, class_counts, neg_pos_ratio=3.0, reduction='mean'):
        super().__init__()
        self.class_counts = class_counts
        self.neg_pos_ratio = neg_pos_ratio
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, logits, targets, label_mask=None):
        """
        Args:
            logits: [batch_size, num_classes] - model predictions
            targets: [batch_size, num_classes] - ground truth labels (0 or 1)
        """
        batch_size, num_classes = targets.shape
        device = targets.device
        
        # Calculate BCE loss for all samples
        bce_losses = self.bce(logits, targets)
        
        # Create mask for selected samples
        selected_mask = torch.zeros_like(targets, dtype=torch.bool)
        
        for i in range(batch_size):
            # Find positive and negative indices for this sample
            pos_indices = torch.where(targets[i] == 1)[0]
            neg_indices = torch.where(targets[i] == 0)[0]
            
            num_pos = len(pos_indices)
            num_neg = len(neg_indices)
            
            # Always include all positive samples
            selected_mask[i, pos_indices] = True
            
            if num_pos > 0 and num_neg > 0:
                # Calculate number of negative samples to select
                num_neg_to_select = min(int(num_pos * self.neg_pos_ratio), num_neg)
                
                # Randomly sample negative indices
                if num_neg_to_select > 0:
                    sampled_neg_indices = torch.randperm(num_neg, device=device)[:num_neg_to_select]
                    selected_neg_indices = neg_indices[sampled_neg_indices]
                    selected_mask[i, selected_neg_indices] = True
            elif num_pos == 0:
                # If no positive samples, sample some negative samples
                num_neg_to_select = min(max(1, int(num_neg * 0.1)), num_neg)  # Sample 10% of negatives
                if num_neg_to_select > 0:
                    sampled_neg_indices = torch.randperm(num_neg, device=device)[:num_neg_to_select]
                    selected_neg_indices = neg_indices[sampled_neg_indices]
                    selected_mask[i, selected_neg_indices] = True
        
        # Apply mask to select losses
        selected_losses = bce_losses[selected_mask]
        
        if self.reduction == 'mean':
            return selected_losses.mean() if len(selected_losses) > 0 else torch.tensor(0.0, device=device)
        elif self.reduction == 'sum':
            return selected_losses.sum()
        else:
            # Return losses with mask for custom reduction
            return bce_losses, selected_mask

# Tagging loss function
# copy from https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True, args=None):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.reduction = args.loss_reduction
        self.args = args
    def forward(self, x, y, label_mask=None):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss and self.training:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss and self.training:
                torch.set_grad_enabled(True)
            loss *= one_sided_w
        loss = loss[:, label_mask]
        if self.reduction == 'mean':
            return -loss.mean()
        elif self.reduction == 'sum':
            return -loss.sum()
        elif self.reduction == 'target_mean':
            return -loss.sum() / y.sum()
        else:
            return -loss    

class CombinedLoss(nn.Module):
    """Combined loss function that can mix different loss types"""
    
    def __init__(self, loss_configs):
        """
        Args:
            loss_configs: List of dict with keys 'type', 'weight', and loss-specific params
                Example: [
                    {'type': 'bce', 'weight': 0.5},
                    {'type': 'focal', 'weight': 0.5, 'alpha': 1.0, 'gamma': 2.0}
                ]
        """
        super().__init__()
        self.losses = nn.ModuleList()
        self.weights = []
        
        for config in loss_configs:
            loss_type = config['type'].lower()
            weight = config.get('weight', 1.0)
            
            if loss_type == 'bce':
                loss_fn = BCELoss(reduction=config.get('reduction', 'mean'))
            elif loss_type == 'focal':
                loss_fn = FocalBCELoss(
                    alpha=config.get('alpha', 1.0),
                    gamma=config.get('gamma', 2.0),
                    reduction=config.get('reduction', 'mean')
                )
            elif loss_type == 'federated':
                loss_fn = FederatedBCELoss(
                    neg_pos_ratio=config.get('neg_pos_ratio', 3.0),
                    reduction=config.get('reduction', 'mean')
                )
            else:
                raise ValueError(f"Unsupported loss type: {loss_type}")
            
            self.losses.append(loss_fn)
            self.weights.append(weight)
    
    def forward(self, logits, targets):
        total_loss = 0
        for loss_fn, weight in zip(self.losses, self.weights):
            loss = loss_fn(logits, targets)
            total_loss += weight * loss
        return total_loss


def get_loss_function(class_counts, args):
    """
    Factory function to create loss function based on arguments
    
    Args:
        args: Arguments containing loss configuration
    
    Returns:
        loss_function: Configured loss function
    """
    loss_type = getattr(args, 'loss_type', 'bce').lower()
    
    if loss_type == 'bce':
        return BCELoss(class_counts=class_counts, reduction=args.loss_reduction, args=args)
    elif loss_type == 'focal':
        alpha = getattr(args, 'focal_alpha', 1.0)
        gamma = getattr(args, 'focal_gamma', 2.0)
        return FocalBCELoss(class_counts=class_counts, alpha=alpha, gamma=gamma, reduction=args.loss_reduction, args=args)
    elif loss_type == 'federated':
        neg_pos_ratio = getattr(args, 'neg_pos_ratio', 3.0)
        return FederatedBCELoss(class_counts=class_counts, neg_pos_ratio=neg_pos_ratio, reduction=args.loss_reduction, args=args)
    elif loss_type == 'asl':
        gamma_neg = getattr(args, 'gamma_neg', 4.0)
        gamma_pos = getattr(args, 'gamma_pos', 1.0)
        clip = getattr(args, 'clip', 0.05)
        eps = getattr(args, 'eps', 1e-8)
        disable_torch_grad_focal_loss = getattr(args, 'disable_torch_grad_focal_loss', True)
        return AsymmetricLoss(gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=clip, eps=eps, disable_torch_grad_focal_loss=disable_torch_grad_focal_loss, args=args)
    elif loss_type == 'combined':
        # Example combined loss configuration
        loss_configs = [
            {'type': 'bce', 'weight': 0.5},
            {'type': 'focal', 'weight': 0.5, 'alpha': 1.0, 'gamma': 2.0}
        ]
        return CombinedLoss(loss_configs)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")