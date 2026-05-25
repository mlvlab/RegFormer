import argparse
import numpy as np

def model_args():
    parser = argparse.ArgumentParser(add_help=False)
# Vision Encoder Arguments
    parser.add_argument('--vision_encoder', type=str, default='openai/clip-vit-base-patch32',
                       help='Vision encoder model name from Hugging Face')
    parser.add_argument('--freeze_vision_encoder', action='store_true', default=False,
                       help='Whether to freeze vision encoder parameters')
    parser.add_argument('--vision_ft_method', type=str, default=None,
                       help='Method to fine-tune vision encoder')
    parser.add_argument('--ft_start_layer', type=int, default=None,
                       help='Start layer for vision encoder fine-tuning')
    parser.add_argument('--ft_end_layer', type=int, default=None,
                       help='End layer for vision encoder fine-tuning')
    parser.add_argument('--layer_idx', type=int, default=-1,
                       help='extracted layer index for vision encoder')
    
    # Vision FT Arguments
    # LoRA
    parser.add_argument('--lora_r', type=int, default=8,
                       help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16,
                       help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.0,
                       help='LoRA dropout')
    
    
    
    # Text Encoder Arguments  
    parser.add_argument('--text_encoder', type=str, default='openai/clip-vit-base-patch32',
                       help='Text encoder model name from Hugging Face')
    parser.add_argument('--freeze_text_encoder', action='store_true', default=False,
                       help='Whether to freeze text encoder parameters')
    parser.add_argument('--use_text_embeddings', action='store_true', default=False,
                       help='Whether to use text embeddings')
    parser.add_argument('--text_ft_method', type=str, default=None,
                       help='Method to fine-tune text encoder', choices=['lora','prompt_tuning'])
    parser.add_argument('--text_ft_start_layer', type=int, default=None,
                       help='Start layer for text encoder fine-tuning')
    parser.add_argument('--text_ft_end_layer', type=int, default=None,
                       help='End layer for text encoder fine-tuning')
    
    # text lora arguments
    parser.add_argument('--text_lora_r', type=int, default=8,
                       help='Text LoRA rank')
    parser.add_argument('--text_lora_alpha', type=int, default=16,
                       help='Text LoRA alpha')
    parser.add_argument('--text_lora_dropout', type=float, default=0.0,
                       help='Text LoRA dropout')
    
    # prompt tuning arguments
    parser.add_argument('--text_num_prompt_tokens', type=int, default=16,
                       help='Number of prompt tokens for prompt tuning')
    parser.add_argument('--text_prompt_insert_layer', type=int, default=None,
                       help='Layer to insert prompt tokens for prompt tuning')
    
    
    # Projector Arguments (Legacy - applies to both if specific args not set)
    parser.add_argument('--projector_type', type=str, default='identity', choices=['identity', 'mlp', 'linear'],
                       help='Type of projector to use (legacy, applies to both if specific args not set)')
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='Hidden dimension for projector (legacy, applies to both if specific args not set)')
    parser.add_argument('--projection_dim', type=int, default=512,
                       help='Final projection dimension for both vision and text embeddings. if set to -1, we will automatically set the projection dimension to the text dimension')
    parser.add_argument('--use_internal_projector', action='store_true', default=False,
                       help='Whether to use internal projector for vision and text embeddings. CLIP uses this.')
    
    # Vision Projector Arguments
    parser.add_argument('--vision_projector_type', type=str, default=None, choices=['identity', 'mlp', 'linear', 'ln_linear','only_linear','lora','attention','mlp_ln','swin'],
                       help='Type of vision projector to use (overrides --projector_type)')
    parser.add_argument('--vision_hidden_dim', type=int, default=None,
                       help='Hidden dimension for vision MLP projector (overrides --hidden_dim)')
    parser.add_argument('--vision_skip_connection', action='store_true', default=False,
                       help='Whether to use skip connection for vision projector')
    # --------------- Vision Attention Arguments ---------------
    # parser.add_argument('--vision_attention_heads', type=int, default=None,
    #                    help='Number of attention heads for vision attention. if None, use the number of attention heads from the vision encoder')
    parser.add_argument('--vision_attention_layers', type=int, default=1,
                       help='Number of attention layers for vision attention')
    
    # Text Projector Arguments
    parser.add_argument('--text_projector_type', type=str, default=None, choices=['identity', 'mlp', 'linear', 'ln_linear','only_linear','lora'],
                       help='Type of text projector to use (overrides --projector_type)')
    parser.add_argument('--text_hidden_dim', type=int, default=None,
                       help='Hidden dimension for text MLP projector (overrides --hidden_dim)')
    parser.add_argument('--text_skip_connection', action='store_true', default=False,
                       help='Whether to use skip connection for text projector')
    
    # Seperate classifier for HOI classes
    parser.add_argument('--use_seperate_classifier', action='store_true', default=False,
                       help='Use seperate classifier for HOI classes. Only support for target type with "hoi"')
    parser.add_argument('--seperate_normalization', action='store_true', default=False,
                       help='Whether to use normalization for verb and object seperately')
    
    # Alignment Arguments
    parser.add_argument('--scale_logit', type=float, default=20.0,
                       help='Scale factor for logits')
    parser.add_argument('--scale_bias', type=float, default=-10.0,
                       help='Bias term for logits')
    parser.add_argument('--no_bias', action='store_true', default=False,
                       help='Whether to use bias for logits')
    
    # Model Arguments
    parser.add_argument('--normalize_embeddings', action='store_true', default=True,
                       help='Whether to L2 normalize embeddings before dot product')
    parser.add_argument('--custom_weight_initialization', action='store_true', default=False,
                       help='Whether to use custom weight initialization for the model')
    
    # Attention Pooling Arguments
    parser.add_argument('--use_attention_pooling', action='store_true', default=False,
                       help='Whether to use AttentionPool2d for vision features')
    parser.add_argument('--attention_type', type=str, default='cross', choices=['self', 'cross', 'dinotext','cross_v2','ml_decoder'],
                       help='Type of attention to use for AttentionPool2d')
    parser.add_argument('--attention_pool_dim', type=int, default=512,
                       help='Output dimension for AttentionPool2d')
    parser.add_argument('--attention_pool_heads', type=int, default=8,
                       help='Number of attention heads for AttentionPool2d')
    parser.add_argument('--prefix_type', type=str, default='avg', choices=['avg','learnable','original','avg_patch','avg_patch_dual'],
                        help='Type of prefix token to use for AttentionPool2d')
    parser.add_argument('--pool_type', type=str, default='token', choices=['token', 'avg'],
                       help='Pooling type for AttentionPool2d')
    parser.add_argument('--no_pos_embed', action='store_true', default=False,
                       help='Whether to use position embedding for AttentionPool2d')
    parser.add_argument('--return_attention_weight', action='store_true', default=False,
                       help='Whether to return attention weight for AttentionPool2d')
    parser.add_argument('--blocks_drop_path', type=float, default=0.0,
                       help='Drop path rate for DINOTXThead')
    parser.add_argument('--only_prefix_as_query', action='store_true', default=False,
                       help='Whether to only use prefix as query for DINOTXThead')
    parser.add_argument('--use_out_norm', action='store_true', default=False,
                       help='Whether to use output normalization for AttentionPool2d (CrossAttentionPool only)')
    parser.add_argument('--layer_scale', type=float, default=None,
                       help='Layer scale for AttentionPool2d (CrossAttentionPool only)')
    parser.add_argument('--use_all_tokens_for_kv', action='store_true', default=False,
                       help='Whether to use all tokens as key and value')
    # --------------- ML Decoder Arguments ---------------
    parser.add_argument('--ml_decoder_num_query', type=int, default=100,
                       help='Number of latent tokens for ML Decoder')
    parser.add_argument('--ml_decoder_query_type', type=str, default='learnable', choices=['learnable', 'triplet', 'object'],
                       help='Type of query for ML Decoder')
    parser.add_argument('--ml_decoder_query_ckpt', type=str, default='object',
                       help='Path to the checkpoint for ML Decoder query')
    parser.add_argument('--ml_decoder_mil_type', type=str, default='max_logit', choices=['max_logit', 'max_pool'],
                       help='Type of MIL token for ML Decoder')
    parser.add_argument('--normalize_object_embeddings', action='store_true', default=False,
                       help='Whether to normalize object embeddings for ML Decoder. This is only used for object query type.')
    parser.add_argument('--ml_decoder_num_register', type=int, default=0,
                       help='Number of register tokens for ML Decoder. This is only used for learnable query type.')
    parser.add_argument('--use_multi_layer_ml_decoder', action='store_true', default=False,
                       help='Whether to use multiple layers for ML Decoder. This is only used for learnable query type.')
    parser.add_argument('--ml_decoder_num_layers', type=int, default=1,
                       help='Number of layers for ML Decoder. This is only used for learnable query type.')
    parser.add_argument('--language_query_as_pos_embed', action='store_true', default=False,
                       help='Whether to use language query as position embedding for ML Decoder. This is only used for learnable query type (object, triplet).')
    parser.add_argument('--language_pos_embed_type', type=str, default='add', choices=['add', 'concat'],
                       help='How language query is processed when acting as position embedding for ML Decoder.')
    parser.add_argument('--content_feature_type', type=str, default='zero', choices=['zero','avg_token','label_features'],
                        help='Type of content feature for ML Decoder. This is only used when language query is used as position embedding for ML Decoder.')
    parser.add_argument('--instance_filter_type', type=str, default='none', choices=['none', 'score', 'remove_query'],
                        help='Type of instance filter for ML Decoder. This is only used when object query type is used for ML Decoder.')
    parser.add_argument('--instance_agg_type', type=str, default='vector', choices=['vector', 'patch','patch_query'],
                        help='Type of instance aggregation for ML Decoder. This is only used when instance filter type is score.')
    parser.add_argument('--instance_activation_type', type=str, default='sigmoid', choices=['sigmoid', 'softmax'],
                        help='Type of instance activation for ML Decoder. This is only used when instance filter type is score.')
    parser.add_argument('--feature_agg_type', type=str, default='hard', choices=['hard','soft'],
                        help='Type of feature aggregation for ML Decoder. This is only used when instance filter type is score.')
    parser.add_argument('--revised_softmax', action='store_true', default=False,
                        help='Whether to use revised softmax for instance filter. This is only used when instance filter type is score.')
    parser.add_argument('--patch_score_stop_gd', action='store_true', default=False,
                        help='Whether to use patch score to get gumbel softmax for instance filter. This is only used when instance filter type is score.')
    parser.add_argument('--instance_scale', type=float, default=20.0,
                       help='Scale for instance filter for ML Decoder. This is only used when object query type is used for ML Decoder.')
    parser.add_argument('--instance_bias', type=float, default=-10.0,
                       help='Bias for instance filter for ML Decoder. This is only used when object query type is used for ML Decoder.')
    parser.add_argument('--instance_prior_factor', type=float, default=0.5,
                       help='Prior factor for instance filter for ML Decoder. This is only used when instance filter type is score.')
    parser.add_argument('--use_human_projector', action='store_true', default=False,
                       help='Whether to use human projector in interactiveness scoring.')
    parser.add_argument('--patch_scale', type=float, default=None,
                       help='Scale for patch for ML Decoder. This is only used when instance filter type is score.')
    parser.add_argument('--patch_score_agg_type', type=str, default='softmax', choices=['score_sum', 'softmax','sparsemax'],
                       help='Type of patch score aggregation for ML Decoder. This is only used when instance filter type is score.')
    parser.add_argument('--post_normalize_instance_scores', action='store_true', default=False,
                       help='Whether to normalize instance scores after aggregation for ML Decoder. This is only used when instance filter type is score and sigmoid is used as instance activation type.')
    parser.add_argument('--use_seperate_so', action='store_true', default=False,
                       help='Whether to use seperate SO for ML Decoder. This is only used when query type is object.')
    parser.add_argument('--instance_score_dim', type=int, default=None,
                       help='Dimension for instance score for ML Decoder. This is only used when use_seperate_so is True.')
    parser.add_argument('--share_so_lang_projection', action='store_true', default=False,
                       help='Whether to share language projection for SO and ML Decoder. This is only used when use_seperate_so is True.')
    parser.add_argument('--share_so_vis_projection', action='store_true', default=False,
                       help='Whether to share vision projection for SO and ML Decoder. This is only used when use_seperate_so is True.')
    parser.add_argument('--use_seperate_so_type', type=str, default='none', choices=['none', 'language', 'vision','language_vision'],
                       help='Whether to use seperate SO query for ML Decoder. This is only used when query type is object.')
    parser.add_argument('--use_seperate_subject_pair', action='store_true', default=False,
                       help='Whether to use seperate text form for subject pair for ML Decoder. This is only used when use_seperate_so is True.')
    # parser.add_argument('--instance_score_post_masking_type', action='store_true', default=False,
    #                    help='Whether to mask instance scores after aggregation for ML Decoder during instance-level inference. This is only used when instance filter type is score.')
    
    parser.add_argument('--instance_score_scheme', type=str, default='image', choices=['image','s_region','o_region','union','so_region'])
    
    # posion encoding
    parser.add_argument('--vis_pos_embed_type', type=str, default='', choices=['','learn','abs'],
                        help='Type of position embedding for ML Decoder. This is only used when instance score scheme is image.')
    parser.add_argument('--pe_temperatureH', type=float, default=20,
                        help='Temperature for H position embedding for ML Decoder. This is only used when instance score scheme is image.')
    parser.add_argument('--pe_temperatureW', type=float, default=20,
                        help='Temperature for W position embedding for ML Decoder. This is only used when instance score scheme is image.')
    parser.add_argument('--vis_pos_agg_strategy', type=str, default='add', choices=['add','concat'], 
                        help='Strategy to aggregate position encoding for ML Decoder. This is only used when instance score scheme is image.')
    
    # Visualization Arguments
    parser.add_argument('--vis_label_weight', action='store_true', default=False,
                       help='Whether to visualize label weight. only for ml decoder')
    parser.add_argument('--vis_instance_score', action='store_true', default=False,
                       help='Whether to visualize instance score. only for ml decoder')
    
    # eval arguments
    parser.add_argument('--compute_latency', action='store_true', default=False,
                       help='Whether to compute latency')
    parser.add_argument('--use_union_cropped_image', action='store_true', default=False,
                       help='Whether to use union cropped image for evaluation')
    return parser

def weak_hoi_args():
    parser = argparse.ArgumentParser(parents=[model_args()])
    
    # Data Arguments
    parser.add_argument('--train_data_path', type=str, required=True,
                       help='Path to the training JSON data file')
    parser.add_argument('--train_image_root', type=str, required=True,
                       help='Root directory for training images')
    parser.add_argument('--val_data_path', type=str, default=None,
                       help='Path to the validation JSON data file (default: same as train)')
    parser.add_argument('--val_image_root', type=str, default=None,
                       help='Root directory for validation images (default: same as train)')
    parser.add_argument('--dataset_name', type=str, default='hico', choices=['hico', 'vcoco', 'swig'],
                       help='Dataset name for label loading')
    parser.add_argument('--target_type', type=str, default='hoi', choices=['verb', 'hoi'],
                       help='Type of target labels to use')
    parser.add_argument('--test_type', type=str, default=None, choices=['verb', 'hoi'],
                       help='Type of test labels to use. If None, use the same as target_type')
    parser.add_argument('--num_classes', type=int, default=None,
                       help='Number of classes (auto-detected from dataset if None)')
    parser.add_argument('--zs_type', type=str, default=None,
                       help='Type of zero-shot classification to use')
    
    # Legacy arguments for backward compatibility
    parser.add_argument('--data_path', type=str, default=None,
                       help='Legacy: Path to the JSON data file (will be used for both train and val if new args not provided)')
    parser.add_argument('--image_root', type=str, default=None,
                       help='Legacy: Root directory for images (will be used for both train and val if new args not provided)')
    
    # Training Arguments
    parser.add_argument('--global_batch_size', type=int, default=128,
                       help='Global batch size across all GPUs and accumulation steps')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Per-GPU batch size (auto-calculated from global_batch_size if not specified)')
    parser.add_argument('--eval_batch_size', type=int, default=64,
                       help='Batch size for evaluation')
    parser.add_argument('--input_resolution', type=int, default=224,
                       help='Input resolution for the model')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate for projectors and other parameters')
    parser.add_argument('--zero_wd_for_scaler', action='store_true', default=False,
                       help='Whether to use zero weight decay for logit_scale and bias')
    
    # Gradient Accumulation Arguments
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Number of gradient accumulation steps before updating weights')
    parser.add_argument('--max_grad_norm', type=float, default=0.0,
                       help='Maximum gradient norm for gradient clipping (0 to disable)')
    parser.add_argument('--vision_lr', type=float, default=None,
                       help='Learning rate for vision encoder (uses --lr if None)')
    parser.add_argument('--text_lr', type=float, default=None,
                       help='Learning rate for text encoder (uses --lr if None)')
    parser.add_argument('--projector_lr', type=float, default=None,
                       help='Learning rate for projectors (uses --lr if None)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    
    # Learning rate scheduler arguments
    parser.add_argument('--use_cosine_scheduler', action='store_true', default=True,
                       help='Use cosine annealing learning rate scheduler')
    parser.add_argument('--warmup_epochs', type=float, default=1.0,
                       help='Number of warmup epochs (can be fractional)')
    parser.add_argument('--warmup_start_lr', type=float, default=1e-6,
                       help='Starting learning rate for warmup (applied to all param groups)')
    parser.add_argument('--min_lr_ratio', type=float, default=0.01,
                       help='Minimum LR as ratio of initial LR (for cosine annealing)')
    parser.add_argument('--min_lr', type=float, default=None,
                       help='Minimum learning rate (for cosine annealing). If not set, use min_lr_ratio * lr')
    parser.add_argument('--scheduler_step_type', type=str, default='iteration', choices=['epoch', 'iteration'],
                       help='Whether to step scheduler per epoch or per iteration')
    
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers for data loading')
    
    # Evaluation Arguments
    parser.add_argument('--eval_only', action='store_true', default=False,
                       help='Whether to evaluate only')
    parser.add_argument('--finetuned_ckpt', type=str, default='',
                       help='Directory to load finetuned model')
    
    # Reproducibility Arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Loss Arguments
    parser.add_argument('--loss_type', type=str, default='bce', 
                       choices=['bce', 'focal', 'federated','asl', 'combined'],
                       help='Type of loss function to use')
    parser.add_argument('--loss_reduction', type=str, default='mean', choices=['mean', 'sum','target_mean'],
                       help='Reduction type for loss function')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                       help='Alpha parameter for focal loss')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Gamma parameter for focal loss')
    parser.add_argument('--neg_pos_ratio', type=float, default=3.0,
                       help='Negative to positive ratio for federated loss')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                       help='Label smoothing parameter')
    parser.add_argument('--interaction_classifcation_loss_weight', type=float, default=1.0,
                       help='Weight for interaction classification loss')
    
    parser.add_argument('--use_federated_loss', action='store_true', default=False,
                       help='Whether to use federated loss')
    parser.add_argument('--verb_centric_negative_sampling', action='store_true', default=False,
                       help='Whether to use verb-centric negative sampling')
    parser.add_argument('--fed_num_samples', type=int, default=100,
                       help='Number of samples to use for federated loss')
    parser.add_argument('--fed_loss_freq_weight', type=float, default=0.5,
                       help='Frequency weight for federated loss')
    
    parser.add_argument('--use_seperate_interactiveness_loss', action='store_true', default=False,
                       help='Whether to use seperate interactiveness loss')
    parser.add_argument('--no_detach_gradient', action='store_true', default=False,
                       help='Whether to detach gradient for interactiveness loss')
    parser.add_argument('--interactiveness_loss_weight', type=float, default=1.0,
                       help='Weight for interactiveness loss')
    
    parser.add_argument('--gamma_neg', type=float, default=4.0,
                       help='Gamma negative parameter for asymmetric loss')
    parser.add_argument('--gamma_pos', type=float, default=1.0,
                       help='Gamma positive parameter for asymmetric loss')
    parser.add_argument('--clip', type=float, default=0.05,
                       help='Clip parameter for asymmetric loss')
    parser.add_argument('--eps', type=float, default=1e-8,
                       help='Epsilon parameter for asymmetric loss')
    # parser.add_argument('--disable_torch_grad_focal_loss', action='store_true', default=False,
    #                    help='Whether to disable torch grad focal loss for asymmetric loss')
    
    # Device Arguments
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for training')
    
    # Output Arguments
    parser.add_argument('--output_dir', type=str, default='./output/pdb',
                       help='Directory to save checkpoints and results')
    
    # Wandb Arguments
    parser.add_argument('--use_wandb', action='store_true', default=False,
                       help='Use Weights & Biases for experiment tracking')
    parser.add_argument('--wandb_project', type=str, default='weak-hoi',
                       help='Wandb project name')
    parser.add_argument('--wandb_id', type=str, default=None,
                       help='Wandb run id')
    parser.add_argument('--wandb_entity', type=str, default=None,
                       help='Wandb entity/team name')
    parser.add_argument('--wandb_name', type=str, default=None,
                       help='Wandb run name (auto-generated if None)')
    parser.add_argument('--wandb_group', type=str, default=None,
                       help='Wandb group name')
    parser.add_argument('--wandb_tags', type=str, nargs='*', default=[],
                       help='Wandb tags for the run')
    parser.add_argument('--wandb_notes', type=str, default=None,
                       help='Wandb run notes/description')
    parser.add_argument('--wandb_watch_freq', type=int, default=100,
                       help='Frequency to log model gradients and parameters')
    
    # Distributed Training Arguments
    parser.add_argument('--use_ddp', action='store_true', default=False,
                       help='Use DistributedDataParallel for training')
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='Local rank for distributed training (set by torchrun)')
    parser.add_argument('--world_size', type=int, default=1,
                       help='Number of processes in distributed training')
    parser.add_argument('--rank', type=int, default=0,
                       help='Global rank of current process')
    parser.add_argument('--dist_backend', type=str, default='nccl',
                       help='Distributed backend (nccl, gloo)')
    parser.add_argument('--dist_url', type=str, default='env://',
                       help='URL used to set up distributed training')
    parser.add_argument('--sync_bn', action='store_true', default=False,
                       help='Use synchronized batch normalization')
    
    return parser