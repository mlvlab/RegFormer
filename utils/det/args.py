import argparse
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--lr-drop', default=10, type=int)
    parser.add_argument('--clip-max-norm', default=0.1, type=float)
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--position-embedding', default='sine', type=str, choices=('sine', 'learned'))


    parser.add_argument('--repr-dim', default=512, type=int)
    parser.add_argument('--hidden-dim', default=256, type=int)
    parser.add_argument('--enc-layers', default=6, type=int)
    parser.add_argument('--dec-layers', default=6, type=int)
    parser.add_argument('--dim-feedforward', default=2048, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num-queries', default=100, type=int)
    parser.add_argument('--pre-norm', action='store_true')

    parser.add_argument('--no-aux-loss', dest='aux_loss', action='store_false')
    parser.add_argument('--set-cost-class', default=1, type=float)
    parser.add_argument('--set-cost-bbox', default=5, type=float)
    parser.add_argument('--set-cost-giou', default=2, type=float)
    parser.add_argument('--bbox-loss-coef', default=5, type=float)
    parser.add_argument('--giou-loss-coef', default=2, type=float)
    parser.add_argument('--eos-coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")


    # training parameters
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--box-score-thresh', default=0.2, type=float)
    parser.add_argument('--fg-iou-thresh', default=0.5, type=float)
    parser.add_argument('--min-instances', default=3, type=int)
    parser.add_argument('--max-instances', default=15, type=int)

    # add CLIP model resenet
    # parser.add_argument('--clip_dir', default='./checkpoints/pretrained_clip/RN50.pt', type=str)
    # parser.add_argument('--clip_visual_layers', default=[3, 4, 6, 3], type=list)
    # parser.add_argument('--clip_visual_output_dim', default=1024, type=int)
    # parser.add_argument('--clip_visual_input_resolution', default=1344, type=int)
    # parser.add_argument('--clip_visual_width', default=64, type=int)
    # parser.add_argument('--clip_visual_patch_size', default=64, type=int)
    # parser.add_argument('--clip_text_output_dim', default=1024, type=int)
    # parser.add_argument('--clip_text_transformer_width', default=512, type=int)
    # parser.add_argument('--clip_text_transformer_heads', default=8, type=int)
    # parser.add_argument('--clip_text_transformer_layers', default=12, type=int)
    # parser.add_argument('--clip_text_context_length', default=13, type=int)

    #### add CLIP vision transformer

    ### ViT-L/14@336px START: emb_dim: 768
    # >>> vision_width: 1024,  vision_patch_size(conv's kernel-size&&stride-size): 14,
    # >>> vision_layers(#layers in vision-transformer): 24 ,  image_resolution:336;
    # >>> transformer_width:768, transformer_layers: 12, transformer_heads:12
    parser.add_argument('--clip_visual_layers_vit', default=24, type=list)
    parser.add_argument('--clip_visual_output_dim_vit', default=768, type=int)
    parser.add_argument('--clip_visual_input_resolution_vit', default=336, type=int)
    parser.add_argument('--clip_visual_width_vit', default=1024, type=int)
    parser.add_argument('--clip_visual_patch_size_vit', default=14, type=int)

    # parser.add_argument('--clip_text_output_dim_vit', default=512, type=int)
    parser.add_argument('--clip_text_transformer_width_vit', default=768, type=int)
    parser.add_argument('--clip_text_transformer_heads_vit', default=12, type=int)
    parser.add_argument('--clip_text_transformer_layers_vit', default=12, type=int)
    # ---END----ViT-L/14@336px----END----

    ### ViT-B-16 START
    # parser.add_argument('--clip_visual_layers_vit', default=12, type=list)
    # parser.add_argument('--clip_visual_output_dim_vit', default=512, type=int)
    # parser.add_argument('--clip_visual_input_resolution_vit', default=224, type=int)
    # parser.add_argument('--clip_visual_width_vit', default=768, type=int)
    # parser.add_argument('--clip_visual_patch_size_vit', default=16, type=int)

    # # parser.add_argument('--clip_text_output_dim_vit', default=512, type=int)
    # parser.add_argument('--clip_text_transformer_width_vit', default=512, type=int)
    # parser.add_argument('--clip_text_transformer_heads_vit', default=8, type=int)
    # parser.add_argument('--clip_text_transformer_layers_vit', default=12, type=int)
    # ---END----ViT-B-16-----END-----
    parser.add_argument('--clip_text_context_length_vit', default=77, type=int) # 13 -77

    parser.add_argument('--feat_mask_type', type=int, default=0,) # 0: dropout(random mask); 1: None

    parser.add_argument('--repeat_factor_sampling', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='apply repeat factor sampling to increase the rate at which tail categories are observed')

    parser.add_argument('--dataset_file', default='coco')

    ## **************** arguments for deformable detr **************** ##
    parser.add_argument('--d_detr', default=False, type=lambda x: (str(x).lower() == 'true'),)
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    # Variants of Deformable `DETR`
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')
    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')
    # * Transformer
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)


    ## **************** arguments for deformable detr **************** ##


    parser.add_argument('--lr-head', default=1e-3, type=float)
    parser.add_argument('--lr-vit', default=1e-3, type=float)

    parser.add_argument('--zs', action='store_true') ## zero-shot
    parser.add_argument('--zs_type', type=str, default='rare_first', choices=['rare_first', 'non_rare_first', 'unseen_verb', 'uc0', 'uc1', 'uc2', 'uc3', 'uc4','unseen_object'])

    parser.add_argument('--dataset', default='hicodet', type=str)
    parser.add_argument('--partitions', nargs='+', default=['train2015', 'test2015'], type=str)
    parser.add_argument('--num_classes', type=int, default=117,)
    parser.add_argument('--data-root', default='./hicodet')


    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--num-workers', default=2, type=int)
    parser.add_argument('--eval', action='store_true')

    parser.add_argument('--clip_dir_vit', default='./checkpoints/pretrained_clip/ViT-B-16.pt', type=str)
    parser.add_argument('--pretrained', default='', help='Path to a pretrained detector')
    parser.add_argument('--resume', default='', help='Resume from a model')
    parser.add_argument('--output-dir', default='checkpoints')


    parser.add_argument('--use_hotoken', action='store_true')
    parser.add_argument('--use_prior', action='store_true')
    parser.add_argument('--use_exp', action='store_true')

    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--gamma', default=0.2, type=float)
    parser.add_argument('--hyper_lambda', type=float, default=2.8)

    # adapter
    parser.add_argument('--use_insadapter', action='store_true')
    parser.add_argument('--adapter_num_layers', type=int, default=1)
    parser.add_argument('--adapt_dim', default=32, type=int)
    parser.add_argument('--adapter_alpha', default=1., type=float)
    parser.add_argument('--adapter_pos', type=str, default='all', choices=['all', 'front', 'end', 'random', 'last', '03','47','811'])
    parser.add_argument('--adapter_scalar', default='learnable_scalar', type=str)

    ## prompt learning
    parser.add_argument('--use_prompt', action='store_true')
    parser.add_argument('--N_CTX', type=int, default=24)  # number of context vectors
    parser.add_argument('--CSC', action='store_true')  # class-specific context
    parser.add_argument('--CTX_INIT', type=str, default='')  # initialization words
    parser.add_argument('--CLASS_TOKEN_POSITION', type=str, default='end')  # # 'middle' or 'end' or 'front'

    # miscellaneous
    parser.add_argument('--job_id', default=1985, type=int)
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')


    parser.add_argument('--seed', default=66, type=int)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--port', default='7894', type=str)
    parser.add_argument('--print-interval', default=100, type=int)
    parser.add_argument('--world-size', default=1, type=int)
    
    parser.add_argument('--visualize_results', default=False, action='store_true')
    parser.add_argument('--no_detector_embeds', default=False, action='store_true')
    parser.add_argument('--position_embedding_dim', default=256, type=int)

    return parser.parse_args()


def training_free_args():
    parser = argparse.ArgumentParser(add_help=False)
    
    parser.add_argument('--use_open_clip', default=False, action='store_true')
    parser.add_argument('--open_clip_model_name', default='ViT-B-16-SigLIP', type=str)
    parser.add_argument('--open_clip_pretrained', default='webli', type=str)
    parser.add_argument('--clip_input_resolution', default=None, type=int)
    # option for training-free
    parser.add_argument('--trainin_free_option', default='ada_cm', choices=['ada_cm','union','contrastive','single'])
    parser.add_argument('--max_num_duplicate_object', default=None, type=int)
    parser.add_argument('--eval_mode', default='all', choices=['single','multi','all'], type=str)
    
    # pre-processing
    parser.add_argument('--no_zero_padding', default=False, action='store_true')
    parser.add_argument('--mask_type', default='hard', choices=['hard','soft','blur'], type=str)
    parser.add_argument('--input_type', default='union', choices=['union','full'], type=str)
    parser.add_argument('--use_nms', default=False, action='store_true')
    
    # post-processing
    parser.add_argument('--prior_scale_factor', default=None, type=float, help = 'detector scale factor')
    parser.add_argument('--post_process_type', default='softmax', choices=['softmax','sigmoid','none'], type=str)
    parser.add_argument('--similarity_normalization', default=False, action='store_true', help = 'cosine similarity normalization')
    
    parser.add_argument('--sigmoid_factor_type', default='fixed', choices=['fixed','cache'], type=str)
    parser.add_argument('--sigmoid_scale_factor', default=1.0, type=float, help = 'scale factor for sigmoid post-processing')
    parser.add_argument('--sigmoid_shift_factor', default=0.0, type=float, help = 'shift factor for sigmoid post-processing')
    
    # contrastive decoding
    # parser.add_argument('--contrastive_guidance', default=False, action='store_true', )
    parser.add_argument('--contrastive_guidance_type', default='fix', choices=['fix','prob_based'], type=str, )
    parser.add_argument('--contrastive_guidance_strength', default=1.0, type=float, )
    parser.add_argument('--contrastive_first', default='',type=str, help= 'minuend of contrastive decoding')
    parser.add_argument('--contrastive_second', default='mask_sub_obj',type=str, help= 'subtrahend of contrastive decoding')    
    parser.add_argument('--contrast_only_target_class', default=False, action='store_true', )
    parser.add_argument('--prob_based_scale_factor', default=1.0, type=float, help = 'scale factor for CD scale factor')
    parser.add_argument('--use_neutral_difference', default=False, action='store_true', )
    # parser.add_argument('--union_masking_type', default='um', choices=['um','sm','om','som'], type=str, 
    #                     help='union masking type. um : union masking, sm : subject masking, om : object masking, som : subject and object masking')
    
    # model forward
    parser.add_argument('--use_attention_reweighting', default=False, action='store_true')
    parser.add_argument('--attention_reweighting_start_layer', default=0, type=int, help='0 means the first layer, and -1 means not use reweighting in the model')
    parser.add_argument('--attention_reweighting_end_layer', default=-1, type=int, help='-1 means the last layer')
    parser.add_argument('--attn_pool_mask', default=False, action='store_true', help='mask for attn_pool, this is only for SigLIP')
    parser.add_argument('--roi_align_layer', default=0, type=int, help='0 means the first layer, it is for full input type')
    parser.add_argument('--image_level_pooling', default=False, action='store_true')
    parser.add_argument('--custom_attn_pool', default=False, action='store_true')
    parser.add_argument('--return_attention_weight', default=False, action='store_true')
    
    # visualize results
    parser.add_argument('--visualize_results', default=False, action='store_true')
    parser.add_argument('--vis_images', default=False, action='store_true')
    
    # eval
    parser.add_argument('--save_results_as_dict', default=False, action='store_true')
    parser.add_argument('--eval_with_saved_results', default=False, action='store_true')
    parser.add_argument('--use_verb_only', default=False, action='store_true')
    
    parser.add_argument('--eval_distance', default=False, action='store_true')
    parser.add_argument('--class_wise_ap', default=False, action='store_true')
    parser.add_argument('--distance_interval', default=0.1, type=float)
    parser.add_argument('--max_distance', default=0.5, type=float)
    
    parser.add_argument('--online_detection', default=False, action='store_true')
    
    parser.add_argument('--custom_detector_results_path', default=None, type=str, help='path to the custom detector results')
    
    # weak model
    parser.add_argument('--use_weak_model', default=False, action='store_true')
    parser.add_argument('--weak_model_ckpt', default='', type=str)
    parser.add_argument('--instance_score_scheme', type=str, default='image')
    parser.add_argument('--interactiveness_scale_factor', default=None, type=float)
    parser.add_argument('--force_no_attention_modulation', default=False, action='store_true')
    parser.add_argument('--instance_score_post_masking_type', default=None, choices=['pre_sum','post_sum'])
    parser.add_argument('--post_sum_scale', default=None, type=float)
    parser.add_argument('--mask_generation_lib', default='numpy', choices=['numpy','torch'], type=str)
    parser.add_argument('--local_scale_factor', default=None, type=float)
    parser.add_argument('--use_masked_global_instance_score_for_sgpq', default=False, action='store_true')
    
    # wandb
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--wandb_id', default=None, type=str)
    parser.add_argument('--project_name', default='RCD_HOI', type=str)
    parser.add_argument('--group_name', default='training_free', type=str)
    parser.add_argument('--run_name', default='', type=str)
    
    # profiling
    parser.add_argument('--debug_attn', default=False, action='store_true')
    parser.add_argument('--vis_instance_score', default=False, action='store_true')
    parser.add_argument('--debug_labels', default=False, action='store_true')
    parser.add_argument('--debug_visualize', default=False, action='store_true')
    return parser


def advanced_detector_args():
    """Arguments for building advanced variants of DETR"""
    parser = argparse.ArgumentParser(add_help=False)
    # Backbone
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position-embedding', default='sine', type=str, choices=('sine', 'learned'))
    parser.add_argument('--position-embedding-scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num-feature-levels', default=4, type=int, help='number of feature levels')
    parser.add_argument("--drop-path-rate", default=0.2, type=float)
    parser.add_argument("--pretrained_backbone_path", default=None, type=str)

    # Transformer
    parser.add_argument('--hidden-dim', default=256, type=int)
    parser.add_argument('--enc-layers', default=6, type=int)
    parser.add_argument('--dec-layers', default=6, type=int)
    parser.add_argument('--dim-feedforward', default=2048, type=int)
    parser.add_argument('--dropout', default=.0, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument("--num-queries-one2one", default=300, type=int,
                        help="Number of query slots for one-to-one matching",)

    # Hybrid matching settings
    parser.add_argument('--num-queries-one2many', default=0, type=int,
                        help="Number of query slots for one-to-many matchining",)

    # Segmentation
    parser.add_argument('--masks', action="store_true")

    # Deformable transformer
    parser.add_argument('--dec-n-points', default=4, type=int)
    parser.add_argument('--enc-n-points', default=4, type=int)
    parser.add_argument('--no-box-refine', dest="with_box_refine",
                        default=True, action='store_false')
    parser.add_argument('--no-two-stage', dest="two_stage",
                        default=True, action='store_false')

    # Tricks
    parser.add_argument("--no-mixed-selection", dest="mixed_selection",
                        action="store_false", default=True)
    parser.add_argument("--no-look-forward-twice", dest="look_forward_twice",
                        action="store_false", default=True)

    # Training
    parser.add_argument('--lr-head', default=1e-4, type=float)
    parser.add_argument('--lr-backbone', default=0., type=float)
    parser.add_argument('--lr-drop', default=20, type=int)
    parser.add_argument('--lr-drop-factor', default=.2, type=float)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--clip-max-norm', default=.1, type=float)
    parser.add_argument("--use-checkpoint", default=False, action="store_true")

    # Evaluation
    parser.add_argument("--topk", default=100, type=int)

    # Loss
    parser.add_argument('--no-aux-loss', dest='aux_loss', action='store_false')
    parser.add_argument('--set-cost-class', default=2, type=float)
    parser.add_argument('--set-cost-bbox', default=5, type=float)
    parser.add_argument('--set-cost-giou', default=2, type=float)
    parser.add_argument("--mask-loss-coef", default=1, type=float)
    parser.add_argument("--dice-loss-coef", default=1, type=float)
    parser.add_argument("--cls-loss-coef", default=2, type=float)
    parser.add_argument('--bbox-loss-coef', default=5, type=float)
    parser.add_argument('--giou-loss-coef', default=2, type=float)
    parser.add_argument("--focal-alpha", default=0.25, type=float)

    # Misc.
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--dataset', default='hicodet', type=str)
    parser.add_argument('--partitions', nargs='+', default=['train2015', 'test2015'], type=str)
    parser.add_argument('--num-workers', default=2, type=int)
    parser.add_argument('--data-root', default='./hicodet')
    parser.add_argument('--output-dir', default='checkpoints')
    parser.add_argument('--pretrained', default='', help='Path to a pretrained detector')
    parser.add_argument('--print-interval', default=100, type=int)
    return parser
