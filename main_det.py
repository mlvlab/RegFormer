"""
Utilities for training, testing and caching results
for HICO-DET and V-COCO evaluations.

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = PROJECT_ROOT / "models"
DETECTOR_ROOT = MODEL_ROOT / "detector"
CLIP_ROOT = MODEL_ROOT / "CLIP"
IMPORT_PATHS = (
    PROJECT_ROOT,
    MODEL_ROOT,
    CLIP_ROOT,
    DETECTOR_ROOT,
    DETECTOR_ROOT / "detr",
)
for path in reversed(IMPORT_PATHS):
    path_str = str(path)
    if path_str in sys.path:
        sys.path.remove(path_str)
    sys.path.insert(0, path_str)

from models.detector.hoi_detector import build_detector
from core.det_engine.detection_engine import custom_collate, CustomisedDLE, DataFactory
# from utils_tip_cache_and_union_finetune import custom_collate, CustomisedDLE, DataFactory
from utils.det import vcoco_text_label
from utils.det.args import training_free_args

import torch
import random
import warnings
import argparse
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler

import pdb, json
warnings.filterwarnings("ignore")

def vcoco_official_evaluation(cache_dir):
    from vsrl_eval import VCOCOeval
    vsrl_ann_file = "./vcoco/data/vcoco/vcoco_test.json"
    coco_file = "./vcoco/data/instances_vcoco_all_2014.json"
    split_file = "./vcoco/data/splits/vcoco_test.ids"
    
    det_file = os.path.join(cache_dir, 'cache.pkl')
    vcocoeval = VCOCOeval(vsrl_ann_file, coco_file, split_file)
    role2_ap = vcocoeval._do_eval(det_file, ovr_thresh=0.5)
    return role2_ap

def vcoco_class_corr():
    """
        Class correspondence matrix in zero-based index
        [
            [hoi_idx, obj_idx, verb_idx],
            ...
        ]

        Returns:
            list[list[3]]
        """
    class_corr = []
    for i, (k, v) in enumerate(vcoco_text_label.vcoco_hoi_text_label.items()):
        class_corr.append([i, k[1], k[0]])
    return class_corr

def vcoco_object_n_verb_to_interaction(num_object_cls, num_action_cls, class_corr):
        """
        The interaction classes corresponding to an object-verb pair

        HICODet.object_n_verb_to_interaction[obj_idx][verb_idx] gives interaction class
        index if the pair is valid, None otherwise

        Returns:
            list[list[117]]
        """
        lut = np.full([num_object_cls, num_action_cls], None)
        for i, j, k in class_corr:
            lut[j, k] = i
        return lut.tolist()

def vcoco_object_to_interaction(num_object_cls, _class_corr):
        """
        class_corr: List[(x["action_id"], x["object_id"], x["id"])]
        
        Returns:
            list[list]
        """
        obj_to_int = [[] for _ in range(num_object_cls)]
        for corr in _class_corr:
            obj_to_int[corr[1]].append(corr[0])
        return obj_to_int

def vcoco_interaction_to_verb(_class_corr):
        """
        interaction to verb

        Returns:
            list[list]
        """ 

        inter_to_verb = []
        for i, corr in enumerate(_class_corr):
            inter_to_verb.append(corr[2])
        return inter_to_verb

def main(rank, args):

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=rank
    )
    args.class_wise_ap=True
    # Fix seed
    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.set_device(rank)
    args.clip_model_name = args.clip_dir_vit.split('/')[-1].split('.')[0]
    if args.clip_model_name == 'ViT-B-16':
        args.clip_model_name = 'ViT-B/16' 
    elif args.clip_model_name == 'ViT-L-14-336px':
        args.clip_model_name = 'ViT-L/14@336px'
    
    if args.backbone == 'resnet101':
        detr_backbone = 'R101-DC5' if args.dilation else 'R101'
    elif args.backbone == 'resnet50':
        detr_backbone = 'R50'
    elif args.backbone == 'swin_large':
        detr_backbone = 'SwinL'
    else: 
        raise NotImplementedError("Backbone should be in [resnet50, resnet101, swin_large]")
    print('[INFO]: detr backbone:', detr_backbone)

    weak_model = None
    saved_args = None
    if args.use_weak_model:
        assert args.trainin_free_option in ['contrastive', 'union'], f"weak model only supports contrastive and union decoding"
        assert args.use_attention_reweighting, f"weak model requires attention reweighting"
        from models.model import VisionTextAlignmentModel
        from core.arguments import model_args
        from utils.label_utils import get_class_labels, get_verb_object_indices
        parser = model_args()
        weak_args = parser.parse_known_args()[0]
        print(f'load weak model from {args.weak_model_ckpt}')
        ckpt = torch.load(args.weak_model_ckpt, map_location='cpu')
        saved_args = ckpt['args']
        for key, value in vars(saved_args).items():
            setattr(weak_args, key, value)
        # args.eval_only = True
        # args.finetuned_ckpt = args.weak_model_ckpt
        # args.use_ddp = False
        # args.use_wandb = False
        weak_args.return_attention_weight = args.debug_attn
        weak_args.vis_instance_score = args.vis_instance_score
        weak_args.instance_score_scheme = args.instance_score_scheme
        weak_args.vis_label_weight = args.debug_visualize
        if args.interactiveness_scale_factor is not None:
            weak_args.instance_prior_factor = args.interactiveness_scale_factor
        if args.instance_score_post_masking_type is not None:
            weak_args.instance_score_post_masking_type = args.instance_score_post_masking_type        
        verb_object_indices = get_verb_object_indices(weak_args.dataset_name)
        weak_model = VisionTextAlignmentModel(verb_object_indices, args=weak_args)
        weak_model.use_det_results = True
        
        weak_model.attention_pooling.instance_score_post_masking_type = args.instance_score_post_masking_type
        weak_model.attention_pooling.post_sum_scale = args.post_sum_scale
        weak_model.attention_pooling.mask_generation_lib = args.mask_generation_lib
        if args.local_scale_factor is not None:
            weak_model.attention_pooling.local_scale_factor = args.local_scale_factor
        if args.use_masked_global_instance_score_for_sgpq:
            weak_model.attention_pooling.use_masked_global_instance_score_for_sgpq = True
        # weak_model.attention_pooling.debug_visualize = args.debug_visualize
        # weak_model.attention_pooling.mask_generation_lib = args.mask_generation_lib
        model_state_dict = ckpt['model_state_dict']
        model_dict = weak_model.state_dict()
        
        # Filter out keys that don't exist in the model
        for key, value in model_dict.items():
            if key in model_state_dict:
                model_dict[key] = model_state_dict[key]
        weak_model.load_state_dict(model_dict)
        # weak_model = weak_model.to(device=args.device)
    trainset = DataFactory(name=args.dataset, partition=args.partitions[0], data_root=args.data_root, clip_model_name=args.clip_model_name, detr_backbone=detr_backbone,
                           max_num_duplicate_object=args.max_num_duplicate_object, args=args, weak_model=weak_model, weak_model_args=saved_args)
    testset = DataFactory(name=args.dataset, partition=args.partitions[1], data_root=args.data_root, clip_model_name=args.clip_model_name, detr_backbone=detr_backbone,
                           max_num_duplicate_object=args.max_num_duplicate_object, args=args, weak_model=weak_model, weak_model_args=saved_args)
    # trainset[0][1]: dict_keys(['boxes_h', 'boxes_o', 'hoi', 'object', 'verb', 'orig_size', 'labels', 'size', 'filename'])
    # trainset[0][0]: (torch.Size([3, 814, 640]), torch.Size([3, 224, 224]))

    if args.dataset == 'vcoco':
        class_corr = vcoco_class_corr()
        trainset.dataset.class_corr = class_corr
        testset.dataset.class_corr = class_corr
        object_n_verb_to_interaction = vcoco_object_n_verb_to_interaction(num_object_cls=len(trainset.dataset.objects), num_action_cls=len(trainset.dataset.actions), class_corr=class_corr)
        trainset.dataset.object_n_verb_to_interaction = object_n_verb_to_interaction
        testset.dataset.object_n_verb_to_interaction = object_n_verb_to_interaction
        object_to_interaction = vcoco_object_to_interaction(num_object_cls=len(trainset.dataset.objects), _class_corr=class_corr)
        trainset.dataset.object_to_interaction = object_to_interaction
        testset.dataset.object_to_interaction = object_to_interaction
        interaction_to_verb = vcoco_interaction_to_verb(_class_corr=class_corr)
        trainset.dataset.interaction_to_verb = interaction_to_verb
        testset.dataset.interaction_to_verb = interaction_to_verb
    
    train_loader = DataLoader(
        dataset=trainset,
        collate_fn=custom_collate, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        sampler=DistributedSampler(
            trainset, 
            num_replicas=args.world_size, 
            rank=rank)
    )
    test_loader = DataLoader(
        dataset=testset,
        collate_fn=custom_collate, batch_size=1,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        # sampler=torch.utils.data.SequentialSampler(testset)
        sampler =DistributedSampler(
        testset, shuffle=False, drop_last=False
        )
    )

    args.human_idx = 0
    if args.dataset == 'hicodet':
        if args.use_verb_only:
            object_to_target = train_loader.dataset.dataset.object_to_verb
            args.num_classes = 117
        else:
            object_to_target = train_loader.dataset.dataset.object_to_interaction
            args.num_classes = 600
        
    elif args.dataset == 'vcoco':
        object_to_target = list(train_loader.dataset.dataset.object_to_action.values())
        args.num_classes = 24
        args.use_verb_only = True
    elif args.dataset == 'swig':
        raise NotImplementedError("SWIG is not supported yet")
    # print("[INFO]: obj2target", object_to_target)
    print('[INFO]: num_classes:', args.num_classes)
    if args.dataset == 'vcoco':
        num_anno = None
    else:
        num_anno = torch.as_tensor(trainset.dataset.anno_interaction)
    kwargs = {}
    if weak_model is not None:
        kwargs['weak_model'] = weak_model
        kwargs['weak_model_args'] = saved_args
    upt = build_detector(args, object_to_target, num_anno, **kwargs)

    # train_loader.dataset.net = upt
    # test_loader.dataset.net = upt
    # train_loader.dataset.human_index = upt.human_idx
    # train_loader.dataset.nms_region_proposals = upt.prepare_region_proposals
    # train_loader.dataset.naive_region_proposals = upt.get_region_proposals
    
    # test_loader.dataset.human_index = upt.human_idx
    # test_loader.dataset.nms_region_proposals = upt.prepare_region_proposals
    # test_loader.dataset.naive_region_proposals = upt.get_region_proposals
    
    if os.path.exists(args.resume):
        print(f"=> Rank {rank}: continue from saved checkpoint {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        upt.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"=> Rank {rank}: start from a randomly initialised model")

    if args.wandb and rank == 0:
        import wandb
        wandb.init(project=args.project_name, 
                   id=args.wandb_id,
                   group=args.group_name, 
                   name=args.run_name or args.output_dir,
                   resume='allow',
                   config=args if args.wandb_id is None else None)
        wandb.watch(upt)

    engine = CustomisedDLE(
        upt, train_loader,
        max_norm=args.clip_max_norm,
        num_classes=args.num_classes,
        args=args,
        print_interval=args.print_interval,
        find_unused_parameters=True,
        cache_dir=args.output_dir,
    )

    if args.cache:
        if args.dataset == 'hicodet':
            engine.cache_hico(test_loader, args.output_dir)
        elif args.dataset == 'vcoco':
            engine.cache_vcoco(test_loader, args.output_dir)
            role2_ap = vcoco_official_evaluation(args.output_dir)
            if rank == 0:
                role2_ap_w_point = np.nanmean(role2_ap)*100
                role2_ap_w_o_point = (np.nanmean(role2_ap) * 25 - role2_ap[-3][0]) / 24 * 100.00
                print(f"Role2 AP (w/point): {role2_ap_w_point:.2f}")
                print(f"Role2 AP (w/o point): {role2_ap_w_o_point:.2f}")
                if args.wandb:
                    wandb.log({"Role2_AP": role2_ap_w_point, "Role2_AP_w_o_point": role2_ap_w_o_point})
        return

    if args.eval:
        if args.dataset == 'vcoco':
            raise NotImplementedError(f"Evaluation on V-COCO has not been implemented.")
        from utils.det.hico_text_label import hico_unseen_index
        
        ap_dict = engine.test_hico(test_loader, args)
        save_dict= {}
        if rank == 0:
            ap = ap_dict['class_meter']
            ap = ap.eval()
            save_dict.update({'all_ap': ap})
            # Fetch indices for rare and non-rare classes
            num_anno = torch.as_tensor(trainset.dataset.anno_interaction)
            rare = torch.nonzero(num_anno < 10).squeeze(1)
            non_rare = torch.nonzero(num_anno >= 10).squeeze(1)

            weak_zs_type = getattr(saved_args, 'zs_type', None) if saved_args is not None else None
            zero_shot_labels = {
                'rare_first': 'RF-UC',
                'non_rare_first': 'NF-UC',
                'unseen_verb': 'UV',
                'unseen_object': 'UO',
                'uc0': 'UC0',
                'uc1': 'UC1',
                'uc2': 'UC2',
                'uc3': 'UC3',
                'uc4': 'UC4',
            }

            if weak_zs_type in hico_unseen_index:
                unseen_indices = set(hico_unseen_index[weak_zs_type])
                ap_unseen = []
                ap_seen = []
                for i, value in enumerate(ap):
                    if i in unseen_indices:
                        ap_unseen.append(value)
                    else:
                        ap_seen.append(value)
                ap_unseen = torch.as_tensor(ap_unseen).mean()
                ap_seen = torch.as_tensor(ap_seen).mean()
                eval_label = zero_shot_labels.get(weak_zs_type, weak_zs_type)
                print(f"{eval_label} eval - unseen: {ap_unseen:.4f}, seen: {ap_seen:.4f}, full: {ap.mean():.4f}")
                if args.wandb and rank == 0:
                    wandb.log({
                        f"{eval_label}_unseen": ap_unseen,
                        f"{eval_label}_seen": ap_seen,
                        f"{eval_label}_full": ap.mean(),
                    })
            else:
                print(
                    f"The mAP is {ap.mean()*100:.2f},"
                    f" rare: {ap[rare].mean()*100:.2f},"
                    f" none-rare: {ap[non_rare].mean()*100:.2f},"
                )
                if args.wandb and rank == 0:
                    wandb.log({"full": ap.mean()*100, "rare": ap[rare].mean()*100, "non_rare": ap[non_rare].mean()*100})
            
            # Print distance statistics
            if 'distance_meter' in ap_dict:
                distance_ap = ap_dict['distance_meter']
                if args.class_wise_ap:
                    distance_ap = [meter.eval().mean() for meter in distance_ap]
                else:
                    distance_ap = distance_ap.eval()
                save_dict.update({'distance_ap': distance_ap})
                print(f"\n=== Distance Statistics ===")
                print(f"Max distance: {args.max_distance}")
                print(f"Distance interval: {args.distance_interval}")
                
                # Print distance AP statistics if available
                if distance_ap is not None:
                    num_intervals = int(args.max_distance / args.distance_interval) + 1
                    print(f"Distance AP by intervals:")
                    for i in range(num_intervals):
                        if i < len(distance_ap):
                            start_dist = i * args.distance_interval
                            end_dist = (i + 1) * args.distance_interval
                            if i == num_intervals - 1:
                                print(f"  [{start_dist:.2f}+]: {distance_ap[i]*100:.2f}")
                                if args.wandb and rank == 0:
                                    wandb.log({f"distance_ap({start_dist:.2f}+)": distance_ap[i]*100})
                            else:
                                print(f"  [{start_dist:.2f}-{end_dist:.2f}): {distance_ap[i]*100:.2f}")
                                if args.wandb and rank == 0:
                                    wandb.log({f"distance_ap({start_dist:.2f}-{end_dist:.2f})": distance_ap[i]*100})
                            # save_dict.update({f"distance_ap({start_dist:.2f}-{end_dist:.2f})": distance_ap[i]*100})
                print("============================\n")
            if rank == 0:
                import pickle                
                with open(os.path.join(args.output_dir, "eval_results.pkl"), "wb") as f:
                    pickle.dump(save_dict, f)
        return

    for p in upt.detector.parameters():
        p.requires_grad = False
    for n, p in upt.named_parameters():
        if n.startswith('adapter'):
            p.requires_grad = True
        else:
            p.requires_grad = False

    for n, p in upt.clip_model.named_parameters():
        if n.startswith('visual.positional_embedding') or n.startswith('visual.ln_post') or n.startswith('visual.proj') : 
            p.requires_grad = True
        else: p.requires_grad = False
    
    param_dicts = [{
        "params": [p for n, p in upt.named_parameters()
        if p.requires_grad]
    }]
    # print(param_dicts)
    n_parameters = sum(p.numel() for p in upt.parameters() if p.requires_grad)

    print('number of params:', n_parameters)
 
    optim = torch.optim.AdamW(
        param_dicts, lr=args.lr_head,
        weight_decay=args.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, args.lr_drop)
    if args.resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch=checkpoint['epoch']
        iteration = checkpoint['iteration']
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    # Override optimiser and learning rate scheduler
        engine.update_state_key(optimizer=optim, lr_scheduler=lr_scheduler, epoch=epoch,iteration=iteration, scaler=scaler)
    else:
        engine.update_state_key(optimizer=optim, lr_scheduler=lr_scheduler)
   
    engine(args.epochs)


@torch.no_grad()
def sanity_check(args):
    dataset = DataFactory(name='hicodet', partition=args.partitions[0], data_root=args.data_root)
    args.human_idx = 0; args.num_classes = 117
    object_to_target = dataset.dataset.object_to_verb
    upt = build_detector(args, object_to_target)
    if args.eval:
        upt.eval()

    image, target = dataset[0]
    outputs = upt([image], [target])

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(parents=[training_free_args()])
    parser.add_argument('--lr-head', default=1e-4, type=float)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr-drop', default=10, type=int)
    parser.add_argument('--clip-max-norm', default=0.1, type=float)

    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--dilation', action='store_true')
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

    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--gamma', default=0.2, type=float)

    parser.add_argument('--dataset', default='hicodet', type=str)
    parser.add_argument('--partitions', nargs='+', default=['train2015', 'test2015'], type=str)
    parser.add_argument('--num-workers', default=4, type=int)
    parser.add_argument('--data-root', default='./hicodet')

    # training parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--port', default='1233', type=str)
    parser.add_argument('--seed', default=66, type=int)
    parser.add_argument('--pretrained', default='', help='Path to a pretrained detector')
    parser.add_argument('--resume', default='', help='Resume from a model')
    parser.add_argument('--output-dir', default='checkpoints')
    parser.add_argument('--print-interval', default=500, type=int)
    parser.add_argument('--world-size', default=1, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--sanity', action='store_true')
    parser.add_argument('--box-score-thresh', default=0.2, type=float)
    parser.add_argument('--fg-iou-thresh', default=0.5, type=float)
    parser.add_argument('--min-instances', default=3, type=int)
    parser.add_argument('--max-instances', default=15, type=int)

    parser.add_argument('--visual_mode', default='vit', type=str)
    # add CLIP vision
    parser.add_argument('--clip_dir_vit', default='./checkpoints/pretrained_clip/ViT-B-16.pt', type=str)

    
    ### ViT-B-16 START
    parser.add_argument('--clip_visual_layers_vit', default=12, type=list)
    parser.add_argument('--clip_visual_output_dim_vit', default=512, type=int)
    parser.add_argument('--clip_visual_input_resolution_vit', default=224, type=int)
    parser.add_argument('--clip_visual_width_vit', default=768, type=int)
    parser.add_argument('--clip_visual_patch_size_vit', default=16, type=int)
    parser.add_argument('--clip_text_transformer_width_vit', default=512, type=int)
    parser.add_argument('--clip_text_transformer_heads_vit', default=8, type=int)
    parser.add_argument('--clip_text_transformer_layers_vit', default=12, type=int)
    # ---END----ViT-B-16-----END-----
    
    parser.add_argument('--clip_text_context_length_vit', default=13, type=int)

    parser.add_argument('--post_process', default=False, action='store_true')
    parser.add_argument('--num_shot', default=1, type=int)

    parser.add_argument('--use_kmeans', default=False, action='store_true')
    parser.add_argument('--file1', default='union_embeddings_cachemodel_crop_padding_zeros_vitb16.p',type=str)
    parser.add_argument('--logits_type', default='HO+U+T', type=str, choices=['HO+U+T', 'U+T', 'HO+T', 'T', 'HO', 'U', "HO+U"]) # 13 -77 # text_add_visual, visual
    # parser.add_argument('--vis_feature_type', default='hum_obj_uni', type=str, choices=('hum_obj_uni', 'hum_uni', 'hum_obj', 'uni'))
    parser.add_argument('--gamma_HO', type=float, default=0.5)
    parser.add_argument('--gamma_U', type=float, default=0.5)
    parser.add_argument('--use_multi_hot', action='store_true')
    parser.add_argument('--temperature', type=float, default=0.05)
    parser.add_argument('--label_choice', default='random', choices=['random', 'single_first', 'multi_first', 'single+multi', 'rare_first', 'non_rare_first', 'rare+non_rare'])
    parser.add_argument('--rm_duplicate_feat', action='store_true')
    parser.add_argument('--sample_choice', default='uniform', choices=['uniform', 'origin'])
    parser.add_argument('--dic_key', type=str, default='hoi', choices=['hoi', 'verb', 'object'])
    
    parser.add_argument('--zs', action='store_true') ## zero-shot
    parser.add_argument('--zs_type', type=str, default='rare_first', choices=['rare_first', 'non_rare_first', 'unseen_verb', 'uc0', 'uc1', 'uc2', 'uc3', 'uc4'])

    parser.add_argument('--repeat_factor_sampling', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='apply repeat factor sampling to increase the rate at which tail categories are observed')
    
    ## **************** arguments for deformable detr **************** ##
    parser.add_argument('--d_detr', default=False, type=lambda x: (str(x).lower() == 'true'),)
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    # Variants of Deformable DETR
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
    
    args = parser.parse_args()
    print(args)

    if args.sanity:
        sanity_check(args)
        sys.exit()

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.port
    # mp.spawn(main, nprocs=args.world_size, args=(args,))
    if args.world_size==1:
        main(0,args)
    else:
        mp.spawn(main, nprocs=args.world_size, args=(args,))
