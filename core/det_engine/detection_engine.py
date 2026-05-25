"""
Utilities

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

from cmath import nan
from code import interact
from fileinput import filename
from locale import normalize
import os
import torch
import pickle
import numpy as np
import scipy.io as sio
import json

from torchvision.transforms import Resize, CenterCrop
from torchvision.ops import box_iou
from torchvision.ops.boxes import batched_nms
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset

from hicodet.hicodet import HICODet
import pocket
from pocket.core import DistributedLearningEngine
from pocket.utils import DetectionAPMeter

import detr.datasets.transforms_clip as T
import pdb
import copy 
import pickle
import torch.nn.functional as F
import clip
from util import box_ops
from PIL import Image, ImageFilter
from core.det_engine.tools import forward_chunks

from utils.det.vis_utils import visualise_entire_image, visualize_gt_hoi
from utils.det.hico_text_label import hico_unseen_index
from utils.det.ops import BoxPairAssociation
import utils.det.ddp as ddp

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def custom_collate(batch):
    images = []
    targets = []
    
    for im, tar in batch:
        images.append(im)
        targets.append(tar)

    return images, targets

def load_detector_pickle(name, path_or_stem):
    candidates = []
    if path_or_stem.endswith(".p"):
        candidates.append(path_or_stem)
    else:
        candidates.append(f"{path_or_stem}.p")

    filename = os.path.basename(candidates[0])
    candidates.extend([
        os.path.join(f"{name}_pkl_files", filename),
        os.path.join("data", f"{name}_pkl_files", filename),
    ])

    for path in candidates:
        if os.path.exists(path):
            print(f"get anno_box from {path}")
            with open(path, "rb") as f:
                return pickle.load(f)
    raise FileNotFoundError(f"Could not find detector pickle for {path_or_stem}. Tried: {candidates}")

class DataFactory(Dataset):
    def __init__(self, name, partition, data_root, clip_model_name, detr_backbone, max_num_duplicate_object=None, score_threshold=0.5, args=None, weak_model=None, weak_model_args=None):
        if name not in ['hicodet', 'vcoco']:
            raise ValueError("Unknown dataset ", name)
        
        self.name = name
        
        self._load_features= False
        self.max_num_duplicate_object = max_num_duplicate_object
        self.score_threshold = score_threshold
        assert clip_model_name in ['ViT-L/14@336px', 'ViT-B/16']
        self.clip_model_name = clip_model_name
        if args.clip_input_resolution is not None:
            self.clip_input_resolution = args.clip_input_resolution
        else:
            if self.clip_model_name == 'ViT-B/16':
                self.clip_input_resolution = 224
            elif self.clip_model_name == 'ViT-L/14@336px':
                self.clip_input_resolution = 336

        if name == 'hicodet':
            # self._text_features = pickle.load(open('inference_features_vit16.p','rb'))
            assert partition in ['train2015', 'test2015'], \
                "Unknown HICO-DET partition " + partition
            self.dataset = HICODet(
                root=os.path.join(data_root, 'hico_20160224_det/images', partition),
                anno_file=os.path.join(data_root, 'instances_{}.json'.format(partition)),
                target_transform=pocket.ops.ToTensor(input_format='dict'),
                args=args
            )
            # anno_box_name = f'{name}_train_bbox_{detr_backbone}' if partition == 'train2015' else f'{name}_test_bbox_{detr_backbone}_{args.pretrained.rsplit("/",1)[-1].split(".")[0]}'
            # print(f'get anno_box from {name}_pkl_files/{anno_box_name}.p')
            if partition == 'train2015':
                anno_box_name = f'{name}_train_bbox_R50'
                if args.eval or args.cache:
                    self.anno_bbox = None
                else:
                    self.anno_bbox = load_detector_pickle(name, anno_box_name)
            else:
                # if 'swin_large' in args.pretrained:
                #     detr_backbone = 'SwinL'
                if args.custom_detector_results_path is not None:
                    anno_box_name = args.custom_detector_results_path
                else:
                    anno_box_name = f'{name}_test_bbox_{detr_backbone}_{args.pretrained.rsplit("/",1)[-1].rsplit(".",1)[0]}'
                self.anno_bbox = load_detector_pickle(name, anno_box_name)
            
            # pdb.set_trace()
        else:
            from vcoco.vcoco import VCOCO

            assert partition in ['train', 'val', 'trainval', 'test'], \
                "Unknown V-COCO partition " + partition
            image_dir = dict(
                train='mscoco2014/train2014',
                val='mscoco2014/train2014',
                trainval='mscoco2014/train2014',
                test='mscoco2014/val2014'
            )
            self.dataset = VCOCO(
                root=os.path.join(data_root, image_dir[partition]),
                anno_file=os.path.join(data_root, 'instances_vcoco_{}.json'.format(partition)
                ), target_transform=pocket.ops.ToTensor(input_format='dict')
            )
            
            if partition == 'trainval':
                if args.eval or args.cache:
                    self.anno_bbox = None
                else:
                    self.anno_bbox = load_detector_pickle(name, f'{name}_train_bbox_R50')
            elif partition == 'test':
                # if 'swin_large' in args.pretrained:
                #     detr_backbone = 'SwinL'
                if args.custom_detector_results_path is not None:
                    anno_box_name = args.custom_detector_results_path
                else:
                    anno_box_name = f'{name}_test_bbox_{detr_backbone}_{args.pretrained.rsplit("/",1)[-1].rsplit(".",1)[0]}'
                self.anno_bbox = load_detector_pickle(name, anno_box_name)

        self.filtered_data = None
        # # Filter dataset based on max_num_duplicate_object
        # if self.max_num_duplicate_object is not None and 'test' in partition:
        #     self._filter_dataset()

        # add clip normalization 
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        normalize_clip = T.Compose([
            T.ToTensor(),
            T.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        ])
        normalize_clip_1 = T.ToTensor()
        normalize_clip_2 = T.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        if partition.startswith('train'):
            self.transforms = [T.Compose([
                T.RandomHorizontalFlip(),
                T.ColorJitter(.4, .4, .4),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=1333),
                    T.Compose([
                        T.RandomResize([400, 500, 600]),
                        T.RandomSizeCrop(384, 600),
                        T.RandomResize(scales, max_size=1333),
                    ]))
                    ]),
        normalize, normalize_clip,
        T.Compose([
                 T.IResize([self.clip_input_resolution,self.clip_input_resolution])
            ])
        ]
        else:   
            self.transforms = [T.Compose([
                T.RandomResize([800], max_size=1333),
            ]),
            normalize, normalize_clip,
            T.Compose([
                 T.IResize([self.clip_input_resolution,self.clip_input_resolution])
            ]),
            normalize_clip_1,
            normalize_clip_2
            ]

        self.partition = partition
        self.name = name
        self.count=0
        self.args = args

        # device = "cuda"
        if weak_model is not None:
            from main_weak import get_transforms_from_model
            self.process = get_transforms_from_model(weak_model, is_train=False, args=weak_model_args)
            # pass
        else:
            if self.args.use_open_clip:
                import open_clip
                _, _, self.process = open_clip.create_model_and_transforms(self.args.open_clip_model_name, pretrained=self.args.open_clip_pretrained)
            else:
                _, self.process = clip.load(self.clip_model_name, device='cpu')
            if self.args.no_zero_padding:
                self.process.transforms[0] = self.process.transforms[0].__class__((self.clip_input_resolution, self.clip_input_resolution), interpolation=BICUBIC)

        # self.process.transforms = self.process.transforms[:-1]
        self.human_index = None
        self.nms_region_proposals = None
        self.naive_region_proposals = None

    def _filter_dataset(self):
        """Filter dataset based on max_num_duplicate_object using anno_bbox"""
        filtered_data = []
        
        for i in range(len(self.dataset)):
            (image, target), filename = self.dataset[i]
            
            # Get anno_bbox information for this filename
            anno_bbox_info = self.anno_bbox[filename][0] if isinstance(self.anno_bbox[filename], list) else self.anno_bbox[filename]
            boxes = torch.as_tensor(anno_bbox_info['boxes'])
            scores = torch.as_tensor(anno_bbox_info['scores'])
            labels = torch.as_tensor(anno_bbox_info['labels'])
            
            max_duplicate_count = self._get_max_duplicate_count_from_anno(boxes, scores, labels)
            
            if max_duplicate_count == self.max_num_duplicate_object:
                filtered_data.append((self.dataset[i]))
        
        # Replace dataset with filtered data
        self.filtered_data = filtered_data
        print(f"Filtered dataset: {len(filtered_data)} samples for max_num_duplicate_object={self.max_num_duplicate_object} (score_threshold={self.score_threshold})")

    def _remove_duplicate_boxes(self, boxes_h, iou_threshold=0.95):
        """Remove duplicate boxes based on IoU threshold"""
        if len(boxes_h) == 0:
            return boxes_h
        
        unique_indices = []
        
        for i, box in enumerate(boxes_h):
            is_duplicate = False
            
            if unique_indices:
                # Calculate IoU between current box and all unique boxes
                unique_boxes = boxes_h[unique_indices]
                ious = box_iou(box.unsqueeze(0), unique_boxes)
                
                if ious.max() >= iou_threshold:
                    is_duplicate = True
            
            if not is_duplicate:
                unique_indices.append(i)
        
        return boxes_h[unique_indices] if unique_indices else torch.empty(0, 4)

    def _get_max_duplicate_count_from_anno(self, boxes, scores, labels):
        """Get the maximum duplicate count from anno_bbox data"""
        # Filter boxes by score threshold
        valid_mask = scores >= self.score_threshold
        if not valid_mask.any():
            return 0
            
        valid_labels = labels[valid_mask]
        
        # Count occurrences of each label
        unique_labels, label_counts = torch.unique(valid_labels, return_counts=True)
        
        # Return the maximum count among all labels
        return label_counts.max().item()

    def _get_max_duplicate_count(self, object_classes, boxes_h):
        """Get the maximum duplicate count for object classes and unique humans (legacy method)"""
        # Count occurrences of each object class
        unique_object_classes, object_counts = torch.unique(object_classes, return_counts=True)
        max_object_count = object_counts.max().item()
        
        # Count unique human boxes (remove duplicate boxes using IoU)
        unique_human_boxes = self._remove_duplicate_boxes(boxes_h)
        unique_human_count = len(unique_human_boxes)
        
        # Return the maximum count between objects and unique humans
        return max(max_object_count, unique_human_count)

    def get_contrastive_images(self, image, target):
        """Generate contrastive images based on specified methods"""
        crop_size_human, crop_size_object, crop_size, keep_boxes, keep_scores, keep_indices = self.get_region_proposals(target, image_h=image.size[1], image_w=image.size[0], return_meta=True)
        crop_size_human, crop_size_object, crop_size = crop_size_human.numpy(), crop_size_object.numpy(), crop_size.numpy()
        keep_boxes, keep_scores = keep_boxes.numpy(), keep_scores.numpy()
        
        contrastive_images = []
        # Generate first type of contrastive images
        first_images = self._generate_contrastive_type(image, self.args.contrastive_first, 
                                                        crop_size_human, crop_size_object, crop_size, keep_boxes, keep_scores, keep_indices)
        contrastive_images.append(first_images)
        
        # Generate second type of contrastive images
        second_images = self._generate_contrastive_type(image, self.args.contrastive_second,
                                                        crop_size_human, crop_size_object, crop_size, keep_boxes, keep_scores, keep_indices)
        contrastive_images.append(second_images)
        
        return torch.cat(contrastive_images, dim=0)
    
    def get_reweighting_images(self, image, target):
        """Generate reweighting images based on specified methods"""
        crop_size_human, crop_size_object, crop_size, keep_boxes, keep_scores, keep_indices = self.get_region_proposals(target, image_h=image.size[1], image_w=image.size[0], return_meta=True)
        crop_size_human, crop_size_object, crop_size = crop_size_human.numpy(), crop_size_object.numpy(), crop_size.numpy()
        keep_boxes, keep_scores = keep_boxes.numpy(), keep_scores.numpy()
        reweighting_images, masks = self._generate_contrastive_type(image, 'mask_background', 
                                                        crop_size_human, crop_size_object, crop_size, keep_boxes, keep_scores, keep_indices, return_masks=True)
        return reweighting_images, masks
        
    def _generate_contrastive_type(self, image, method, crop_size_human, crop_size_object, crop_size, keep_boxes, keep_scores, keep_indices, return_masks=False):
        """Generate specific type of contrastive images based on method"""
        # Generate union box images
        union_images = []
        masks = []
        for i, crop_s in enumerate(crop_size):
            
            new_img = image.crop(crop_s)
            if not self.args.no_zero_padding:
                new_img = self.expand2square(new_img, (0, 0, 0))
            union_images.append(new_img)
        
        if 'mask' in method:
            keep_indices_h, keep_indices_o = keep_indices
            # Generate union images with selective masking based on method name
            mask_sub = 'sub' in method
            mask_obj = 'obj' in method
            
            for i, (crop_s_h, crop_s_o, crop_s) in enumerate(zip(crop_size_human, crop_size_object, crop_size)):
                # Create union image
                union_img_array = np.array(union_images[i])
                
                # Calculate relative positions of sub and obj boxes within union box
                union_x1, union_y1, union_x2, union_y2 = crop_s
                
                if 'background' in method:
                    # Create mask if return_masks is True
                    if return_masks:
                        mask = np.full((union_img_array.shape[0], union_img_array.shape[1]), -1, dtype=np.float32)
                    
                    # mask out all the instances that is in the union box
                    # First soft mask out all other instances based on their scores
                    for j, (box, score) in enumerate(zip(keep_boxes, keep_scores)):
                        if j == keep_indices_h[i] or j == keep_indices_o[i]:
                            continue
                        
                        h_x1, h_y1, h_x2, h_y2 = box
                        rel_h_x1 = max(0, int(h_x1 - union_x1))
                        rel_h_y1 = max(0, int(h_y1 - union_y1))
                        rel_h_x2 = min(union_img_array.shape[1], int(h_x2 - union_x1))
                        rel_h_y2 = min(union_img_array.shape[0], int(h_y2 - union_y1))
                        
                        # Update mask for masked regions based on mask_type
                        if return_masks:
                            if self.args.mask_type == 'soft':
                                mask[rel_h_y1:rel_h_y2, rel_h_x1:rel_h_x2] = score
                            elif self.args.mask_type == 'hard':
                                mask[rel_h_y1:rel_h_y2, rel_h_x1:rel_h_x2] = 0
                        
                        # Masking based on type and score
                        if self.args.mask_type == 'soft':
                            mask_factor = 1.0 - score  # score 1 -> factor 0 (full mask), score 0 -> factor 1 (no mask)
                            union_img_array[rel_h_y1:rel_h_y2, rel_h_x1:rel_h_x2] = \
                                (union_img_array[rel_h_y1:rel_h_y2, rel_h_x1:rel_h_x2] * mask_factor).astype(union_img_array.dtype)
                        elif self.args.mask_type == 'hard':
                            union_img_array[rel_h_y1:rel_h_y2, rel_h_x1:rel_h_x2] = 0
                        elif self.args.mask_type == 'blur':
                            # Blur the region based on score: higher score = stronger blur
                            blur_radius = score * 10  # score 0 -> radius 0 (no blur), score 1 -> radius 10 (strong blur)
                            if blur_radius > 0:
                                # Extract the region to blur
                                region_to_blur = union_img_array[rel_h_y1:rel_h_y2, rel_h_x1:rel_h_x2]
                                if region_to_blur.shape[0] > 0 and region_to_blur.shape[1] > 0:
                                    # Convert to PIL Image for blurring
                                    region_pil = Image.fromarray(region_to_blur)
                                    # Apply Gaussian blur
                                    blurred_region = region_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
                                    # Convert back to numpy and replace the region
                                    union_img_array[rel_h_y1:rel_h_y2, rel_h_x1:rel_h_x2] = np.array(blurred_region)
                        else:
                            raise NotImplementedError(f"Mask type {self.args.mask_type} not implemented")
                    
                    # Then restore the subject and object regions from original image
                    original_union_img = image.crop(crop_s)
                    original_array = np.array(original_union_img)
                    
                    # Restore subject region
                    h_x1, h_y1, h_x2, h_y2 = crop_s_h
                    rel_h_x1 = max(0, int(h_x1 - union_x1))
                    rel_h_y1 = max(0, int(h_y1 - union_y1))
                    rel_h_x2 = min(union_img_array.shape[1], int(h_x2 - union_x1))
                    rel_h_y2 = min(union_img_array.shape[0], int(h_y2 - union_y1))
                    union_img_array[rel_h_y1:rel_h_y2, rel_h_x1:rel_h_x2] = original_array[rel_h_y1:rel_h_y2, rel_h_x1:rel_h_x2]
                    
                    # Set subject region to -1 in mask
                    if return_masks:
                        mask[rel_h_y1:rel_h_y2, rel_h_x1:rel_h_x2] = -1
                    
                    # Restore object region
                    o_x1, o_y1, o_x2, o_y2 = crop_s_o
                    rel_o_x1 = max(0, int(o_x1 - union_x1))
                    rel_o_y1 = max(0, int(o_y1 - union_y1))
                    rel_o_x2 = min(union_img_array.shape[1], int(o_x2 - union_x1))
                    rel_o_y2 = min(union_img_array.shape[0], int(o_y2 - union_y1))
                    union_img_array[rel_o_y1:rel_o_y2, rel_o_x1:rel_o_x2] = original_array[rel_o_y1:rel_o_y2, rel_o_x1:rel_o_x2]
                    
                    # Set object region to -1 in mask
                    if return_masks:
                        mask[rel_o_y1:rel_o_y2, rel_o_x1:rel_o_x2] = -1
                    
                    # Add mask to masks list
                    if return_masks:
                        if not self.args.no_zero_padding:
                            # Apply same square padding as expand2square with 1 values
                            h, w = mask.shape
                            if h == w:
                                pass  # Already square
                            elif w > h:
                                # Pad height (top and bottom)
                                padded_mask = np.full((w, w), 1, dtype=np.float32)
                                y_offset = (w - h) // 2
                                padded_mask[y_offset:y_offset+h, :] = mask
                                mask = padded_mask
                            else:  # h > w
                                # Pad width (left and right)
                                padded_mask = np.full((h, h), 1, dtype=np.float32)
                                x_offset = (h - w) // 2
                                padded_mask[:, x_offset:x_offset+w] = mask
                                mask = padded_mask
                        masks.append(mask)
                    
                # Mask human box if 'sub' is in method
                if mask_sub:
                    h_x1, h_y1, h_x2, h_y2 = crop_s_h
                    rel_h_x1 = max(0, int(h_x1 - union_x1))
                    rel_h_y1 = max(0, int(h_y1 - union_y1))
                    rel_h_x2 = min(union_img_array.shape[1], int(h_x2 - union_x1))
                    rel_h_y2 = min(union_img_array.shape[0], int(h_y2 - union_y1))
                    
                    # maskout for human region based on subject score
                    if self.args.mask_type == 'blur':
                        # Blur the human region based on subject score
                        subject_score = keep_scores[keep_indices_h[i]]
                        blur_radius = subject_score * 10  # score 0 -> radius 0 (no blur), score 1 -> radius 10 (strong blur)
                        if blur_radius > 0:
                            # Extract the region to blur
                            region_to_blur = union_img_array[rel_h_y1:rel_h_y2, rel_h_x1:rel_h_x2]
                            if region_to_blur.shape[0] > 0 and region_to_blur.shape[1] > 0:
                                # Convert to PIL Image for blurring
                                region_pil = Image.fromarray(region_to_blur)
                                # Apply Gaussian blur
                                blurred_region = region_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
                                # Convert back to numpy and replace the region
                                union_img_array[rel_h_y1:rel_h_y2, rel_h_x1:rel_h_x2] = np.array(blurred_region)
                    else:
                        union_img_array[rel_h_y1:rel_h_y2, rel_h_x1:rel_h_x2] = 0
                
                # Mask object box if 'obj' is in method
                if mask_obj:
                    o_x1, o_y1, o_x2, o_y2 = crop_s_o
                    rel_o_x1 = max(0, int(o_x1 - union_x1))
                    rel_o_y1 = max(0, int(o_y1 - union_y1))
                    rel_o_x2 = min(union_img_array.shape[1], int(o_x2 - union_x1))
                    rel_o_y2 = min(union_img_array.shape[0], int(o_y2 - union_y1))
                    
                    # maskout for object region based on object score
                    if self.args.mask_type == 'blur':
                        # Blur the object region based on object score
                        object_score = keep_scores[keep_indices_o[i]]
                        blur_radius = object_score * 10  # score 0 -> radius 0 (no blur), score 1 -> radius 10 (strong blur)
                        if blur_radius > 0:
                            # Extract the region to blur
                            region_to_blur = union_img_array[rel_o_y1:rel_o_y2, rel_o_x1:rel_o_x2]
                            if region_to_blur.shape[0] > 0 and region_to_blur.shape[1] > 0:
                                # Convert to PIL Image for blurring
                                region_pil = Image.fromarray(region_to_blur)
                                # Apply Gaussian blur
                                blurred_region = region_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
                                # Convert back to numpy and replace the region
                                union_img_array[rel_o_y1:rel_o_y2, rel_o_x1:rel_o_x2] = np.array(blurred_region)
                    else:
                        union_img_array[rel_o_y1:rel_o_y2, rel_o_x1:rel_o_x2] = 0
                
                # Convert back to PIL and process
                masked_img = Image.fromarray(union_img_array)
                if not self.args.no_zero_padding:
                    masked_img = self.expand2square(masked_img, (0, 0, 0))
                
                union_images[i] = masked_img
            
            if self.args.vis_images:
                for i in range(len(union_images)):
                    save_dir = getattr(self, 'masked_images_dir', './masked_images')
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    
                    # Create a unique filename
                    save_filename = f"masked_{method}_{i}_{len(union_images)}.jpg"
                    save_path = os.path.join(save_dir, save_filename)
                    union_images[i].save(save_path)
                self.args.vis_images
        processed_images = torch.stack([self.process(img) for img in union_images])
        
        if return_masks and masks:
            return processed_images, torch.tensor(np.stack(masks), dtype=torch.float32)
        else:
            return processed_images

    def __len__(self):
        if self.filtered_data is None:
            return len(self.dataset)
        else:
            return len(self.filtered_data)

    ##  padding zeros
    def __getitem__(self, i):
        # Use filtered data if filtering is applied
        if self.filtered_data is not None:
            (image, target), filename = self.filtered_data[i]
        else:
            (image, target), filename = self.dataset[i]
        w,h = image.size
        target['orig_size'] = torch.tensor([h,w])
        target['filename'] = filename
        if self.args.online_detection:
            if self.name == 'hicodet':
                target['labels'] = target['verb']
                # Convert ground truth boxes to zero-based index and the
                # representation from pixel indices to coordinates
                target['boxes_h'][:, :2] -= 1
                target['boxes_o'][:, :2] -= 1
            else:
                target['labels'] = target['actions']
                target['object'] = target.pop('objects')
            # pdb.set_trace()
            image_0, target_0 = self.transforms[0](image, target)
            image, _ = self.transforms[1](image_0, None)
            image_clip, target = self.transforms[3](image_0, target_0)
            image_clip, target = self.transforms[2](image_clip, target)
            if image_0.size[-1] >1344 or image_0.size[-2] >1344:print(image_0.size)
            
            return (image,image_clip), target
        else:

            anno_bbox_list = self.anno_bbox[filename][0] if isinstance(self.anno_bbox[filename], list) else self.anno_bbox[filename]
            target['ex_bbox'] = torch.as_tensor(anno_bbox_list['boxes'])
            target['ex_scores'] = torch.as_tensor(anno_bbox_list['scores'])
            target['ex_labels'] = torch.as_tensor(anno_bbox_list['labels'])
            target['ex_hidden_states'] = torch.as_tensor(anno_bbox_list['hidden_states']) if 'hidden_states' in anno_bbox_list else None
            # pdb.set_trace()
            if self.name == 'hicodet':
                target['labels'] = target['verb'] 
                # Convert ground truth boxes to zero-based index and the
                # representation from pixel indices to coordinates
                target['boxes_h'][:, :2] -= 1 ## why not [:,:4] -= 1?
                target['boxes_o'][:, :2] -= 1
            else:
                target['labels'] = target['actions']
                target['object'] = target.pop('objects')

            if self._load_features:
                raise NotImplementedError
                all_images = torch.as_tensor(self._text_features[filename])

            elif self.args.trainin_free_option == 'contrastive' and not self.args.use_attention_reweighting:
                all_images = self.get_contrastive_images(image, target)
            # elif self.args.use_attention_reweighting:
            #     all_images, all_masks = self.get_reweighting_images(image, target)
            else:
                crop_size_human, crop_size_object, crop_size, keep_boxes, keep_scores, keep_labels, (x_keep, y_keep) = self.get_region_proposals(target,image_h=image.size[1], image_w=image.size[0], return_meta=True)
                assert len(crop_size_human) == len(crop_size_object) == len(crop_size) == len(x_keep) == len(y_keep)
                crop_size_human, crop_size_object, crop_size = crop_size_human.numpy(), crop_size_object.numpy(), crop_size.numpy()
                if self.args.input_type == 'full':
                    if not self.args.no_zero_padding:
                        new_img = self.expand2square(image,(0,0,0))
                    else:
                        new_img = image
                    all_images = self.process(new_img)
                else:
                    all_images = []
                    all_objects = []
                    all_human = []
                    for crop_s, crop_s_o, crop_s_h in zip(crop_size,crop_size_object,crop_size_human):
                        
                        new_img = image.crop(crop_s)
                        if not self.args.no_zero_padding:
                            new_img = self.expand2square(new_img,(0,0,0)) #
                        all_images.append(self.process(new_img))
                        if not self.args.use_attention_reweighting:
                            new_img = image.crop(crop_s_o)
                            if not self.args.no_zero_padding:
                                new_img = self.expand2square(new_img,(0,0,0)) #
                            all_objects.append(self.process(new_img))
                            new_img = image.crop(crop_s_h)
                            if not self.args.no_zero_padding:
                                new_img = self.expand2square(new_img,(0,0,0)) #
                            # Save the human crop image
                            # new_img.save(f"human_crop_{len(all_human)}.jpg")
                            all_human.append(self.process(new_img))

                    if self.args.use_attention_reweighting:
                        all_images = torch.stack(all_images)
                    else:
                        # if len(all_images) == 0:
                        #     all_images = torch.zeros(0,3,self.clip_input_resolution,self.clip_input_resolution)
                        # else:
                        all_images = torch.stack(all_images)
                        all_images_object = torch.stack(all_objects)
                        all_images_human = torch.stack(all_human)
                        all_images = torch.cat([all_images_human,all_images_object,all_images],dim=0)
            
            image_0, target_0 = self.transforms[3](image, target)
            image_clip, target = self.transforms[2](image_0, target_0)
            if image_0.size[-1] >self.clip_input_resolution or image_0.size[-2] >self.clip_input_resolution:
                print(image_0.size)

            mask = torch.zeros((len(target['ex_bbox']), 224, 224), dtype=torch.bool)
            for i in range(len(target['ex_bbox'])):
                t = target['ex_bbox'][i].clamp(0,224).int()
                mask[i, t[1]:t[3], t[0]:t[2]] = 1
            # pdb.set_trace()
            assert mask.shape[0] != 0
            mask = F.interpolate(mask[None].float(), size=(7,7)).to(dtype=torch.bool)[0]
            target['ex_mask'] = mask
            if self.args.use_attention_reweighting:
                # target['feat_mask'] = all_masks
                target['keep_boxes'] = keep_boxes.numpy()
                target['keep_scores'] = keep_scores.numpy()
                target['keep_labels'] = keep_labels.numpy()
                target['keep_human_idx'] = x_keep.numpy()
                target['keep_object_idx'] = y_keep.numpy()
                target['crop_human'] = crop_size_human
                target['crop_object'] = crop_size_object
                target['crop_union'] = crop_size
            return all_images, target
    

    def expand2square(self, pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result
            
    def get_region_proposals(self, results,image_h, image_w, return_meta=False):
        if self.human_index is not None:
            human_idx = self.human_index
            # device = torch.device('cpu')
            if self.args.use_nms:
                region_props = self.nms_region_proposals([results])[0]
                boxes = region_props['boxes']
                scores = region_props['scores']
                labels = region_props['labels']
            else:
                region_props = self.naive_region_proposals([results])[0]
                boxes = region_props['boxes']
                scores = region_props['scores']
                labels = region_props['labels']
        else:
            human_idx = 0
            min_instances = 3
            max_instances = 15
            bx = results['ex_bbox']
            sc = results['ex_scores']
            lb = results['ex_labels'] ## object-category labels(0~80)
            
            if self.args.use_nms:
                keep = batched_nms(bx, sc, lb, 0.5)
                sc = sc[keep]
                lb = lb[keep]
                bx = bx[keep]
                keep = torch.nonzero(sc >= self.args.box_score_thresh).squeeze(1)
                
            else:
                keep = torch.arange(len(bx))
            
            is_human = lb == human_idx
            hum = torch.nonzero(is_human).squeeze(1)
            obj = torch.nonzero(is_human == 0).squeeze(1)
            n_human = is_human[keep].sum(); n_object = len(keep) - n_human
            # Keep the number of human and object instances in a specified interval
            # device = torch.device('cpu')
            if n_human < min_instances:
                keep_h = sc[hum].argsort(descending=True)[:min_instances]
                keep_h = hum[keep_h]
            elif n_human > max_instances:
                keep_h = sc[hum].argsort(descending=True)[:max_instances]
                keep_h = hum[keep_h]
            else:
                keep_h = torch.nonzero(is_human[keep]).squeeze(1)
                keep_h = keep[keep_h]
                # keep_h = hum

            if n_object < min_instances:
                keep_o = sc[obj].argsort(descending=True)[:min_instances]
                keep_o = obj[keep_o]
            elif n_object > max_instances:
                keep_o = sc[obj].argsort(descending=True)[:max_instances]
                keep_o = obj[keep_o]
            else:
                # keep_o = obj
                keep_o = torch.nonzero(is_human[keep] == 0).squeeze(1)
                keep_o = keep[keep_o]
            keep = torch.cat([keep_h, keep_o])
            if len(keep_h)==0:
                null_box = torch.zeros(1,4, dtype=bx.dtype)
                null_score = torch.zeros(1, dtype=sc.dtype)
                null_label = torch.zeros(1, dtype=lb.dtype)
                # null_hidden_states = torch.zeros(1,512) if hs is not None else None
                # null_mask = torch.zeros(1) if ms is not None else None
                
            boxes=bx[keep] if len(keep_h)>0 else torch.cat([null_box, bx[keep]], dim=0)
            scores=sc[keep] if len(keep_h)>0 else torch.cat([null_score, sc[keep]], dim=0)
            labels=lb[keep] if len(keep_h)>0 else torch.cat([null_label, lb[keep]], dim=0)
            # hidden_states=hs[keep] if hs is not None else None
            
        is_human = labels == human_idx
            
        n_h = torch.sum(is_human); n = len(boxes)
        # Permute human instances to the top
        if not torch.all(labels[:n_h]==human_idx):
            h_idx = torch.nonzero(is_human).squeeze(1)
            o_idx = torch.nonzero(is_human == 0).squeeze(1)
            perm = torch.cat([h_idx, o_idx])
            boxes = boxes[perm]; scores = scores[perm]
            labels = labels[perm]; unary_tokens = unary_tokens[perm]
        # Skip image when there are no valid human-object pairs
        if n_h == 0 or n <= 1:
            print(n_h, n)

        # Get the pairwise indices
        x, y = torch.meshgrid(
            torch.arange(n),
            torch.arange(n)
        )
        # Valid human-object pairs
        x_keep, y_keep = torch.nonzero(torch.logical_and(x != y, x < n_h)).unbind(1)
        
        # boxes[:,0::2].clamp_(0, image_w)
        # boxes[:,1::2].clamp_(0, image_h)
        sub_boxes = boxes[x_keep]
        obj_boxes = boxes[y_keep]
        lt = torch.min(sub_boxes[..., :2], obj_boxes[..., :2]) # left point
        rb = torch.max(sub_boxes[..., 2:], obj_boxes[..., 2:]) # right point
        union_boxes = torch.cat([lt,rb],dim=-1)
        sub_boxes[:,0].clamp_(0, image_w)
        sub_boxes[:,1].clamp_(0, image_h)
        sub_boxes[:,2].clamp_(0, image_w)
        sub_boxes[:,3].clamp_(0, image_h)

        obj_boxes[:,0].clamp_(0, image_w)
        obj_boxes[:,1].clamp_(0, image_h)
        obj_boxes[:,2].clamp_(0, image_w)
        obj_boxes[:,3].clamp_(0, image_h)

        union_boxes[:,0].clamp_(0, image_w)
        union_boxes[:,1].clamp_(0, image_h)
        union_boxes[:,2].clamp_(0, image_w)
        union_boxes[:,3].clamp_(0, image_h)

        if return_meta:
            keep_indices = torch.cat([x_keep, y_keep]).unique()
            keep_boxes = boxes[keep_indices]
            keep_scores = scores[keep_indices]
            keep_labels = labels[keep_indices]
            return sub_boxes, obj_boxes, union_boxes, keep_boxes, keep_scores, keep_labels,(x_keep, y_keep)
        else:
            return sub_boxes, obj_boxes, union_boxes


class CacheTemplate(defaultdict):
    """A template for VCOCO cached results """
    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            self[k] = v
    def __missing__(self, k):
        seg = k.split('_')
        # Assign zero score to missing actions
        if seg[-1] == 'agent':
            return 0.
        # Assign zero score and a tiny box to missing <action,role> pairs
        else:
            return [0., 0., .1, .1, 0.]

class CustomisedDLE(DistributedLearningEngine):
    def __init__(self, net, dataloader, max_norm=0, num_classes=117, args=None, **kwargs):
        super().__init__(net, None, dataloader, **kwargs)
        self.max_norm = max_norm
        self.num_classes = num_classes
        self.args = args
        # self.cache_dir = kwargs['cache_dir']
    def _on_each_iteration(self):
        loss_dict = self._state.net(
            *self._state.inputs, targets=self._state.targets)
        if loss_dict['interaction_loss'].isnan():
            raise ValueError(f"The HOI loss is NaN for rank {self._rank}")

        self._state.loss = sum(loss for loss in loss_dict.values())
        self._state.optimizer.zero_grad(set_to_none=True)
        self._state.loss.backward()
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self._state.net.parameters(), self.max_norm)
        self._state.optimizer.step()

    def get_distance(self, boxes_h, boxes_o, image_size):
        h,w = image_size
        norm_boxes_h = boxes_h / torch.tensor([w,h,w,h])
        norm_boxes_o = boxes_o / torch.tensor([w,h,w,h])
        center_boxes_h = (norm_boxes_h[:,:2]+norm_boxes_h[:,2:])/2
        center_boxes_o = (norm_boxes_o[:,:2]+norm_boxes_o[:,2:])/2
        distance = (center_boxes_h-center_boxes_o).pow(2).sum(dim=1).sqrt()
        
        # distance_label = min(int(distance/self.args.distance_interval), int(self.args.max_distance / self.args.distance_interval))
        distance_label = torch.clamp(distance//self.args.distance_interval, max=int(self.args.max_distance / self.args.distance_interval))
        return distance_label

    @torch.no_grad()
    def test_hico(self, dataloader, args):
        net = self._state.net
        net.eval()
        
        if self.args.save_results_as_dict and not self.args.eval_with_saved_results:
            all_predictions = []
            all_targets = []
        
        if self.args.eval_with_saved_results:
            with open(os.path.join(self.args.output_dir, 'results.pkl'), 'rb') as f:
                results_dict = pickle.load(f)
            output_dict = {}
            for res in results_dict['predictions']:
                output_dict[res['filename']] = res
        
        dataset = dataloader.dataset.dataset
        interaction_to_verb = torch.as_tensor(dataset.interaction_to_verb)
        associate = BoxPairAssociation(min_iou=0.5, return_gt_det_idx=True)
        conversion = torch.from_numpy(np.asarray(
            dataset.object_n_verb_to_interaction, dtype=float
        ))

        tgt_num_classes = 600
        num_gt = dataset.anno_interaction
        meter = DetectionAPMeter(
            tgt_num_classes, nproc=1,
            num_gt=num_gt,
            algorithm='11P'
        )
        
        if self.args.eval_distance:
            tgt_distance = dataset.distance_statistics
            if self.args.class_wise_ap:
                distance_meter = [DetectionAPMeter(
                    len(tgt_distance[i]), nproc=1,
                        num_gt=tgt_distance[i],
                        algorithm='11P'
                    ) for i in range(len(tgt_distance))]
            else:
                distance_meter = DetectionAPMeter(
                    len(tgt_distance), nproc=1,
                    num_gt=tgt_distance,
                    algorithm='11P'
                )
        else:
            tgt_distance = None
            distance_meter = None
        
        all_preds = []
        distance_pred_list = [[] for _ in range(len(tgt_distance))] if self.args.eval_distance and self.args.class_wise_ap else []
        
        for batch in tqdm(dataloader):
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            outputs = net(inputs,batch[1])

            if self.args.visualize_results:
                vis_dir = os.path.join(self.args.output_dir, 'visualization')
                if not os.path.exists(vis_dir):
                    os.makedirs(vis_dir)
                for output, target in zip(outputs, batch[1]):
                    target_image_path = dataloader.dataset.dataset._root + '/' + target['filename']
                    target_image = Image.open(target_image_path)
                    vis_output = copy.deepcopy(output)
                    vis_output = pocket.ops.relocate_to_cpu(vis_output, ignore=True)
                    if net.module.num_classes == 600:
                        vis_output['labels'] = torch.as_tensor(dataset.interaction_to_verb)[vis_output['labels']]
                    
                    visualise_entire_image(target_image, vis_output, dataloader.dataset.dataset.verbs,
                                           thresh=0.05, box_thresh=0.2, save_filename=os.path.join(self.args.output_dir, 'visualization', target['filename']))
            

            # continue
            # Skip images without detections
            if outputs is None or len(outputs) == 0:
                continue
            # # Batch size is fixed as 1 for inference
            # assert len(output) == 1, f"Batch size is not 1 but {len(outputs)}."
            for output, target in zip(outputs, batch[-1]):
                output = pocket.ops.relocate_to_cpu(output, ignore=True)
                # pdb.set_trace()
                # Format detections
                boxes = output['boxes']
                boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
                objects = output['objects']
                scores = output['scores']
                verbs = output['labels']
                if net.module.class_nums==117:
                    interactions = conversion[objects, verbs]
                else:
                    interactions = verbs
                # Recover target box scale
                gt_bx_h = net.module.recover_boxes(target['boxes_h'], target['size'])
                gt_bx_o = net.module.recover_boxes(target['boxes_o'], target['size'])
                # pdb.set_trace()
                # Associate detected pairs with ground truth pairs
                labels = torch.zeros_like(scores)
                unique_hoi = interactions.unique()
                
                gt_matched_indices = torch.full_like(target['hoi'], -1)
                if self.args.eval_distance:
                    distances = self.get_distance(boxes_h, boxes_o, output['size'])
                    gt_distances = self.get_distance(gt_bx_h, gt_bx_o, output['size'])
                for hoi_idx in unique_hoi:
                    gt_idx = torch.nonzero(target['hoi'] == hoi_idx).squeeze(1)
                    det_idx = torch.nonzero(interactions == hoi_idx).squeeze(1)
                    if len(gt_idx):
                        labels[det_idx], gt_det_idx = associate(
                            (gt_bx_h[gt_idx].view(-1, 4),
                            gt_bx_o[gt_idx].view(-1, 4)),
                            (boxes_h[det_idx].view(-1, 4),
                            boxes_o[det_idx].view(-1, 4)),
                            scores[det_idx].view(-1)
                        )
                        gt_matched_indices[gt_idx] = det_idx[gt_det_idx]
                        
                        if self.args.eval_distance:
                            distances[det_idx[gt_det_idx]] = gt_distances[gt_idx] # replace matched distance label with gt distance label
                # meter.append(scores, interactions, labels)   # scores human*object*verb, interaction（600), labels
                all_preds.append((scores, interactions, labels))
                
                # for distance
                if self.args.eval_distance:
                    if self.args.class_wise_ap:
                        for i_ in range(len(tgt_distance)):
                            distance_pred_list[i_].append((scores[distances==i_], interactions[distances==i_], labels[distances==i_]))
                    else:
                        distance_pred_list.append((scores, distances, labels))
                
                if self.args.save_results_as_dict and not self.args.eval_with_saved_results:
                    res_out = {}
                    res_out.update(output)
                    res_out['filename'] = target['filename']
                    res_out['gt_matched_indices'] = gt_matched_indices
                    # res_out['data_root'] = target['data_root']
                    all_predictions.append(res_out)
                    target_copy = copy.deepcopy(target)
                    if 'feat_mask' in target_copy:
                        del target_copy['feat_mask']
                    all_targets.append(target_copy)
            # break
        if self.args.save_results_as_dict and not self.args.eval_with_saved_results:
            print(f"Saving results to {os.path.join(self.args.output_dir, 'results.pkl')}")
            gathered_predictions = []
            gathered_targets = []
            for gathered_pred in ddp.all_gather(all_predictions):
                gathered_predictions.extend(gathered_pred)
            for gathered_target in ddp.all_gather(all_targets):
                gathered_targets.extend(gathered_target)
            if self._rank == 0: 
                with open(os.path.join(self.args.output_dir, 'results.pkl'), 'wb') as f:
                    pickle.dump({'predictions':gathered_predictions, 'targets':gathered_targets}, f)
                print(f"Saved results to {os.path.join(self.args.output_dir, 'results.pkl')}")
            del gathered_predictions, gathered_targets
            
        gathered_pred_list = []
        for preds in ddp.all_gather(all_preds):
            gathered_pred_list.extend(preds)
        for pred in gathered_pred_list:
            meter.append(*pred)
        # if self._rank == 0:
        meter_dict = {
            # 'class_meter': meter.eval(),
            'class_meter': meter,
        }
        
        if self.args.eval_distance:
            if self.args.class_wise_ap:
                for i_ in range(len(tgt_distance)):
                    gathered_distance_pred_list = []
                    for preds in ddp.all_gather(distance_pred_list[i_]):
                        gathered_distance_pred_list.extend(preds)
                    for pred in gathered_distance_pred_list:
                        distance_meter[i_].append(*pred)
            else:
                gathered_distance_pred_list = []
                for preds in ddp.all_gather(distance_pred_list):
                    gathered_distance_pred_list.extend(preds)
                for pred in gathered_distance_pred_list:
                    distance_meter.append(*pred)
            # meter_dict['distance_meter'] = distance_meter.eval() if not self.args.class_wise_ap else [meter.eval().mean() for meter in distance_meter]
            meter_dict['distance_meter'] = distance_meter if not self.args.class_wise_ap else [meter for meter in distance_meter]
        return meter_dict

    @torch.no_grad()
    def cache_hico(self, dataloader, cache_dir='matlab'):
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset
        conversion = torch.from_numpy(np.asarray(
            dataset.object_n_verb_to_interaction, dtype=float
        ))
        object2int = dataset.object_to_interaction

        # Include empty images when counting
        nimages = len(dataset.annotations)
        all_results = np.empty((600, nimages), dtype=object)

        for i, batch in enumerate(tqdm(dataloader)):
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            output = net(inputs, batch[1])

            # Skip images without detections
            if output is None or len(output) == 0:
                continue
            # Batch size is fixed as 1 for inference
            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = pocket.ops.relocate_to_cpu(output[0], ignore=True)
            # NOTE Index i is the intra-index amongst images excluding those
            # without ground truth box pairs
            image_idx = dataset._idx[i]
            # Format detections
            boxes = output['boxes']
            boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
            objects = output['objects']
            scores = output['scores']
            interactions = output['labels']
            # pdb.set_trace()
            # interactions = conversion[objects, verbs]
            # Rescale the boxes to original image size
            ow, oh = dataset.image_size(i)
            h, w = output['size']
            scale_fct = torch.as_tensor([
                ow / w, oh / h, ow / w, oh / h
            ]).unsqueeze(0)
            boxes_h *= scale_fct
            boxes_o *= scale_fct

            # Convert box representation to pixel indices
            boxes_h[:, 2:] -= 1
            boxes_o[:, 2:] -= 1

            # Group box pairs with the same predicted class
            permutation = interactions.argsort()
            boxes_h = boxes_h[permutation]
            boxes_o = boxes_o[permutation]
            interactions = interactions[permutation]
            scores = scores[permutation]

            # Store results
            unique_class, counts = interactions.unique(return_counts=True)
            n = 0
            for cls_id, cls_num in zip(unique_class, counts):
                all_results[cls_id.long(), image_idx] = torch.cat([
                    boxes_h[n: n + cls_num],
                    boxes_o[n: n + cls_num],
                    scores[n: n + cls_num, None]
                ], dim=1).numpy()
                n += cls_num
        
        # Replace None with size (0,0) arrays
        for i in range(600):
            for j in range(nimages):
                if all_results[i, j] is None:
                    all_results[i, j] = np.zeros((0, 0))
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        # Cache results
        for object_idx in range(80):
            interaction_idx = object2int[object_idx]
            sio.savemat(
                os.path.join(cache_dir, f'detections_{(object_idx + 1):02d}.mat'),
                dict(all_boxes=all_results[interaction_idx])
            )
            # pdb.set_trace() 
            pickle.dump(dict(all_boxes=all_results[interaction_idx]), 
                        open(os.path.join(cache_dir, f'detections_{(object_idx + 1):02d}.p'), 'wb')
                        )

    @torch.no_grad()
    def cache_vcoco(self, dataloader, cache_dir='vcoco_cache'):
        net = self._state.net
        net.eval()
        dataset = dataloader.dataset.dataset
        all_results = []
        for i, batch in enumerate(tqdm(dataloader)):
            
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            output = net(inputs, batch[1])

            # Skip images without detections
            if output is None or len(output) == 0:
                continue
            # Batch size is fixed as 1 for inference
            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = pocket.ops.relocate_to_cpu(output[0], ignore=True)
            # NOTE Index i is the intra-index amongst images excluding those
            # without ground truth box pairs
            image_id = dataset.image_id(i)
            # Format detections
            boxes = output['boxes']
            boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
            scores = output['scores']
            if net.module.num_classes == 24:
                actions = output['labels']
            elif net.module.num_classes == 236:
                interactions = output['labels']
                actions = torch.as_tensor(dataset.interaction_to_verb)[interactions]
            # Rescale the boxes to original image size
            ow, oh = dataset.image_size(i)
            h, w = output['size']
            scale_fct = torch.as_tensor([
                ow / w, oh / h, ow / w, oh / h
            ]).unsqueeze(0)
            boxes_h *= scale_fct
            boxes_o *= scale_fct

            for bh, bo, s, a in zip(boxes_h, boxes_o, scores, actions):
                a_name = dataset.actions[a].split()
                result = CacheTemplate(image_id=image_id, person_box=bh.tolist())
                result[a_name[0] + '_agent'] = s.item()
                result['_'.join(a_name)] = bo.tolist() + [s.item()]
                all_results.append(result)
        
        # gather all results
        gathered_all_results = []
        for gathered_result in ddp.all_gather(all_results):
            gathered_all_results.extend(gathered_result)

        if self._rank == 0:
            print(f"Total number of results: {len(gathered_all_results)}")
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            print(f'save cache.pkl in {self.args.output_dir}')
            with open(os.path.join(self.args.output_dir, 'cache.pkl'), 'wb') as f:
                # Use protocol 2 for compatibility with Python2
                pickle.dump(gathered_all_results, f, 2)


if __name__ == '__main__':
    meter = DetectionAPMeter(
            60, #nproc=1,
            # num_gt=dataset.anno_interaction,
            algorithm='11P'
        )
    scores = torch.rand(10000)
    pred = torch.randint(0, 60, (10000,))
    trueorfalse = torch.randint(0, 2, (10000,))
    meter.append(scores, pred, trueorfalse)
    ap = meter.eval()
    mAP = ap.mean()
    print(mAP) ## 0.5537

    meter.reset()
    ## 加上一些 false positive 和 false negative 
    ## (detr bbox和 gt bbox相差大的那部分一定是false positive或者false negative)
    scores = torch.cat([scores, torch.ones(5000) * 0.01], dim=0)
    pred = torch.cat([pred, torch.randint(0, 60, (5000,))], dim=0)
    trueorfalse = torch.cat([trueorfalse, torch.zeros(5000)], dim=0)
    meter.append(scores, pred, trueorfalse)
    ap = meter.eval()
    mAP = ap.mean()
    print(mAP) ## 0.3817

    
