"""
Visualise detected human-object interactions in an image

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import torch
import pocket
import warnings
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as peff

from mpl_toolkits.axes_grid1 import make_axes_locatable

# from utils import DataFactory
# from utils_tip_cache_and_union_finetune import custom_collate, CustomisedDLE, DataFactory

# from upt import build_detector
# from upt_tip_cache_model_free_finetune_distill3 import build_detector
import pdb
import random
from pocket.ops import relocate_to_cpu, relocate_to_cuda
warnings.filterwarnings("ignore")

OBJECTS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

def draw_boxes(ax, boxes, scores=None, box_thresh=0.0):
    xy = boxes[:, :2].unbind(0)
    h, w = (boxes[:, 2:] - boxes[:, :2]).unbind(1)
    for i, (a, b, c) in enumerate(zip(xy, h.tolist(), w.tolist())):
        if scores is not None:
            if scores[i] < box_thresh:
                continue
        patch = patches.Rectangle(a.tolist(), b, c, facecolor='none', edgecolor='w')
        ax.add_patch(patch)
        txt = plt.text(*a.tolist(), str(i+1), fontsize=20, fontweight='semibold', color='w')
        txt.set_path_effects([peff.withStroke(linewidth=5, foreground='#000000')])
        plt.draw()

def visualise_entire_image(image, output, actions, action=None, thresh=0.2, box_thresh=0.2,topk=10, save_filename=None, failure=False):
    """Visualise bounding box pairs in the whole image by classes"""
    # Rescale the boxes to original image size
    ow, oh = image.size
    h, w = output['size']
    scale_fct = torch.as_tensor([
        ow / w, oh / h, ow / w, oh / h
    ]).unsqueeze(0)
    boxes = output['boxes'].cpu() * scale_fct
    # Find the number of human and object instances
    nh = len(output['pairing'][0].unique()); no = len(boxes)

    scores = output['scores'].cpu()
    objects = output['objects'].cpu()
    pred = output['labels'].cpu()
    
    # Sort by scores in descending order
    sorted_indices = torch.argsort(scores, descending=True)
    scores = scores[sorted_indices]
    objects = objects[sorted_indices]
    pred = pred[sorted_indices]
    
    # Reorder pairing based on sorted indices
    pairing = output['pairing'][:, sorted_indices]
    
    # Visualise detected human-object pairs with attached scores
    # pdb.set_trace()
    unique_actions = torch.unique(pred)
    
    if action is not None:
        plt.cla()
        if failure:
            keep = torch.nonzero(torch.logical_and(scores < thresh, pred == action)).squeeze(1)
        else:
            keep = torch.nonzero(torch.logical_and(scores >= thresh, pred == action)).squeeze(1)
        bx_h, bx_o = boxes[output['pairing']].unbind(0)
        pocket.utils.draw_box_pairs(image, bx_h[keep], bx_o[keep], width=5)
        plt.imshow(image)
        plt.axis('off')
        # pdb.set_trace()
        # if len(keep) == 0: return 
        for i in range(len(keep)):
            txt = plt.text(*bx_h[keep[i], :2], f"{scores[keep[i]]:.2f}", fontsize=15, fontweight='semibold', color='w')
            txt.set_path_effects([peff.withStroke(linewidth=5, foreground='#000000')])
            plt.draw()
        # plt.show()
        plt.savefig(save_filename, bbox_inches='tight', pad_inches=0.0)
        # plt.savefig(save_filename)
        plt.cla()
        return

    # pairing = output['pairing']
    # coop_attn = output['attn_maps'][0]
    # comp_attn = output['attn_maps'][1]

    # Visualise attention from the cooperative layer
    # for i, attn_1 in enumerate(coop_attn):
    #     fig, axe = plt.subplots(2, 4)
    #     fig.suptitle(f"Attention in coop. layer {i}")
    #     axe = np.concatenate(axe)
    #     ticks = list(range(attn_1[0].shape[0]))
    #     labels = [v + 1 for v in ticks]
    #     for ax, attn in zip(axe, attn_1):
    #         im = ax.imshow(attn.squeeze().T, vmin=0, vmax=1)
    #         divider = make_axes_locatable(ax)
    #         ax.set_xticks(ticks)
    #         ax.set_xticklabels(labels)
    #         ax.set_yticks(ticks)
    #         ax.set_yticklabels(labels)
    #         cax = divider.append_axes('right', size='5%', pad=0.05)
    #         fig.colorbar(im, cax=cax)

    # x, y = torch.meshgrid(torch.arange(nh), torch.arange(no))
    # x, y = torch.nonzero(x != y).unbind(1)
    # pairs = [str((i.item() + 1, j.item() + 1)) for i, j in zip(x, y)]

    # Visualise attention from the competitive layer
    # fig, axe = plt.subplots(2, 4)
    # fig.suptitle("Attention in comp. layer")
    # axe = np.concatenate(axe)
    # ticks = list(range(len(pairs)))
    # for ax, attn in zip(axe, comp_attn):
    #     im = ax.imshow(attn, vmin=0, vmax=1)
    #     divider = make_axes_locatable(ax)
    #     ax.set_xticks(ticks)
    #     ax.set_xticklabels(pairs, rotation=45)
    #     ax.set_yticks(ticks)
    #     ax.set_yticklabels(pairs)
    #     cax = divider.append_axes('right', size='5%', pad=0.05)
    #     fig.colorbar(im, cax=cax)

    # Filter actions above threshold and create text for display
    threshold = thresh  # You can adjust this threshold
    filtered_actions = []
    # unique_actions = torch.unique(pred)
    
    # for verb in unique_actions:
    #     sample_idx = torch.nonzero(pred == verb).squeeze(1)
    for idx in range(topk):
        if scores[idx] >= threshold:
            verb = pred[idx]
            idxh, idxo = pairing[:, idx] + 1
            action_text = f"{actions[verb]}: {scores[idx]:.3f} ({idxh}→{idxo})"
            filtered_actions.append(action_text)
            if len(filtered_actions) >= topk:
                break
    
    # Draw the bounding boxes
    plt.figure(figsize=(12, 8))  # Make figure wider to accommodate text
    plt.imshow(image)
    plt.axis('off')
    ax = plt.gca()
    # draw_boxes(ax, boxes, scores=scores, box_thresh=box_thresh)
    draw_boxes(ax, boxes)
    
    # Add predicted actions text on the right side
    if filtered_actions:
        # Create text box on the right side
        text_x = 0.95  # Position text at 95% of image width
        text_y = 0.95  # Start from top
        line_height = 0.05  # Height between lines
        
        for i, action_text in enumerate(filtered_actions):
            plt.text(text_x, text_y - i * line_height, action_text,
                    transform=ax.transAxes, fontsize=12, fontweight='bold',
                    color='white', ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    plt.savefig(save_filename)

def visualize_gt_hoi(image, target, actions, save_filename=None, unseen_hoi_idx=None):
    """Visualise ground truth HOI pairs in the image. unseen_hoi_idx: list of unseen hoi indices (int)"""
    # Rescale the boxes to original image size
    ow, oh = image.size
    h, w = target['size']
    scale_fct = torch.as_tensor([
        ow / w, oh / h, ow / w, oh / h
    ]).unsqueeze(0)
    boxes_h = target['boxes_h'].cpu() * scale_fct
    boxes_o = target['boxes_o'].cpu() * scale_fct
    verbs = target['verb'].cpu()
    hois = target['hoi'].cpu() if 'hoi' in target else None
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.axis('off')
    ax = plt.gca()
    # Draw all boxes with index (for all unique boxes)
    all_boxes = torch.cat([boxes_h, boxes_o], dim=0)
    xy = all_boxes[:, :2].unbind(0)
    h_box, w_box = (all_boxes[:, 2:] - all_boxes[:, :2]).unbind(1)
    for i, (a, b, c) in enumerate(zip(xy, h_box.tolist(), w_box.tolist())):
        patch = patches.Rectangle(a.tolist(), b, c, facecolor='none', edgecolor='w')
        ax.add_patch(patch)
        txt = plt.text(*a.tolist(), str(i+1), fontsize=20, fontweight='semibold', color='w')
        txt.set_path_effects([peff.withStroke(linewidth=5, foreground='#000000')])
        plt.draw()
    # Draw subject/object boxes for each HOI with color, and collect text color
    filtered_actions = []
    filtered_colors = []
    for idx, (verb,) in enumerate(zip(verbs)):
        idxh = idx  # subject index in boxes_h
        idxo = idx  # object index in boxes_o
        color = 'r' if (unseen_hoi_idx is not None and hois is not None and int(hois[idx]) in unseen_hoi_idx) else 'deepskyblue'
        # Draw subject box
        patch_h = patches.Rectangle(boxes_h[idxh, :2].tolist(),
                                    (boxes_h[idxh, 2] - boxes_h[idxh, 0]).item(),
                                    (boxes_h[idxh, 3] - boxes_h[idxh, 1]).item(),
                                    facecolor='none', edgecolor=color, linewidth=3)
        ax.add_patch(patch_h)
        # Draw object box
        patch_o = patches.Rectangle(boxes_o[idxo, :2].tolist(),
                                    (boxes_o[idxo, 2] - boxes_o[idxo, 0]).item(),
                                    (boxes_o[idxo, 3] - boxes_o[idxo, 1]).item(),
                                    facecolor='none', edgecolor=color, linewidth=3, linestyle='--')
        ax.add_patch(patch_o)
        # Action text
        action_text = f"{actions[verb]} ({idxh+1}→{idxo+1+len(boxes_h)})"
        filtered_actions.append(action_text)
        filtered_colors.append(color)
    # Add HOI text on the right, with color
    if filtered_actions:
        text_x = 0.95
        text_y = 0.95
        line_height = 0.05
        for i, (action_text, color) in enumerate(zip(filtered_actions, filtered_colors)):
            plt.text(text_x, text_y - i * line_height, action_text,
                     transform=ax.transAxes, fontsize=12, fontweight='bold',
                     color=color, ha='right', va='top',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    plt.savefig(save_filename)