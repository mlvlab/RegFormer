""" Attention Pool 2D

Implementations of 2D spatial feature pooling using multi-head attention instead of average pool.

Based on idea in CLIP by OpenAI, licensed Apache 2.0
https://github.com/openai/CLIP/blob/3b473b0e682c091a9e53623eebc1ca1657385717/clip/model.py

Hacked together by / Copyright 2021 Ross Wightman
"""
""" 
(modified) SelfAttentionPool code from https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/attention_pool2d.py
(modified) CrossAttentionPool code from https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/attention_pool.py
"""
from math import e
import sys
import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import Optional, Union, Tuple, Type, Callable
from functools import partial

from timm.layers.config import use_fused_attn
from timm.layers.helpers import to_2tuple
from timm.layers.pos_embed import resample_abs_pos_embed
# from timm.layers.pos_embed_sincos import apply_rot_embed, RotaryEmbedding
from timm.layers.weight_init import trunc_normal_
from timm.layers.attention import maybe_add_mask
from timm.layers.mlp import Mlp
from timm.layers.weight_init import trunc_normal_tf_

# from sparsemax import Sparsemax
from entmax import sparsemax

from models.position_encoding import build_position_encoding
from utils.common import hard_softmax, gumbel_softmax

from models.layers import (
    LayerScale,
    NestedTensorBlock as AttentionBlock,
    SwiGLUFFNAligned as SwiGLUFFN,
)

class SelfAttentionPool(nn.Module):
    """ Attention based 2D feature pooling w/ learned (absolute) pos embedding.
    This is a multi-head attention based replacement for (spatial) average pooling in NN architectures.

    It was based on impl in CLIP by OpenAI
    https://github.com/openai/CLIP/blob/3b473b0e682c091a9e53623eebc1ca1657385717/clip/model.py

    NOTE: This requires feature size upon construction and well prevent adaptive sizing of the network.
    """
    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            in_features: int,
            feat_size: Union[int, Tuple[int, int]] = 7,
            out_features: Optional[int] = None,
            embed_dim: Optional[int] = None,
            head_dim: Optional[int] = 64,
            num_heads: Optional[int] = None,
            qkv_bias: bool = True,
            qkv_separate: bool = False,
            pool_type: str = 'token',
            # class_token: bool = False,
            drop_rate: float = 0.,
            args=None,
    ):
        super().__init__()
        assert pool_type in ('', 'token')
        self.args = args
        self.embed_dim = embed_dim = embed_dim or in_features
        self.in_features = in_features
        self.out_features = out_features or in_features
        if num_heads is not None:
            assert embed_dim % num_heads == 0
            head_dim = embed_dim // num_heads
        else:
            assert embed_dim % head_dim == 0
            num_heads = embed_dim // head_dim
        self.feat_size = to_2tuple(feat_size)
        self.seq_len = self.feat_size[0] * self.feat_size[1]
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.pool_type = pool_type
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.prefix_type = args.prefix_type
        if self.prefix_type == 'learnable':
            self.cls_token = nn.Parameter(torch.zeros(1, embed_dim))
        else:
            self.cls_token = None

        if qkv_separate:
            self.q = nn.Linear(in_features, embed_dim, bias=qkv_bias)
            self.k = nn.Linear(in_features, embed_dim, bias=qkv_bias)
            self.v = nn.Linear(in_features, embed_dim, bias=qkv_bias)
            self.qkv = None
        else:
            self.q = self.k = self.v = None
            self.qkv = nn.Linear(in_features, embed_dim * 3, bias=qkv_bias)
        self.drop = nn.Dropout(drop_rate)
        self.proj = nn.Linear(embed_dim, self.out_features)
        self.use_pos_embed = not self.args.no_pos_embed
        if self.use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(self.seq_len + 1, in_features))
            
        else:
            self.pos_embed = None
        self.init_weights()
        
    def init_weights(self, zero_init_last: bool = False):
        if self.qkv is None:
            in_features = self.q.in_features
            trunc_normal_(self.q.weight, std=in_features ** -0.5)
            nn.init.zeros_(self.q.bias)
            trunc_normal_(self.k.weight, std=in_features ** -0.5)
            nn.init.zeros_(self.k.bias)
            trunc_normal_(self.v.weight, std=in_features ** -0.5)
            nn.init.zeros_(self.v.bias)
        else:
            in_features = self.qkv.in_features
            trunc_normal_(self.qkv.weight, std=in_features ** -0.5)
            nn.init.zeros_(self.qkv.bias)
        if self.use_pos_embed:
            trunc_normal_(self.pos_embed, std=in_features ** -0.5)

    def reset(self, num_classes: Optional[int] = None, pool_type: Optional[str] = None):
        # NOTE: this module is being used as a head, so need compatible reset()
        if pool_type is not None:
            assert pool_type in ('', 'token')
            self.pool_type = pool_type
        if num_classes is not None:
            self.proj = nn.Linear(self.in_features, num_classes) if num_classes > 0 else nn.Identity()
            self.out_features = num_classes if num_classes > 0 else self.embed_dim

    def _pool(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        if self.pool_type == 'token':
            x = x[:, 0]
        else:
            # if not pooled, return spatial output without token
            x = x[:, 1:].reshape(x.shape[0], H, W, -1).permute(0, 3, 1, 2)
        return x

    def forward(self, x, pre_logits: bool = False, attn_mask: Optional[torch.Tensor] = None):
        B = x.shape[0]
        prefix_tokens, image_tokens = x[:,:-self.seq_len], x[:,-self.seq_len:]
        image_tokens = image_tokens.reshape(B,self.feat_size[0],self.feat_size[1],-1)
        image_tokens = image_tokens.permute(0,3,1,2)
        num_prefix_tokens = 1 if self.prefix_type in ['learnable', 'avg'] else prefix_tokens.shape[1]
        B, _, H, W = image_tokens.shape
        N = H * W
        if self.prefix_type in ['learnable', 'avg']:
            x = image_tokens.flatten(2).transpose(1, 2)
            if self.prefix_type == 'learnable':
                x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
            elif self.prefix_type == 'avg':
                x = torch.cat([x.mean(1, keepdim=True), x], dim=1)
            else:
                raise ValueError(f"Invalid prefix type: {self.prefix_type}")
            
            if self.use_pos_embed:
                pos_embed = resample_abs_pos_embed(self.pos_embed.unsqueeze(0), (H, W), num_prefix_tokens=num_prefix_tokens)
                x = x + pos_embed
                
        elif self.prefix_type == 'original':
            if self.use_pos_embed:
                pos_embed = resample_abs_pos_embed(self.pos_embed.unsqueeze(0), (H, W), num_prefix_tokens=num_prefix_tokens)
                x = x + pos_embed

        if self.qkv is None:
            q = self.q(x).reshape(B, N + num_prefix_tokens, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k(x).reshape(B, N + num_prefix_tokens, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v(x).reshape(B, N + num_prefix_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            x = self.qkv(x).reshape(B, -1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = x.unbind(0)

        if self.fused_attn:
            x = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = maybe_add_mask(attn, attn_mask)
            attn = attn.softmax(dim=-1)
            x = attn @ v
            
        x = x.transpose(1, 2).reshape(B, N + num_prefix_tokens, -1)
        x = self.drop(x)
        if pre_logits:
            x = self._pool(x, H, W)
            return x
        x = self.proj(x)
        x = self._pool(x, H, W)
        return x, None

class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio, qkv_bias, norm_layer, act_layer, drop, seq_len, args):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias
        self.norm_layer = norm_layer
        self.act_layer = act_layer
        self.drop = drop
        self.args = args
        self.seq_len = seq_len
        self.q = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.kv = nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()
        self.mlp = Mlp(embed_dim, int(embed_dim * mlp_ratio), act_layer=act_layer)
        
        self.drop1 = nn.Dropout(drop)
        self.drop2 = nn.Dropout(drop)
        # self.init_weights()
        
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def forward(self, query, key_value, attn_mask: Optional[torch.Tensor] = None):
        B, N_q, C = query.shape
        B, N_kv, C = key_value.shape
        q = self.q(query).reshape(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(key_value).reshape(B, N_kv, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = maybe_add_mask(attn, attn_mask)
        attn = attn.softmax(dim=-1)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N_q, -1)
        x = query + self.drop1(x)
        x = self.norm(x)
        x = x + self.drop2(self.mlp(x))
        return x, attn[:,:,-self.seq_len:]
        

class CrossAttentionPool_ml_decoder(nn.Module):
    """ ML Decoder type
    """
    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            in_features: int,
            out_features: int = None,
            embed_dim: int = None,
            num_heads: int = 8,
            feat_size: Optional[int] = None,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            pos_embed: str = '',
            pool_type: str = 'token',
            language_dim: int = 512,
            norm_layer: Optional[Type[nn.Module]] = nn.LayerNorm,
            act_layer: Optional[Type[nn.Module]] = nn.GELU,
            drop: float = 0.0,
            args=None,
    ):
        super().__init__()
        embed_dim = embed_dim or in_features
        out_features = out_features or in_features
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.feat_size = feat_size
        self.use_multi_layer_ml_decoder = args.use_multi_layer_ml_decoder
        self.num_layers = args.ml_decoder_num_layers
        self.seq_len = self.feat_size * self.feat_size
        self.scale = self.head_dim ** -0.5
        self.pool = pool_type
        self.fused_attn = use_fused_attn() and not args.return_attention_weight
        self.args = args
        self.prefix_type = args.prefix_type
        self.use_seperate_so_type = args.use_seperate_so_type

        self.vis_pos_embed = build_position_encoding(embed_dim, args)
        self.vis_pos_agg_type = args.vis_pos_agg_strategy

        self.num_queries = args.ml_decoder_num_query
        self.query_type = args.ml_decoder_query_type
        if self.query_type == 'learnable':
            self.query = nn.Parameter(torch.zeros(1, self.num_queries, embed_dim))
        else:
            self.query = None
        
        self.use_seperate_so = args.use_seperate_so
        
        
        if self.query_type in ['triplet', 'object']:
            # if args.content_feature_type == 'label_features' and args.instance_filter_type == 'score' and args.instance_agg_type == 'patch_query' and not args.language_query_as_pos_embed:
            if (args.content_feature_type == 'label_features' or 'query' in args.instance_agg_type) and not args.language_query_as_pos_embed:
                self.language_proj = nn.Identity()
                if self.use_seperate_so:
                    self.label_feat_proj = nn.Sequential(
                        nn.Linear(embed_dim*2, embed_dim),
                        nn.LayerNorm(embed_dim),
                    )
                    if self.vis_pos_embed is not None:
                        self.query_pos_embed_proj = nn.Sequential(
                            nn.Linear(embed_dim*2, embed_dim),
                            nn.LayerNorm(embed_dim),
                        )
                else:
                    self.label_feat_proj = nn.Sequential(
                        nn.Linear(embed_dim, embed_dim),
                        nn.LayerNorm(embed_dim),
                    )
                    if self.vis_pos_embed is not None:
                        self.query_pos_embed_proj = nn.Sequential(
                            nn.Linear(embed_dim, embed_dim),
                            nn.LayerNorm(embed_dim),
                        )
            else:
                if self.use_seperate_so:
                    input_dim = language_dim*2
                else:
                    input_dim = language_dim
                self.language_proj = nn.Sequential(
                        nn.Linear(input_dim, embed_dim),
                        nn.LayerNorm(embed_dim),
                    )

        if self.use_multi_layer_ml_decoder:
            self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias, norm_layer, act_layer, drop, self.seq_len, args) for _ in range(self.num_layers)])
        else:
            self.language_pos_embed_type = args.language_pos_embed_type
            self.language_query_as_pos_embed = args.language_query_as_pos_embed
            self.content_feature_type = args.content_feature_type
            if self.query_type in ['triplet', 'object']:
                if self.language_query_as_pos_embed and self.language_pos_embed_type == 'concat' and self.content_feature_type != 'zero':
                    self.q = nn.Linear(embed_dim*2, embed_dim, bias=qkv_bias)
                elif self.vis_pos_embed is not None:
                    if self.vis_pos_agg_type == 'concat':
                        self.q_content = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
                        self.q_pos = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
                    else:
                        self.q = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
                else:
                    self.q = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
            else:
                self.q = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
            if self.vis_pos_embed is not None:
                if self.vis_pos_agg_type == 'add':
                    self.k = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
                    self.v = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        
                elif self.vis_pos_agg_type == 'concat':
                    self.k_content = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
                    self.k_pos = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
                    self.v = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
                else:
                    raise ValueError(f"Invalid vis pos agg type: {self.vis_pos_agg_type}")
            else:
                self.kv = nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias)
            # self.ls1 = LayerScale(embed_dim, init_values=1e-5)
            self.ls1 = nn.Identity()
            
            self.norm = nn.LayerNorm(embed_dim)
            self.mlp = Mlp(embed_dim, int(embed_dim * mlp_ratio), act_layer=act_layer)
            # self.ls2 = LayerScale(embed_dim, init_values=1e-5)
            self.ls2 = nn.Identity()
            
        if self.query_type == 'triplet':
            self.out_norm = nn.Identity()
            self.out_proj = nn.Linear(embed_dim, 1)
            nn.init.xavier_uniform_(self.out_proj.weight)
            nn.init.constant_(self.out_proj.bias, args.scale_bias)
        else:
            self.out_norm = nn.LayerNorm(embed_dim)
            self.out_proj = nn.Linear(embed_dim, out_features, bias=False)
        
        if args.ml_decoder_num_register > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1,args.ml_decoder_num_register, embed_dim))
        else:
            self.register_tokens = None
        
        self.instance_filter_type = args.instance_filter_type
        if self.query_type == 'object':
            if self.instance_filter_type == 'score' or self.content_feature_type == 'label_features':
                # self.score_type = args.score_type
                instance_score_dim = args.instance_score_dim or out_features
                if self.use_seperate_so:
                    if args.share_so_vis_projection:
                        assert args.instance_score_dim is not None, "instance_score_dim must be specified when share_so_vis_projection is True. This is for discriminating the subject and object."
                        
                    
                    self.subject_instance_out_norm = nn.LayerNorm(embed_dim)
                    self.subject_instance_out_proj = nn.Linear(embed_dim, instance_score_dim, bias=False)
                    self.subject_patch_scale = nn.Parameter(torch.ones(1)*np.log(args.patch_scale)) if args.patch_scale is not None else None
                    if self.instance_filter_type == 'score':
                        self.subject_instance_scale = nn.Parameter(torch.ones(1)*np.log(args.instance_scale))
                        self.subject_instance_bias = nn.Parameter(torch.ones(1)*args.instance_bias)
                    self.use_human_projector = args.use_human_projector
                    if self.use_human_projector:
                        self.subject_embedding = nn.Parameter(torch.randn(1, instance_score_dim))
                    else:
                        self.subject_language_proj = nn.Sequential(nn.LayerNorm(language_dim), nn.Linear(language_dim, instance_score_dim, bias=False)) if args.instance_score_dim is not None else nn.Identity()
                    
                    
                    self.object_instance_out_norm = nn.LayerNorm(embed_dim) if not args.share_so_vis_projection else self.subject_instance_out_norm
                    self.object_instance_out_proj = nn.Linear(embed_dim, instance_score_dim, bias=False) if not args.share_so_vis_projection else self.subject_instance_out_proj
                    self.object_patch_scale = nn.Parameter(torch.ones(1)*np.log(args.patch_scale)) if args.patch_scale is not None else None
                    if self.instance_filter_type == 'score':
                        self.object_instance_scale = nn.Parameter(torch.ones(1)*np.log(args.instance_scale))
                        self.object_instance_bias = nn.Parameter(torch.ones(1)*args.instance_bias)
                    if args.share_so_lang_projection:
                        self.object_language_proj = self.subject_language_proj
                    else:
                        self.object_language_proj = nn.Sequential(nn.LayerNorm(language_dim), nn.Linear(language_dim, instance_score_dim, bias=False)) if args.instance_score_dim is not None else nn.Identity()
                else:
                    self.instance_out_norm = nn.LayerNorm(embed_dim)
                    self.instance_out_proj = nn.Linear(embed_dim, instance_score_dim, bias=False)
                    # if 'vision' in self.use_seperate_so_type:
                    #     self.subject_instance_out_norm = nn.LayerNorm(embed_dim)
                    #     self.subject_instance_out_proj = nn.Linear(embed_dim, instance_score_dim, bias=False)
                    self.so_language_proj = nn.Linear(language_dim, instance_score_dim, bias=False) if args.instance_score_dim is not None else nn.Identity()
                    self.instance_scale = nn.Parameter(torch.ones(1)*np.log(args.instance_scale))
                    self.instance_bias = nn.Parameter(torch.ones(1)*args.instance_bias)
                    self.patch_scale = nn.Parameter(torch.ones(1)*np.log(args.patch_scale)) if args.patch_scale is not None else None
                    
            elif self.instance_filter_type == 'remove_query':
                raise NotImplementedError("remove_query is not implemented")
        
        self.init_weights()

        self.instance_score_scheme = args.instance_score_scheme
        self.use_seperate_subject_pair = args.use_seperate_subject_pair
        # this can be modified in the detection stage
        self.image_level_pooling = True
        self.instance_score_post_masking_type = None
        self.post_sum_scale = None
        self.mask_generation_lib = 'torch' # 'numpy'
        self.local_scale_factor = 1.0
        self.use_masked_global_instance_score_for_sgpq = False
        
    def init_weights(self):
        # if self.pos_embed is not None:
        #     trunc_normal_tf_(self.pos_embed, std=self.pos_embed.shape[1] ** -0.5)
        if self.query is not None:
            trunc_normal_tf_(self.query, std=self.query.shape[1] ** -0.5)
        if self.register_tokens is not None:
            trunc_normal_tf_(self.register_tokens, std=self.register_tokens.shape[1] ** -0.5)
        if isinstance(self.out_proj, nn.Linear):
            nn.init.normal_(self.out_proj.weight, std=self.out_proj.in_features**-0.5)
        
        if self.use_seperate_so:
            if hasattr(self, 'subject_instance_out_proj') and isinstance(self.subject_instance_out_proj, nn.Linear):
                nn.init.normal_(self.subject_instance_out_proj.weight, std=self.subject_instance_out_proj.in_features**-0.5)
            if hasattr(self, 'object_instance_out_proj') and isinstance(self.object_instance_out_proj, nn.Linear):
                nn.init.normal_(self.object_instance_out_proj.weight, std=self.object_instance_out_proj.in_features**-0.5)
        else:
            if hasattr(self, 'instance_out_proj') and isinstance(self.instance_out_proj, nn.Linear):
                nn.init.normal_(self.instance_out_proj.weight, std=self.instance_out_proj.in_features**-0.5)    
        
        if hasattr(self, 'subject_embedding') and isinstance(self.subject_embedding, nn.Parameter):
            nn.init.normal_(self.subject_embedding, std=self.subject_embedding.shape[1] ** -0.5)
        # if 'language' in self.use_seperate_so_type:
        #     nn.init.normal_(self.subject_embedding, std=self.subject_embedding.shape[1] ** -0.5)
        #     nn.init.normal_(self.object_embedding, std=self.object_embedding.shape[1] ** -0.5)

    def boxes_to_region(self, boxes, H, W, P, device):
        grid_y = torch.arange(H, device=device).view(1, H, 1)  # (1,H,1)
        grid_x = torch.arange(W, device=device).view(1, 1, W)  # (1,1,W)
        # boxes: (P,4) [x1,y1,x2,y2]
        x1 = boxes[:, 0].view(P, 1, 1)
        y1 = boxes[:, 1].view(P, 1, 1)
        x2 = boxes[:, 2].view(P, 1, 1)
        y2 = boxes[:, 3].view(P, 1, 1)
        inside_x = (grid_x >= x1) & (grid_x < x2)
        inside_y = (grid_y >= y1) & (grid_y < y2)
        return inside_x & inside_y  # (P,H,W) bool

    def interactivenss_scoring(self, i2p_scores, meta_data, instance_score_scheme=''):
        input_size = meta_data[0]['input_size']
        h,w = input_size
        B, N, C = i2p_scores.shape
        reshaped_scores = i2p_scores.reshape(B, self.feat_size, self.feat_size, C)
        image_size = torch.tensor([w,h,w,h], device=reshaped_scores.device)
        feat_size = torch.tensor([self.feat_size,self.feat_size,self.feat_size,self.feat_size], device=reshaped_scores.device)
        ratio = feat_size/image_size
        sub_boxes = meta_data[0]['human_boxes']*ratio
        obj_boxes = meta_data[0]['object_boxes']*ratio
        union_boxes = meta_data[0]['union_boxes']*ratio
        
        sub_boxes[:,:2] = torch.floor(sub_boxes[:,:2])
        sub_boxes[:,2:] = torch.ceil(sub_boxes[:,2:])
        obj_boxes[:,:2] = torch.floor(obj_boxes[:,:2])
        obj_boxes[:,2:] = torch.ceil(obj_boxes[:,2:])
        union_boxes[:,:2] = torch.floor(union_boxes[:,:2])
        union_boxes[:,2:] = torch.ceil(union_boxes[:,2:])
        
        sub_boxes = sub_boxes.int()
        obj_boxes = obj_boxes.int()
        union_boxes = union_boxes.int()
        
        num_pairs = len(sub_boxes)
        if self.mask_generation_lib == 'numpy':
            masks = np.ones((num_pairs, self.feat_size, self.feat_size))
            for i in range(num_pairs):
                if instance_score_scheme == 's_region':
                    masks[i][sub_boxes[i][1].item():sub_boxes[i][3].item(), sub_boxes[i][0].item():sub_boxes[i][2].item()] = 0
                elif instance_score_scheme == 'o_region':
                    masks[i][obj_boxes[i][1].item():obj_boxes[i][3].item(), obj_boxes[i][0].item():obj_boxes[i][2].item()] = 0
                elif instance_score_scheme == 'union':
                    masks[i][union_boxes[i][1].item():union_boxes[i][3].item(), union_boxes[i][0].item():union_boxes[i][2].item()] = 0
                elif instance_score_scheme == 'so_region':
                    masks[i][sub_boxes[i][1].item():sub_boxes[i][3].item(), sub_boxes[i][0].item():sub_boxes[i][2].item()] = 0
                    masks[i][obj_boxes[i][1].item():obj_boxes[i][3].item(), obj_boxes[i][0].item():obj_boxes[i][2].item()] = 0
                else:
                    raise ValueError(f"Invalid instance score scheme: {instance_score_scheme}")
            masks = torch.from_numpy(masks).to(reshaped_scores.device)
        
        elif self.mask_generation_lib == 'torch':
            device = reshaped_scores.device
            if instance_score_scheme == 's_region':
                masks = self.boxes_to_region(sub_boxes, self.feat_size, self.feat_size, num_pairs, device)
            elif instance_score_scheme == 'o_region':
                masks = self.boxes_to_region(obj_boxes, self.feat_size, self.feat_size, num_pairs, device)
            elif instance_score_scheme == 'union':
                masks = self.boxes_to_region(union_boxes, self.feat_size, self.feat_size, num_pairs, device)
            elif instance_score_scheme == 'so_region':
                s_masks = self.boxes_to_region(sub_boxes, self.feat_size, self.feat_size, num_pairs, device)
                o_masks = self.boxes_to_region(obj_boxes, self.feat_size, self.feat_size, num_pairs, device)
                masks = s_masks | o_masks
            else:
                raise ValueError(f"Invalid instance score scheme: {instance_score_scheme}")
            masks = (~masks).to(dtype=reshaped_scores.dtype)
        else:
            raise ValueError(f"Invalid mask generation library: {self.mask_generation_lib}")
        # Apply mask: set masked regions (where mask=1) to -inf
        # if self.instance_score_post_masking_type is None:
        mask_value = float('-inf')
        reshaped_scores = reshaped_scores.masked_fill(masks.unsqueeze(-1).bool(), mask_value)
            # reshaped_scores = reshaped_scores * masks
        return reshaped_scores.flatten(1,2), masks.flatten(1,2)
    
    def score_normalization(self, unnormalized_scores):
        if self.args.patch_score_agg_type == 'softmax':
            scores = unnormalized_scores.softmax(dim=-2)
        elif self.args.patch_score_agg_type == 'score_sum':
            norm_scores = unnormalized_scores.sigmoid()
            scores = norm_scores/norm_scores.sum(dim=-2, keepdim=True)
        elif self.args.patch_score_agg_type == 'sparsemax':
            scores = sparsemax(unnormalized_scores, dim=-2)
        else:
            raise ValueError(f"Invalid patch score aggregation type: {self.args.patch_score_agg_type}")
        return scores
    
    def forward(self, x, attn_mask: Optional[torch.Tensor] = None, language_query: Optional[torch.Tensor] = None, so_indices=None, meta_data=None):
        B, N, C = x.shape
        extra_out_dict = {}
        
        if self.args.use_all_tokens_for_kv:
            patch_tokens = x
        else:
            patch_tokens = x[:,-self.seq_len:]
        
        # generate position encoding if needed
        vis_pos_embed = None
        if self.vis_pos_embed is not None:
            vis_pos_embed = self.vis_pos_embed(patch_tokens[...,0].reshape(B, self.feat_size,self.feat_size)).flatten(2).transpose(1,2)
        
        label_features = None
        if self.query_type == 'object':
            if self.instance_filter_type == 'score':
                if self.args.instance_agg_type == 'vector':
                    image_vectors = patch_tokens.mean(dim=1)
                    image_vectors = self.instance_out_proj(self.instance_out_norm(image_vectors))
                    image_vectors = F.normalize(image_vectors, dim=-1)
                    norm_language_query = F.normalize(language_query, dim=-1)
                    instance_logits = image_vectors @ norm_language_query.transpose(0,1)
                    instance_logits = instance_logits * self.instance_scale.exp() + self.instance_bias
                    
                    if self.args.instance_activation_type == 'sigmoid':
                        instance_scores = instance_logits.sigmoid()
                    elif self.args.instance_activation_type == 'softmax':
                        instance_scores = instance_logits.softmax(dim=-1)
                    else:
                        raise ValueError(f"Invalid instance activation type: {self.args.instance_activation_type}")
                elif self.args.instance_agg_type in ['patch', 'patch_query']:
                    patch_features = x[:,-self.seq_len:]
                    if self.use_seperate_so:
                        # subject score
                        sub_patch_projected_features = self.subject_instance_out_proj(self.subject_instance_out_norm(patch_features))
                        sub_patch_projected_features = F.normalize(sub_patch_projected_features, dim=-1)
                        if self.use_human_projector:
                            sub_language_projected = F.normalize(self.subject_embedding, dim=-1)
                            # raise NotImplementedError("use_human_projector is not implemented for seperate SO")
                        else:
                            sub_language_projected = F.normalize(self.subject_language_proj(language_query), dim=-1)
                        sub_patch_instance_similarity = sub_patch_projected_features @ sub_language_projected.transpose(0,1)
                        sub_patch_instance_logits = sub_patch_instance_similarity * self.subject_instance_scale.exp() + self.subject_instance_bias
                        
                        if self.subject_patch_scale is not None:
                            sub_i2p_scores = sub_patch_instance_similarity * self.subject_patch_scale.exp()
                        else:
                            sub_i2p_scores = sub_patch_instance_logits
                        
                        if meta_data is not None and self.instance_score_scheme != 'image':
                            # raise NotImplementedError("instance score scheme is not implemented for seperate SO")
                            if self.instance_score_post_masking_type is not None:
                                orig_sub_instance_to_patch_scores = self.score_normalization(sub_i2p_scores)
                            sub_i2p_scores, sub_pair_masks = self.interactivenss_scoring(sub_i2p_scores, meta_data, 's_region')
                            
                        # if self.args.patch_score_agg_type == 'softmax':
                        #     sub_instance_to_patch_scores = sub_i2p_scores.softmax(dim=-2)
                        # elif self.args.patch_score_agg_type == 'score_sum':
                        #     sub_instance_scores = sub_i2p_scores.sigmoid()
                        #     sub_instance_to_patch_scores = sub_instance_scores/sub_instance_scores.sum(dim=-2, keepdim=True)
                        # elif self.args.patch_score_agg_type == 'sparsemax':
                        #     sub_instance_to_patch_scores = sparsemax(dim=-2)(sub_i2p_scores)
                        # else:
                        #     raise ValueError(f"Invalid patch score aggregation type: {self.args.patch_score_agg_type}")
                        sub_instance_to_patch_scores = self.score_normalization(sub_i2p_scores)
                    
                        if self.args.instance_activation_type == 'sigmoid':
                            if self.args.post_normalize_instance_scores:
                                sub_instance_scores = sub_patch_instance_logits.sigmoid()
                                if meta_data is not None and self.instance_score_post_masking_type is not None:
                                    if self.instance_score_post_masking_type == 'pre_sum':
                                        sub_instance_scores = (orig_sub_instance_to_patch_scores * ~(sub_pair_masks.unsqueeze(-1).bool()) * sub_instance_scores).sum(dim=-2)
                                    elif self.instance_score_post_masking_type == 'post_sum':
                                        masked_sub_instance_scores = (sub_instance_to_patch_scores * sub_instance_scores).sum(dim=-2)
                                        masked_patch_importance_score = (orig_sub_instance_to_patch_scores * ~(sub_pair_masks.unsqueeze(-1).bool())).sum(dim=-2)
                                        sub_scale = self.post_sum_scale if self.post_sum_scale is not None else 1.0
                                        if self.args.vis_label_weight:
                                            extra_out_dict['patch_level_sub_instance_scores'] = sub_instance_scores
                                            extra_out_dict['masked_sub_instance_scores'] = masked_sub_instance_scores
                                            extra_out_dict['masked_sub_patch_importance_score'] = masked_patch_importance_score
                                            extra_out_dict['instance_level_sub_patch_importance_score'] = sub_instance_to_patch_scores
                                            extra_out_dict['image_level_sub_patch_importance_score'] = orig_sub_instance_to_patch_scores
                                            extra_out_dict['sub_pair_masks'] = sub_pair_masks
                                        sub_instance_scores = masked_sub_instance_scores**self.local_scale_factor * masked_patch_importance_score ** sub_scale
                                        
                                        
                                    else:
                                        raise ValueError(f"Invalid instance score post masking type: {self.instance_score_post_masking_type}")
                                else:
                                    sub_instance_scores = (sub_instance_to_patch_scores * sub_instance_scores).sum(dim=-2)
                            else:
                                sub_instance_scores = (sub_instance_to_patch_scores * sub_patch_instance_logits).sum(dim=-2)
                                sub_instance_scores = sub_instance_scores.sigmoid()
                        elif self.args.instance_activation_type == 'softmax':
                            raise NotImplementedError("softmax is not implemented for seperate SO")
                        else:
                            raise ValueError(f"Invalid instance activation type: {self.args.instance_activation_type}")
                        
                        # object score
                        obj_patch_projected_features = self.object_instance_out_proj(self.object_instance_out_norm(patch_features))
                        obj_patch_projected_features = F.normalize(obj_patch_projected_features, dim=-1)
                        obj_language_projected = F.normalize(self.object_language_proj(language_query), dim=-1)
                        obj_patch_instance_similarity = obj_patch_projected_features @ obj_language_projected.transpose(0,1)
                        obj_patch_instance_logits = obj_patch_instance_similarity * self.object_instance_scale.exp() + self.object_instance_bias
                        
                        if self.object_patch_scale is not None:
                            obj_i2p_scores = obj_patch_instance_similarity * self.object_patch_scale.exp()
                        else:
                            obj_i2p_scores = obj_patch_instance_logits
                        
                        if meta_data is not None and self.instance_score_scheme != 'image':
                            # raise NotImplementedError("instance score scheme is not implemented for seperate SO")
                            if self.instance_score_post_masking_type is not None:
                                orig_obj_instance_to_patch_scores = self.score_normalization(obj_i2p_scores)
                            obj_i2p_scores, obj_pair_masks = self.interactivenss_scoring(obj_i2p_scores, meta_data, 'o_region')
                            
                        # if self.args.patch_score_agg_type == 'softmax':
                        #     obj_instance_to_patch_scores = obj_i2p_scores.softmax(dim=-2)
                        # elif self.args.patch_score_agg_type == 'score_sum':
                        #     obj_instance_scores = obj_i2p_scores.sigmoid()
                        #     obj_instance_to_patch_scores = obj_instance_scores/obj_instance_scores.sum(dim=-2, keepdim=True)
                        # elif self.args.patch_score_agg_type == 'sparsemax':
                        #     obj_instance_to_patch_scores = sparsemax(dim=-2)(obj_i2p_scores)
                        # else:
                        #     raise ValueError(f"Invalid patch score aggregation type: {self.args.patch_score_agg_type}")
                        obj_instance_to_patch_scores = self.score_normalization(obj_i2p_scores)
                        
                        if self.args.instance_activation_type == 'sigmoid':
                            if self.args.post_normalize_instance_scores:
                                obj_instance_scores = obj_patch_instance_logits.sigmoid()
                                if meta_data is not None and self.instance_score_post_masking_type is not None:
                                    if self.instance_score_post_masking_type == 'pre_sum':
                                        obj_instance_scores = (obj_instance_to_patch_scores * ~(obj_pair_masks.unsqueeze(-1).bool()) * obj_instance_scores).sum(dim=-2)
                                    elif self.instance_score_post_masking_type == 'post_sum':
                                        masked_obj_instance_scores = (obj_instance_to_patch_scores * obj_instance_scores).sum(dim=-2)
                                        masked_patch_importance_score = (orig_obj_instance_to_patch_scores * ~(obj_pair_masks.unsqueeze(-1).bool())).sum(dim=-2)
                                        obj_scale = self.post_sum_scale if self.post_sum_scale is not None else 1.0
                                        if self.args.vis_label_weight:
                                            extra_out_dict['patch_level_obj_instance_scores'] = obj_instance_scores
                                            extra_out_dict['masked_obj_instance_scores'] = masked_obj_instance_scores
                                            extra_out_dict['masked_obj_patch_importance_score'] = masked_patch_importance_score
                                            extra_out_dict['instance_level_obj_patch_importance_score'] = obj_instance_to_patch_scores
                                            extra_out_dict['image_level_obj_patch_importance_score'] = orig_obj_instance_to_patch_scores
                                            extra_out_dict['obj_pair_masks'] = obj_pair_masks
                                        obj_instance_scores = masked_obj_instance_scores**self.local_scale_factor * masked_patch_importance_score ** obj_scale
                                        
                                    else:
                                        raise ValueError(f"Invalid instance score post masking type: {self.instance_score_post_masking_type}")
                                else:
                                    obj_instance_scores = (obj_instance_to_patch_scores * obj_instance_scores).sum(dim=-2)
                            else:
                                obj_instance_scores = (obj_instance_to_patch_scores * obj_patch_instance_logits).sum(dim=-2)
                                obj_instance_scores = obj_instance_scores.sigmoid()
                        elif self.args.instance_activation_type == 'softmax':
                            raise NotImplementedError("softmax is not implemented for seperate SO")
                        else:
                            raise ValueError(f"Invalid instance activation type: {self.args.instance_activation_type}")
                        
                        if self.use_seperate_subject_pair:
                            instance_scores = (sub_instance_scores * obj_instance_scores)
                        else:
                            instance_scores = (sub_instance_scores[:,:1] * obj_instance_scores)
                        
                        if 'query' in self.args.instance_agg_type:
                            if self.args.feature_agg_type == 'hard':
                                sub_hard_pred_per_patch = hard_softmax(sub_patch_instance_logits, dim=-1)
                                obj_hard_pred_per_patch = hard_softmax(obj_patch_instance_logits, dim=-1)
                                sub_features = sub_hard_pred_per_patch.transpose(1,2) @ patch_features
                                sub_features = sub_features/sub_hard_pred_per_patch.sum(dim=1).unsqueeze(-1).clamp(min=1.)
                                obj_features = obj_hard_pred_per_patch.transpose(1,2) @ patch_features
                                obj_features = obj_features/obj_hard_pred_per_patch.sum(dim=1).unsqueeze(-1).clamp(min=1.)
                                if self.use_seperate_subject_pair:
                                    sub_features = sub_features
                                else:
                                    sub_features = sub_features[:,:1].repeat(1, obj_features.shape[1], 1)
                                label_features = torch.cat([sub_features, obj_features], dim=-1)
                            elif self.args.feature_agg_type == 'soft':
                                sub_features = sub_instance_to_patch_scores.transpose(1,2) @ patch_features
                                obj_features = obj_instance_to_patch_scores.transpose(1,2) @ patch_features
                                if self.use_seperate_subject_pair:
                                    sub_features = sub_features
                                else:
                                    sub_features = sub_features[:,:1].repeat(1, obj_features.shape[1], 1) # just because the subject is always human                                
                                if vis_pos_embed is not None:
                                    sub_pos_embed = sub_instance_to_patch_scores[...,:1].transpose(1,2) @ vis_pos_embed
                                    obj_pos_embed = obj_instance_to_patch_scores.transpose(1,2) @ vis_pos_embed
                                    sub_pos_embed = sub_pos_embed.repeat(1, obj_pos_embed.shape[1], 1)
                                    query_pos_embed = torch.cat([sub_pos_embed, obj_pos_embed], dim=-1)
                
                                label_features = torch.cat([sub_features, obj_features], dim=-1)
                                    
                            else:
                                raise ValueError(f"Invalid feature aggregation type: {self.args.feature_agg_type}")
                        
                        extra_out_dict['sub_instance_scores'] = sub_instance_scores
                        extra_out_dict['obj_instance_scores'] = obj_instance_scores
                        
                        if self.args.vis_label_weight:
                            extra_out_dict['sub_instance_to_patch_scores'] = sub_instance_to_patch_scores
                            extra_out_dict['obj_instance_to_patch_scores'] = obj_instance_to_patch_scores
                            extra_out_dict['sub_patch_to_instance_scores'] = sub_patch_instance_logits.sigmoid()
                            extra_out_dict['obj_patch_to_instance_scores'] = obj_patch_instance_logits.sigmoid()
                        
                    else:
                        patch_projected_features = self.instance_out_proj(self.instance_out_norm(patch_features))
                        patch_projected_features = F.normalize(patch_projected_features, dim=-1)
                        norm_language_query = F.normalize(self.so_language_proj(language_query), dim=-1)
                        
                        patch_instance_similarity = patch_projected_features @ norm_language_query.transpose(0,1)
                        patch_instance_logits = patch_instance_similarity * self.instance_scale.exp() + self.instance_bias
                        # patch_to_instance_scores = patch_instance_logits.sigmoid()
                        if self.patch_scale is not None:
                            i2p_scores = patch_instance_similarity * self.patch_scale.exp()
                        else:
                            i2p_scores = patch_instance_logits
                        
                        if meta_data is not None and self.instance_score_scheme != 'image':
                            i2p_scores, pair_masks = self.interactivenss_scoring(i2p_scores, meta_data, self.instance_score_scheme)
                        instance_to_patch_scores = i2p_scores.softmax(dim=-2)
                            
                        if self.args.patch_score_stop_gd:
                            instance_to_patch_scores = instance_to_patch_scores.detach()
                        
                        if self.args.instance_activation_type == 'sigmoid':
                            instance_scores = (instance_to_patch_scores * patch_instance_logits).sum(dim=-2)
                            instance_scores = instance_scores.sigmoid()
                        elif self.args.instance_activation_type == 'softmax':
                            if self.args.revised_softmax:
                                patch_to_instance_scores = patch_instance_logits.softmax(dim=-1)
                                instance_scores = (instance_to_patch_scores * patch_to_instance_scores).sum(dim=-2)
                            else:
                                instance_scores = (instance_to_patch_scores * patch_instance_logits).sum(dim=-2)
                                instance_scores = instance_scores.softmax(dim=-1)
                            
                        else:
                            raise ValueError(f"Invalid instance activation type: {self.args.instance_activation_type}")
                        
                        if self.args.vis_label_weight:
                            extra_out_dict['instance_to_patch_scores'] = instance_to_patch_scores
                            extra_out_dict['patch_to_instance_scores'] = patch_to_instance_scores if self.args.instance_activation_type == 'softmax' else patch_instance_logits.sigmoid()
                            
                        if 'query' in self.args.instance_agg_type:
                            # replace language query with aggregated patch features
                            if self.args.feature_agg_type == 'hard':
                                hard_pred_per_patch = hard_softmax(patch_instance_logits, dim=-1)
                                label_features = hard_pred_per_patch.transpose(1,2) @ patch_features
                                label_features = label_features/hard_pred_per_patch.sum(dim=1).unsqueeze(-1).clamp(min=1.)
                            elif self.args.feature_agg_type == 'soft':
                                label_features = instance_to_patch_scores.transpose(1,2) @ patch_features
                                if vis_pos_embed is not None:
                                    query_pos_embed = instance_to_patch_scores.transpose(1,2) @ vis_pos_embed
                            else:
                                raise ValueError(f"Invalid feature aggregation type: {self.args.feature_agg_type}")
                        
                else:
                    raise ValueError(f"Invalid instance aggregation type: {self.args.instance_agg_type}")
                # if self.score_type == 'so':
                #     # TODO: This uses index 0 only for HOI tasks. Need to modify for SGG.
                #     instance_scores = instance_scores * instance_scores[:,0].unsqueeze(1)
                extra_out_dict['instance_scores'] = instance_scores
                
        if self.query_type == 'learnable':
            num_queries = self.num_queries
            q_latent = self.query.expand(B, -1, -1)
        elif self.query_type in ['triplet','object']:
            if self.use_seperate_so:
                sub_language_query = language_query[:1]
                if so_indices is not None:
                    obj_language_query = language_query[so_indices]
                else:
                    obj_language_query = language_query
                sub_language_query = sub_language_query.repeat(len(obj_language_query), 1)
                so_language_query = torch.cat([sub_language_query, obj_language_query], dim=-1)
                num_queries = len(so_language_query)
                unsqueeze_dim = 0 if self.image_level_pooling else 1
                proj_language_query = self.language_proj(so_language_query).unsqueeze(unsqueeze_dim)
                
            else:
                
                if so_indices is not None and self.query_type == 'object':
                    language_query = language_query[so_indices]
                num_queries = len(language_query)
                proj_language_query = self.language_proj(language_query).unsqueeze(0)
            if not self.image_level_pooling:
                if self.query_type == 'object':
                    num_queries = 1
                    attn_mask = None
                else:
                    attn_mask = None
            if self.language_query_as_pos_embed:
                if self.content_feature_type == 'avg_token':
                    q_latent = patch_tokens.mean(1, keepdim=True).expand(-1, num_queries, -1)
                    pos_embed = proj_language_query.expand(B, -1, -1)
                elif self.content_feature_type == 'zero':
                    q_latent = torch.zeros(B, num_queries, C, device=x.device, dtype=x.dtype)
                    pos_embed = proj_language_query.expand(B, -1, -1)
                elif self.content_feature_type == 'label_features':
                    if label_features is not None:
                        # shape of label_features: B, num_queries, C
                        if so_indices is not None:
                            label_features = label_features[:,so_indices]
                        q_latent = label_features
                    else:
                        patch_features = x[:,-self.seq_len:]
                        patch_projected_features = self.instance_out_proj(self.instance_out_norm(patch_features))
                        patch_projected_features = F.normalize(patch_projected_features, dim=-1)
                        norm_language_query = F.normalize(language_query, dim=-1)
                        patch_instance_similarity = patch_projected_features @ norm_language_query.transpose(0,1)
                        patch_instance_logits = patch_instance_similarity * self.instance_scale.exp() + self.instance_bias
                        hard_pred_per_patch = hard_softmax(patch_instance_logits, dim=-1)
                        label_features = hard_pred_per_patch.transpose(1,2) @ patch_features
                        label_features = label_features/hard_pred_per_patch.sum(dim=1).unsqueeze(-1).clamp(min=1.)
                        if so_indices is not None:
                            label_features = label_features[:,so_indices]
                        q_latent = label_features
                    pos_embed = proj_language_query.expand(B, -1, -1)
                else:
                    raise ValueError(f"Invalid content feature type: {self.content_feature_type}")
            else:
                # q_latent = proj_language_query.expand(B, -1, -1)
                
                if self.content_feature_type == 'label_features' or 'query' in self.args.instance_agg_type:

                    if label_features is not None:
                        # shape of label_features: B, num_queries, C
                        if so_indices is not None:
                            if self.image_level_pooling:
                                unsqueeze_dim=0
                            else:
                                unsqueeze_dim=1
                            label_features = label_features[torch.arange(label_features.shape[0]),so_indices].unsqueeze(unsqueeze_dim)
                            if vis_pos_embed is not None:
                                query_pos_embed = query_pos_embed[torch.arange(query_pos_embed.shape[0]),so_indices].unsqueeze(unsqueeze_dim)
                        # q_latent = self.label_feat_proj(label_features)
                    else:
                        if self.use_seperate_so:
                            patch_features = x[:,-self.seq_len:]
                            # subject score
                            sub_patch_projected_features = self.subject_instance_out_proj(self.subject_instance_out_norm(patch_features))
                            sub_patch_projected_features = F.normalize(sub_patch_projected_features, dim=-1)
                            if self.use_human_projector:
                                sub_language_projected = F.normalize(self.subject_embedding, dim=-1)
                            else:
                                sub_language_projected = F.normalize(self.subject_language_proj(language_query), dim=-1)
                            sub_patch_instance_similarity = sub_patch_projected_features @ sub_language_projected.transpose(0,1)
                            # sub_patch_instance_logits = sub_patch_instance_similarity * self.subject_instance_scale.exp() + self.subject_instance_bias
                            
                            sub_i2p_scores = sub_patch_instance_similarity * self.subject_patch_scale.exp() if self.subject_patch_scale is not None else sub_patch_instance_similarity
                                                        
                            if meta_data is not None and self.instance_score_scheme != 'image':
                                # raise NotImplementedError("instance score scheme is not implemented for seperate SO")
                                if self.use_masked_global_instance_score_for_sgpq:
                                    orig_sub_instance_to_patch_scores = self.score_normalization(sub_i2p_scores)
                                sub_i2p_scores, sub_pair_masks = self.interactivenss_scoring(sub_i2p_scores, meta_data, 's_region')
                            
                            sub_instance_to_patch_scores = sub_i2p_scores.softmax(dim=-2)
                            
                            # object score
                            obj_patch_projected_features = self.object_instance_out_proj(self.object_instance_out_norm(patch_features))
                            obj_patch_projected_features = F.normalize(obj_patch_projected_features, dim=-1)
                            obj_language_projected = F.normalize(self.object_language_proj(language_query), dim=-1)
                            obj_patch_instance_similarity = obj_patch_projected_features @ obj_language_projected.transpose(0,1)
                            # obj_patch_instance_logits = obj_patch_instance_similarity * self.object_instance_scale.exp() + self.object_instance_bias
                            
                            obj_i2p_scores = obj_patch_instance_similarity * self.object_patch_scale.exp() if self.object_patch_scale is not None else obj_patch_instance_similarity
                            
                            if meta_data is not None and self.instance_score_scheme != 'image':
                                # raise NotImplementedError("instance score scheme is not implemented for seperate SO")
                                if self.use_masked_global_instance_score_for_sgpq:
                                    orig_obj_instance_to_patch_scores = self.score_normalization(obj_i2p_scores)
                                obj_i2p_scores, obj_pair_masks = self.interactivenss_scoring(obj_i2p_scores, meta_data, 'o_region')
                            obj_instance_to_patch_scores = obj_i2p_scores.softmax(dim=-2)
                            
                            if self.args.feature_agg_type == 'hard':
                                sub_hard_pred_per_patch = hard_softmax(sub_patch_instance_logits, dim=-1)
                                obj_hard_pred_per_patch = hard_softmax(obj_patch_instance_logits, dim=-1)
                                sub_features = sub_hard_pred_per_patch.transpose(1,2) @ patch_features
                                sub_features = sub_features/sub_hard_pred_per_patch.sum(dim=1).unsqueeze(-1).clamp(min=1.)
                                obj_features = obj_hard_pred_per_patch.transpose(1,2) @ patch_features
                                obj_features = obj_features/obj_hard_pred_per_patch.sum(dim=1).unsqueeze(-1).clamp(min=1.)
                                if self.use_seperate_subject_pair:
                                    sub_features = sub_features
                                else:
                                    sub_features = sub_features[:,:1].repeat(1, obj_features.shape[1], 1)
                                label_features = torch.cat([sub_features, obj_features], dim=-1)
                            elif self.args.feature_agg_type == 'soft':
                                sub_features = sub_instance_to_patch_scores.transpose(1,2) @ patch_features
                                obj_features = obj_instance_to_patch_scores.transpose(1,2) @ patch_features
                                if self.use_seperate_subject_pair:
                                    sub_features = sub_features
                                else:
                                    sub_features = sub_features[:,:1].repeat(1, obj_features.shape[1], 1) # just because the subject is always human
                                label_features = torch.cat([sub_features, obj_features], dim=-1)
                                if vis_pos_embed is not None:
                                    sub_pos_embed = sub_instance_to_patch_scores[...,:1].transpose(1,2) @ vis_pos_embed
                                    obj_pos_embed = obj_instance_to_patch_scores.transpose(1,2) @ vis_pos_embed
                                    sub_pos_embed = sub_pos_embed.repeat(1,obj_pos_embed.shape[1], 1)
                                    query_pos_embed = torch.cat([sub_pos_embed, obj_pos_embed], dim=-1)
                            else:
                                raise ValueError(f"Invalid feature aggregation type: {self.args.feature_agg_type}")
                            
                            if self.use_masked_global_instance_score_for_sgpq:
                                masked_sub_instance_scores = (orig_sub_instance_to_patch_scores * ~(sub_pair_masks.unsqueeze(-1).bool())).sum(dim=-2)
                                masked_obj_instance_scores = (orig_obj_instance_to_patch_scores * ~(obj_pair_masks.unsqueeze(-1).bool())).sum(dim=-2)
                                sub_scale = self.post_sum_scale if self.post_sum_scale is not None else 1.0
                                sub_instance_scores = masked_sub_instance_scores**sub_scale
                                obj_scale = self.post_sum_scale if self.post_sum_scale is not None else 1.0
                                obj_instance_scores = masked_obj_instance_scores**obj_scale
                                extra_out_dict['masked_sub_instance_scores'] = masked_sub_instance_scores
                                extra_out_dict['masked_obj_instance_scores'] = masked_obj_instance_scores
                                extra_out_dict['sub_pair_masks'] = sub_pair_masks
                                extra_out_dict['obj_pair_masks'] = obj_pair_masks
                                extra_out_dict['sub_instance_scores'] = sub_instance_scores
                                extra_out_dict['obj_instance_scores'] = obj_instance_scores
                                instance_scores = (sub_instance_scores[:,:1] * obj_instance_scores)
                                extra_out_dict['instance_scores'] = instance_scores
                            
                            if self.args.vis_label_weight:
                                extra_out_dict['sub_instance_to_patch_scores'] = sub_instance_to_patch_scores
                                extra_out_dict['obj_instance_to_patch_scores'] = obj_instance_to_patch_scores
                        else:
                            patch_features = x[:,-self.seq_len:]
                            patch_projected_features = self.instance_out_proj(self.instance_out_norm(patch_features))
                            patch_projected_features = F.normalize(patch_projected_features, dim=-1)
                            norm_language_query = F.normalize(self.so_language_proj(language_query), dim=-1)
                            
                            patch_instance_similarity = patch_projected_features @ norm_language_query.transpose(0,1)
                            # patch_instance_logits = patch_instance_similarity * self.instance_scale.exp() + self.instance_bias
                            # patch_to_instance_scores = patch_instance_logits.sigmoid()
                            i2p_scores = patch_instance_similarity * self.patch_scale.exp() if self.patch_scale is not None else patch_instance_similarity
                            
                            if meta_data is not None and self.args.instance_score_scheme != 'image':
                                i2p_scores, pair_masks = self.interactivenss_scoring(i2p_scores, meta_data, self.args.instance_score_scheme)
                            instance_to_patch_scores = i2p_scores.softmax(dim=-2)
                                
                            if self.args.patch_score_stop_gd:
                                instance_to_patch_scores = instance_to_patch_scores.detach()
                            
                            if self.args.vis_label_weight:
                                extra_out_dict['instance_to_patch_scores'] = instance_to_patch_scores
                                
                            # replace language query with aggregated patch features
                            if self.args.feature_agg_type == 'hard':
                                hard_pred_per_patch = hard_softmax(patch_instance_logits, dim=-1)
                                label_features = hard_pred_per_patch.transpose(1,2) @ patch_features
                                label_features = label_features/hard_pred_per_patch.sum(dim=1).unsqueeze(-1).clamp(min=1.)
                            elif self.args.feature_agg_type == 'soft':
                                label_features = instance_to_patch_scores.transpose(1,2) @ patch_features
                                if vis_pos_embed is not None:
                                    query_pos_embed = instance_to_patch_scores.transpose(1,2) @ vis_pos_embed
                            else:
                                raise ValueError(f"Invalid feature aggregation type: {self.args.feature_agg_type}")
                            
                        if so_indices is not None:
                            if self.image_level_pooling:
                                unsqueeze_dim=0
                            else:
                                unsqueeze_dim=1
                            label_features = label_features[torch.arange(label_features.shape[0]),so_indices].unsqueeze(unsqueeze_dim)
                            if vis_pos_embed is not None:
                                query_pos_embed = query_pos_embed[torch.arange(query_pos_embed.shape[0]),so_indices].unsqueeze(unsqueeze_dim)
                        # patch_features = x[:,-self.seq_len:]
                        # patch_projected_features = self.instance_out_proj(self.instance_out_norm(patch_features))
                        # patch_projected_features = F.normalize(patch_projected_features, dim=-1)
                        # norm_language_query = F.normalize(language_query, dim=-1)
                        # patch_instance_similarity = patch_projected_features @ norm_language_query.transpose(0,1)
                        # patch_instance_logits = patch_instance_similarity * self.instance_scale.exp() + self.instance_bias
                        # hard_pred_per_patch = hard_softmax(patch_instance_logits, dim=-1)
                        # label_features = hard_pred_per_patch.transpose(1,2) @ patch_features
                        # label_features = label_features/hard_pred_per_patch.sum(dim=1).unsqueeze(-1).clamp(min=1.)
                        # if so_indices is not None:
                        #     label_features = label_features[:,so_indices]
                        # q_latent = label_features
                    
                    q_latent = self.label_feat_proj(label_features)
                    if vis_pos_embed is not None:
                        query_pos_embed = self.query_pos_embed_proj(query_pos_embed)
                else:
                    if self.use_seperate_so and not self.image_level_pooling:
                        q_latent = proj_language_query
                    else:
                        q_latent = proj_language_query.expand(B, -1, -1)
        else:
            raise ValueError(f"Invalid query type: {self.query_type}")
        
        if self.register_tokens is not None:
            register_tokens = self.register_tokens.expand(B, -1, -1)
            patch_tokens = torch.cat([register_tokens, patch_tokens], dim=1)
            if attn_mask is not None:
                register_mask = torch.zeros((B, 1, attn_mask.shape[-2], self.register_tokens.shape[1]),device=attn_mask.device, dtype=attn_mask.dtype)
                attn_mask = torch.cat([register_mask, attn_mask], dim=-1)
            
        num_seq = patch_tokens.shape[1]
        # if self.pos_embed is not None:
        #     # FIXME interpolate
        #     patch_tokens = patch_tokens + self.pos_embed.unsqueeze(0).to(patch_tokens.dtype)
    
        if self.use_multi_layer_ml_decoder:
            attn = None
            x = q_latent
            for block in self.blocks:
                x, attn = block(x, patch_tokens, attn_mask)
        else:
            if self.language_query_as_pos_embed:
                if self.language_pos_embed_type == 'concat' and self.content_feature_type != 'zero':
                    qq = torch.cat([q_latent, pos_embed], dim=-1)
                elif self.language_pos_embed_type == 'add':
                    qq = q_latent + pos_embed
                else:
                    raise ValueError(f"Invalid language pos embed type: {self.language_pos_embed_type}")
            else:
                qq = q_latent
                
            if vis_pos_embed is not None:
                if self.vis_pos_agg_type == 'add':
                    q = self.q(qq+query_pos_embed).reshape(B, num_queries, self.num_heads, self.head_dim).transpose(1, 2)
                elif self.vis_pos_agg_type == 'concat':
                    q = self.q_content(qq).reshape(B, num_queries, self.num_heads, self.head_dim)
                    q_pos = self.q_pos(query_pos_embed).reshape(B, num_queries, self.num_heads, self.head_dim)
                    q = torch.cat([q, q_pos], dim=-1).transpose(1, 2)
                else:
                    raise ValueError(f"Invalid vis pos agg type: {self.vis_pos_agg_type}")
            else:
                q = self.q(qq).reshape(B, num_queries, self.num_heads, self.head_dim).transpose(1, 2)

            if vis_pos_embed is not None:
                if self.vis_pos_agg_type == 'add':
                    k = self.k(patch_tokens+vis_pos_embed).reshape(B, num_seq, self.num_heads, self.head_dim).transpose(1, 2)
                    v = self.v(patch_tokens).reshape(B, num_seq, self.num_heads, self.head_dim).transpose(1, 2)
                elif self.vis_pos_agg_type == 'concat':
                    
                    k_content = self.k_content(patch_tokens).reshape(B, num_seq, self.num_heads, self.head_dim)
                    k_pos = self.k_pos(vis_pos_embed).reshape(B, num_seq, self.num_heads, self.head_dim)
                    k = torch.cat([k_content, k_pos], dim=-1).transpose(1, 2)
                    v = self.v(patch_tokens).reshape(B, num_seq, self.num_heads, self.head_dim).transpose(1, 2)
                else:
                    raise ValueError(f"Invalid vis pos agg type: {self.vis_pos_agg_type}")
                
            else:
                kv = self.kv(patch_tokens).reshape(B, num_seq, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
                k, v = kv.unbind(0)

            attn = None
            if self.fused_attn:
                attn_x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            else:
                q = q * self.scale
                attn = q @ k.transpose(-2, -1)
                attn = maybe_add_mask(attn, attn_mask)
                attn = attn.softmax(dim=-1)
                attn_x = attn @ v
                attn = attn[...,-self.seq_len:]
                
            attn_x = attn_x.transpose(1, 2).reshape(B, num_queries, C)
            x = q_latent + self.ls1(attn_x)
            x = self.norm(x)
            x = x + self.ls2(self.mlp(x))
        x = self.out_norm(x)
        x = self.out_proj(x)
        # optional pool if latent seq_len > 1 and pooled output is desired
        if self.query_type == 'triplet':
            x = x.squeeze(-1)
        
        if not self.image_level_pooling and self.query_type!='triplet':
            x = x.transpose(0,1)
        return x, attn, extra_out_dict

class CrossAttentionPool_v2(nn.Module):
    """ Attention pooling w/ latent query
    """
    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            in_features: int,
            out_features: int = None,
            embed_dim: int = None,
            num_heads: int = 8,
            feat_size: Optional[int] = None,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            latent_len: int = 1,
            latent_dim: int = None,
            pos_embed: str = '',
            pool_type: str = 'token',
            norm_layer: Optional[Type[nn.Module]] = nn.LayerNorm,
            act_layer: Optional[Type[nn.Module]] = nn.GELU,
            drop: float = 0.0,
            args=None,
    ):
        super().__init__()
        embed_dim = embed_dim or in_features
        out_features = out_features or in_features
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.feat_size = feat_size
        self.seq_len = self.feat_size * self.feat_size
        self.scale = self.head_dim ** -0.5
        self.pool = pool_type
        self.fused_attn = use_fused_attn() and not args.return_attention_weight
        self.args = args
        self.prefix_type = args.prefix_type
        if pos_embed == 'abs':
            assert feat_size is not None
            self.pos_embed = nn.Parameter(torch.zeros(self.seq_len, in_features))
        else:
            self.pos_embed = None
            
        if self.prefix_type == 'learnable':
            self.latent_dim = latent_dim or embed_dim
            self.latent_len = latent_len
            self.latent = nn.Parameter(torch.zeros(1, self.latent_len, embed_dim))
        
        if 'dual' in self.prefix_type:
            self.latent_dim = latent_dim or embed_dim
            self.latent_len = 2
            self.latent = nn.Parameter(torch.zeros(1, self.latent_len, embed_dim))

        self.q = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.kv = nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias)
        self.ls1 = LayerScale(embed_dim, init_values=1e-5)
        
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()
        self.mlp = Mlp(embed_dim, int(embed_dim * mlp_ratio), act_layer=act_layer)
        self.ls2 = LayerScale(embed_dim, init_values=1e-5)

        self.out_norm = norm_layer(embed_dim)
        if self.prefix_type == 'avg_patch_dual':
            self.out_proj = nn.Linear(embed_dim, out_features//2)
        else:
            self.out_proj = nn.Linear(embed_dim, out_features)
        self.init_weights()

    def init_weights(self):
        if self.pos_embed is not None:
            trunc_normal_tf_(self.pos_embed, std=self.pos_embed.shape[1] ** -0.5)
        if self.prefix_type == 'learnable':
            trunc_normal_tf_(self.latent, std=self.latent_dim ** -0.5)
        if isinstance(self.out_proj, nn.Linear):
            nn.init.normal_(self.out_proj.weight, std=self.out_proj.in_features**-0.5)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        B, N, C = x.shape
        if self.prefix_type == 'learnable':
            num_prefix_tokens = self.latent_len
            q_latent = self.latent.expand(B, -1, -1)
        elif self.prefix_type == 'avg':
            num_prefix_tokens = 1
            q_latent = x.mean(1, keepdim=True)
        elif self.prefix_type == 'avg_patch':
            num_prefix_tokens = 1
            q_latent = x[:,-self.seq_len:].mean(1, keepdim=True)
        elif self.prefix_type == 'avg_patch_dual':
            num_prefix_tokens = 2
            q_latent = x[:,-self.seq_len:].mean(1, keepdim=True)
            q_latent = q_latent + self.latent
        elif self.prefix_type == 'original':
            num_prefix_tokens = N - self.seq_len
            q_latent = x[:,:num_prefix_tokens]
        else:
            raise ValueError(f"Invalid prefix type: {self.prefix_type}")
        
        if self.args.use_all_tokens_for_kv:
            patch_tokens = x
        else:
            patch_tokens = x[:,-self.seq_len:]
        num_seq = self.seq_len
        if self.pos_embed is not None:
            # FIXME interpolate
            patch_tokens = patch_tokens + self.pos_embed.unsqueeze(0).to(patch_tokens.dtype)
    
        q = self.q(q_latent).reshape(B, num_prefix_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        kv = self.kv(patch_tokens).reshape(B, num_seq, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        attn = None
        if self.fused_attn:
            attn_x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = maybe_add_mask(attn, attn_mask)
            attn = attn.softmax(dim=-1)
            attn_x = attn @ v
            
        attn_x = attn_x.transpose(1, 2).reshape(B, num_prefix_tokens, C)
        x = q_latent + self.ls1(attn_x)
        x = x + self.ls2(self.mlp(self.norm(x)))
        x = self.out_norm(x)
        x = self.out_proj(x)
        # optional pool if latent seq_len > 1 and pooled output is desired
        if 'dual' in self.prefix_type:
            x = torch.cat([x[:,0], x[:,1]], dim=-1)
        else:
            if self.pool == 'token':
                x = x[:, 0]
            elif self.pool == 'avg':
                x = x.mean(1)
        return x, attn

class CrossAttentionPool(nn.Module):
    """ Attention pooling w/ latent query
    """
    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            in_features: int,
            out_features: int = None,
            embed_dim: int = None,
            num_heads: int = 8,
            feat_size: Optional[int] = None,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            latent_len: int = 1,
            latent_dim: int = None,
            pos_embed: str = '',
            pool_type: str = 'token',
            norm_layer: Optional[Type[nn.Module]] = nn.LayerNorm,
            act_layer: Optional[Type[nn.Module]] = nn.GELU,
            drop: float = 0.0,
            args=None,
    ):
        super().__init__()
        embed_dim = embed_dim or in_features
        out_features = out_features or in_features
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.feat_size = feat_size
        self.seq_len = self.feat_size * self.feat_size
        self.scale = self.head_dim ** -0.5
        self.pool = pool_type
        self.fused_attn = use_fused_attn() and not args.return_attention_weight
        self.args = args
        self.prefix_type = args.prefix_type
        if pos_embed == 'abs':
            assert feat_size is not None
            self.pos_embed = nn.Parameter(torch.zeros(self.seq_len, in_features))
        else:
            self.pos_embed = None
            
        if self.prefix_type == 'learnable':
            self.latent_dim = latent_dim or embed_dim
            self.latent_len = latent_len
            self.latent = nn.Parameter(torch.zeros(1, self.latent_len, embed_dim))
        

        self.q = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.kv = nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias)
        if qk_norm:
            qk_norm_layer = norm_layer or nn.LayerNorm
            self.q_norm = qk_norm_layer(self.head_dim)
            self.k_norm = qk_norm_layer(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(drop)

        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()
        self.mlp = Mlp(embed_dim, int(embed_dim * mlp_ratio), act_layer=act_layer)
        self.layer_scale = LayerScale(embed_dim, init_values=args.layer_scale) if args.layer_scale is not None else nn.Identity()

        self.out_norm = norm_layer(embed_dim) if args.use_out_norm else nn.Identity()
        self.out_proj = nn.Linear(embed_dim, out_features)
        self.init_weights()

    def init_weights(self):
        if self.pos_embed is not None:
            trunc_normal_tf_(self.pos_embed, std=self.pos_embed.shape[1] ** -0.5)
        if self.prefix_type == 'learnable':
            trunc_normal_tf_(self.latent, std=self.latent_dim ** -0.5)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        B, N, C = x.shape
        if self.prefix_type == 'learnable':
            num_prefix_tokens = self.latent_len
            q_latent = self.latent.expand(B, -1, -1)
        elif self.prefix_type == 'avg':
            num_prefix_tokens = 1
            q_latent = x.mean(1, keepdim=True)
        elif self.prefix_type == 'avg_patch':
            num_prefix_tokens = 1
            q_latent = x[:,-self.seq_len:].mean(1, keepdim=True)
        elif self.prefix_type == 'original':
            num_prefix_tokens = N - self.seq_len
            q_latent = x[:,:num_prefix_tokens]
        else:
            raise ValueError(f"Invalid prefix type: {self.prefix_type}")
        
        if not self.args.use_all_tokens_for_kv:
            x = x[:,-self.seq_len:]
            
        num_seq = self.seq_len
        if self.pos_embed is not None:
            # FIXME interpolate
            x = x + self.pos_embed.unsqueeze(0).to(x.dtype)
    
        q = self.q(q_latent).reshape(B, num_prefix_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        kv = self.kv(x).reshape(B, num_seq, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        q, k = self.q_norm(q), self.k_norm(k)

        attn = None
        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = maybe_add_mask(attn, attn_mask)
            attn = attn.softmax(dim=-1)
            x = attn @ v
            
        x = x.transpose(1, 2).reshape(B, num_prefix_tokens, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x + self.layer_scale(self.mlp(self.norm(x)))
        x = self.out_norm(x)
        x = self.out_proj(x)
        # optional pool if latent seq_len > 1 and pooled output is desired
        if self.pool == 'token':
            x = x[:, 0]
        elif self.pool == 'avg':
            x = x.mean(1)
        return x, attn
    
class DINOTXThead(nn.Module):
    # code from https://github.com/facebookresearch/dinov2/blob/main/dinov2/hub/text/vision_tower.py#L45    
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        blocks_drop_path: float,
        use_class_token: bool,
        use_patch_tokens: bool,
        use_linear_projection: bool,
        feat_size: int,
        pool_type: str,
        args=None
    ):
        super().__init__()
        block_list = [nn.Identity()]
        self.ln_final = nn.Identity()
        if num_blocks > 0:
            block_list = [
                AttentionBlock(
                    input_dim,
                    num_heads,
                    ffn_layer=partial(SwiGLUFFN, align_to=64),
                    init_values=1e-5,
                    drop_path=blocks_drop_path,
                    args=args,
                )
                for _ in range(num_blocks)
            ]
            self.ln_final = nn.LayerNorm(input_dim)
        self.block_list = nn.ModuleList(block_list)
        self.num_blocks = num_blocks
        multiplier = 2 if use_class_token and use_patch_tokens else 1
        self.linear_projection = nn.Identity()
        if multiplier * input_dim != embed_dim or use_linear_projection:
            assert embed_dim % multiplier == 0, f"Expects {embed_dim} to be divisible by {multiplier}"
            self.linear_projection = nn.Linear(input_dim, embed_dim // multiplier, bias=False)
        
        # ours
        self.prefix_type = args.prefix_type
        if self.prefix_type == 'learnable':
            self.cls_token = nn.Parameter(torch.zeros(1,1, embed_dim))
            
        self.feat_size = feat_size
        self.seq_len = self.feat_size * self.feat_size
        self.pool_type = pool_type
        self.args = args
        
        print(f"Number of parameters in AttentionPool: {sum(p.numel() for p in self.parameters())}")
        
    def init_weights(self):
        if self.num_blocks > 0:
            for i in range(self.num_blocks):
                block = self.block_list[i]
                named_apply(init_weights_vit_timm, block)
            self.ln_final.reset_parameters()
        if isinstance(self.linear_projection, nn.Linear):
            nn.init.normal_(self.linear_projection.weight, std=self.linear_projection.in_features**-0.5)

    def forward(self, image_tokens: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        B,N,C = image_tokens.shape
        patch_tokens = image_tokens[:,-self.seq_len:]
        if self.prefix_type == 'learnable':
            q_latent = self.cls_token.expand(B, -1, -1)
        elif self.prefix_type == 'avg':
            q_latent = image_tokens.mean(1, keepdim=True)
        elif self.prefix_type == 'avg_patch':
            q_latent = patch_tokens.mean(1, keepdim=True)
        elif self.prefix_type == 'original':
            q_latent = image_tokens[:,:N-self.seq_len]
        else:
            raise ValueError(f"Invalid prefix type: {self.prefix_type}")
        image_tokens = torch.cat([q_latent, patch_tokens], dim=1)
        
        attn_weights = None
        for block in self.block_list:
            image_tokens, attn_weights = block(image_tokens, attn_mask=attn_mask)
        image_tokens = self.ln_final(image_tokens)
        if self.pool_type == 'token':
            pool_tokens = image_tokens[:, 0]
        elif self.pool_type == 'avg':
            pool_tokens = image_tokens.mean(1)
        else:
            raise ValueError(f"Invalid pool type: {self.pool_type}")
        return self.linear_projection(pool_tokens), attn_weights
