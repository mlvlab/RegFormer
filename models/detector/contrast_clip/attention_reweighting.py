import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import global_pool_nlc
from timm.layers.attention import maybe_add_mask
from torchvision.ops import roi_align

NUM_PER_CHUNK = 60


class AttentionReweightingModule(nn.Module):
    def __init__(self, model, preprocess, args):
        super().__init__()

        self.args = args
        self.model = model
        self.preprocess = preprocess
        self.image_level_pooling = args.image_level_pooling
        self.use_weak_model = args.use_weak_model
        if self.image_level_pooling:
            assert self.args.input_type == 'full'
        if self.use_weak_model:
            self.image_level_pooling = args.image_level_pooling
            self.feat_size = self.model.feat_size
        else:
            self.attention_reweighting_start_layer = args.attention_reweighting_start_layer if args.attention_reweighting_start_layer != -1 else len(self.model.trunk.blocks)
            self.attention_reweighting_end_layer = args.attention_reweighting_end_layer if args.attention_reweighting_end_layer != -1 else len(self.model.trunk.blocks) - 1
            self.model_type = args.open_clip_model_name
            if 'SigLIP' in self.model_type:
                self.attn_pool_mask = args.attn_pool_mask
                self.feat_size = int(np.sqrt(self.model.trunk.patch_embed.num_patches))

            if self.args.input_type == 'full' and not self.image_level_pooling:
                assert self.args.roi_align_layer <= self.attention_reweighting_start_layer, "roi align must be done before attention reweighting"
                assert self.args.roi_align_layer <= len(self.model.trunk.blocks), "roi align layer must be less than or equal to the number of blocks"

    def feat_roialign(self, x, meta_data):
        spatial_scale = self.feat_size / meta_data[0]['input_size'][0]
        resized_x = x.permute(0, 2, 1).reshape(-1, x.shape[-1], self.feat_size, self.feat_size)
        resized_x = roi_align(
            resized_x,
            [meta_data[0]['union_boxes']],
            (self.feat_size, self.feat_size),
            spatial_scale=spatial_scale,
            aligned=True,
        )
        x = resized_x.flatten(2).permute(0, 2, 1)
        return x

    @staticmethod
    def forward_attn_pool(ctx, x, attn_mask=None, return_attention_weight=False):
        B, N, C = x.shape

        if ctx.pos_embed is not None:
            x = x + ctx.pos_embed.unsqueeze(0).to(x.dtype)

        q_latent = ctx.latent.expand(B, -1, -1)
        q = ctx.q(q_latent).reshape(B, ctx.latent_len, ctx.num_heads, ctx.head_dim).transpose(1, 2)

        kv = ctx.kv(x).reshape(B, N, 2, ctx.num_heads, ctx.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        q, k = ctx.q_norm(q), ctx.k_norm(k)

        if ctx.fused_attn and not return_attention_weight:
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            attn = None
        else:
            q = q * ctx.scale
            attn = q @ k.transpose(-2, -1)
            attn = maybe_add_mask(attn, attn_mask)
            attn = attn.softmax(dim=-1)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, ctx.latent_len, C)
        x = ctx.proj(x)
        x = ctx.proj_drop(x)
        x = x + ctx.mlp(ctx.norm(x))

        if ctx.pool == 'token':
            x = x[:, 0]
        elif ctx.pool == 'avg':
            x = x.mean(1)
        return x, attn

    def forward_timm(self, x, attn_mask=None, meta_data=None):
        if self.args.debug_attn:
            pixel_mean = torch.tensor(self.preprocess.transforms[-1].mean, device=x.device).view(1, 3, 1, 1)
            pixel_std = torch.tensor(self.preprocess.transforms[-1].std, device=x.device).view(1, 3, 1, 1)
            orig_image = x * pixel_std + pixel_mean
            orig_image = orig_image.permute(0, 2, 3, 1).cpu().numpy()
            orig_image = (orig_image * 255).astype(np.uint8).clip(0, 255)

        trunk = self.model.trunk
        head = self.model.head
        repeat_count = (
            len(attn_mask)
            if attn_mask is not None
            else len(meta_data[0]['union_boxes']) if meta_data is not None else x.shape[0]
        )

        x = trunk.patch_embed(x)
        x = trunk._pos_embed(x)
        x = trunk.patch_drop(x)
        x = trunk.norm_pre(x)

        for i, blk in enumerate(trunk.blocks):
            if not self.image_level_pooling and self.args.input_type == 'full' and i == self.args.roi_align_layer:
                x = self.feat_roialign(x, meta_data)

            if i >= self.attention_reweighting_start_layer and i <= self.attention_reweighting_end_layer:
                if self.image_level_pooling and i == self.attention_reweighting_start_layer:
                    x = x.repeat(repeat_count, 1, 1)
                x = blk(x, attn_mask=attn_mask)
            else:
                x = blk(x)

        if self.image_level_pooling:
            if len(trunk.blocks) == self.attention_reweighting_start_layer:
                x = x.repeat(repeat_count, 1, 1)
        else:
            if self.args.input_type == 'full' and self.args.roi_align_layer == len(trunk.blocks):
                x = self.feat_roialign(x, meta_data)

        x = trunk.norm(x)
        if trunk.attn_pool is not None:
            if not trunk.pool_include_prefix:
                x = x[:, trunk.num_prefix_tokens:]

            if self.args.custom_attn_pool:
                x, attn = self.forward_attn_pool(
                    trunk.attn_pool,
                    x,
                    attn_mask=attn_mask if self.attn_pool_mask else None,
                    return_attention_weight=self.args.return_attention_weight,
                )
                if self.args.debug_attn and attn is not None:
                    x = x.repeat(repeat_count, 1)
                    averaged_attn = attn.mean(dim=1).squeeze(1).reshape(-1, self.feat_size, self.feat_size)
                    import matplotlib.pyplot as plt
                    import os
                    from skimage.transform import resize

                    vis_dir = os.path.join(self.args.output_dir, "attention_visualizations")
                    os.makedirs(vis_dir, exist_ok=True)
                    for n in range(averaged_attn.shape[0]):
                        attn_map = averaged_attn[n].cpu().numpy()
                        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
                        orig_img = orig_image[0]
                        attn_resized = resize(attn_map, (orig_img.shape[0], orig_img.shape[1]), preserve_range=True)
                        axes[0].imshow(orig_img)
                        axes[0].set_title(f'Original Image - Sample {n}')
                        axes[0].axis('off')
                        im1 = axes[1].imshow(attn_resized, cmap='viridis', interpolation='nearest')
                        axes[1].set_title(f'Attention Map - Sample {n}')
                        axes[1].axis('off')
                        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
                        axes[2].imshow(orig_img)
                        im2 = axes[2].imshow(attn_resized, cmap='viridis', alpha=0.5, interpolation='nearest')
                        axes[2].set_title(f'Image + Attention Overlay - Sample {n}')
                        axes[2].axis('off')
                        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
                        save_path = os.path.join(vis_dir, f'attention_comparison_sample_{n}.png')
                        plt.tight_layout()
                        plt.savefig(save_path, dpi=150, bbox_inches='tight')
                        plt.close()
                    print(f"Attention visualizations saved to {vis_dir}")
            else:
                x = trunk.attn_pool(x, attn_mask=attn_mask if self.attn_pool_mask else None)
        else:
            x = global_pool_nlc(
                x,
                pool_type=trunk.global_pool,
                num_prefix_tokens=trunk.num_prefix_tokens,
                reduce_include_prefix=trunk.pool_include_prefix,
            )
        x = trunk.fc_norm(x)
        x = trunk.head_drop(x)
        x = trunk.head(x)
        x = head(x)
        return x

    def _prepare_meta_data(self, x, targets):
        input_size = x.shape[-2:]
        tgt_lists = []
        for tgt in targets:
            new_dict = {}
            h, w = tgt['orig_size']
            scale_factor = torch.tensor([input_size[1] / w, input_size[0] / h, input_size[1] / w, input_size[0] / h], device=x.device)
            new_dict['union_boxes'] = torch.tensor(tgt['crop_union'], device=x.device) * scale_factor
            new_dict['human_boxes'] = torch.tensor(tgt['crop_human'], device=x.device) * scale_factor
            new_dict['object_boxes'] = torch.tensor(tgt['crop_object'], device=x.device) * scale_factor
            new_dict['input_size'] = input_size
            new_dict['filename'] = tgt['filename']
            tgt_lists.append(new_dict)

        return tgt_lists

    def forward(self, x, targets=None):
        if targets is not None and not self.args.force_no_attention_modulation:
            raise NotImplementedError("Attention modulation masks were removed; run with --force_no_attention_modulation.")
        attn_mask = None

        res = []
        if hasattr(self.model, 'args') and hasattr(self.model.args, 'attention_type') and self.model.args.attention_type == 'ml_decoder':
            keep_labels = targets[0]['keep_labels']
            keep_object_indices = targets[0]['keep_object_idx']
            so_indices = keep_labels[keep_object_indices]
            if 'contrastive' in self.args.trainin_free_option:
                so_indices = np.concatenate([so_indices, so_indices], axis=0)
            so_indices = torch.tensor(so_indices, device=x.device, dtype=torch.long)
        else:
            so_indices = None

        if self.args.input_type == 'union':
            if self.use_weak_model:
                if self.model.args.attention_type == 'ml_decoder':
                    self.model.attention_pooling.image_level_pooling = False
            chunked_input = torch.tensor_split(x, x.shape[0] // NUM_PER_CHUNK + 1)
            chunked_attn_mask = [None] * len(chunked_input)
            if so_indices is not None:
                chunked_so_indices = torch.tensor_split(so_indices, so_indices.shape[0] // NUM_PER_CHUNK + 1)
            else:
                chunked_so_indices = [None] * len(chunked_input)

            for chunk, att_mask, so_indices in zip(chunked_input, chunked_attn_mask, chunked_so_indices):
                if self.use_weak_model:
                    if self.model.args.attention_type == 'ml_decoder':
                        output = self.model(chunk, None, attn_mask=att_mask, so_indices=so_indices)
                    else:
                        output = self.model.encode_vision(chunk, attn_mask=att_mask, so_indices=so_indices)
                else:
                    if 'SigLIP' in self.model_type:
                        output = self.forward_timm(chunk, attn_mask=att_mask)

                res.append(output)

            if self.use_weak_model and self.model.args.attention_type == 'ml_decoder':
                if self.model.args.ml_decoder_query_type != 'triplet':
                    res = [torch.cat(res, dim=1)]
                else:
                    res = [torch.cat(res, dim=0)]

        elif self.args.input_type == 'full':
            meta_data = self._prepare_meta_data(x, targets)
            if self.use_weak_model:
                if self.model.args.attention_type == 'ml_decoder':
                    if 'contrastive' in self.args.trainin_free_option:
                        for i in range(len(meta_data)):
                            for k, v in meta_data[i].items():
                                if k in ['union_boxes', 'human_boxes', 'object_boxes']:
                                    meta_data[i][k] = torch.cat([v, v], dim=0)
                    output = self.model(x.unsqueeze(0), None, attn_mask=attn_mask, so_indices=so_indices, meta_data=meta_data)
                else:
                    output = self.model.encode_vision(x.unsqueeze(0), attn_mask=attn_mask)
            else:
                if 'SigLIP' in self.model_type:
                    output = self.forward_timm(x.unsqueeze(0), attn_mask=attn_mask, meta_data=meta_data)

            res.append(output)
        else:
            raise ValueError(f"Invalid input type: {self.args.input_type}")
        return torch.cat(res, dim=0)
