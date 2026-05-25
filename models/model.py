import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch 
from line_profiler import profile
import open_clip

from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor, CLIPTextModelWithProjection, AutoTokenizer, SiglipTextModel
import numpy as np
import math
import warnings
from models.model_utils import get_clip_vision_hidden_dim, text_global_pool
from models.attn_pool import (
    SelfAttentionPool, CrossAttentionPool, DINOTXThead, CrossAttentionPool_v2,
    CrossAttentionPool_ml_decoder
)
from models.attention import TransformerEncoder
from models.position_embedding import PositionEmbeddingSine
from models.layers.swin import SwinTransformer
from utils.label_utils import get_class_labels

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

class Projector(nn.Module):
    """Projector module for embedding projection"""
    
    def __init__(self, input_dim, output_dim, projector_type='identity', hidden_dim=512, skip_connection=False, feat_size=None, args=None):
        super().__init__()
        self.projector_type = projector_type
        self.skip_connection = skip_connection
        if projector_type == 'identity':
            self.projection = nn.Identity()
        elif projector_type == 'only_linear':
            self.projection = nn.Linear(input_dim, output_dim)
            nn.init.xavier_uniform_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)
        elif projector_type == 'lora':
            self.projection = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim)
            )
        elif projector_type == 'ln_linear':
            self.projection = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, output_dim)
            )
        elif projector_type == 'linear':
            self.projection = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim)
            )
        elif projector_type == 'mlp':
            self.projection = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim)
            )
        elif projector_type == 'mlp_ln':
            self.projection = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )
        elif projector_type == 'attention':
            self.projection = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
            self.attention = TransformerEncoder(d_model=output_dim, 
                                                nhead=args.attention_pool_heads, 
                                                mlp_ratio=4.0, 
                                                num_layers=args.vision_attention_layers, 
                                                args=args)
            self.pe = PositionEmbeddingSine(num_pos_feats=output_dim//2, temperature=20, normalize=True)
            # self.attention = AttentionBlock()
        elif projector_type == 'swin':
            self.projection = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.GELU(),
                nn.Linear(output_dim, output_dim)
            )
            self.layers = SwinTransformer(dim=output_dim, num_layers=args.vision_attention_layers, num_heads=args.attention_pool_heads)
            self.feat_size = feat_size
        else:
            raise ValueError(f"Unsupported projector type: {projector_type}")

        if self.skip_connection:
            assert input_dim == output_dim, "Input and output dimension must be the same for skip connection"
            # Zero initialize the last layer if projector is not identity
            if projector_type != 'identity':
                if projector_type in ['mlp','lora']:
                    # For linear projector, zero init the Linear layer
                    nn.init.zeros_(self.projection[-1].weight)
                    nn.init.zeros_(self.projection[-1].bias)
                    
    def forward(self, x):
        if self.skip_connection:
            return self.projection(x) + x
        else:
            if 'attention' in self.projector_type:
                x = self.projection(x)
                x = x.transpose(0,1)
                
                # generate sinusoidal pe
                # TODO: now only support pe for square images. later support for non-square images, irregular image shapes within batch.
                patch_length = int(np.sqrt(x.shape[0]))
                dummy_mask = torch.zeros((x.shape[1], patch_length, patch_length),device=x.device, dtype=torch.bool)
                pe = self.pe(dummy_mask)
                pe = pe.flatten(2).permute(2,0,1)
                if pe.shape[0] < x.shape[0]:
                    pad_pe = torch.zeros((x.shape[0]-pe.shape[0], pe.shape[1], pe.shape[2]),device=x.device, dtype=pe.dtype)
                    pe = torch.cat([pad_pe, pe], dim=0)
                x = self.attention(x, pos=pe)
                return x.transpose(0,1)
            elif 'swin' in self.projector_type:
                x = self.projection(x[:,-self.feat_size*self.feat_size:])
                B,N,C = x.shape
                x = x.reshape(B, self.feat_size, self.feat_size, C)
                x = self.layers(x)
                return x.flatten(1,2)
            else:
                return self.projection(x)


class VisionTextAlignmentModel(nn.Module):
    """Vision-Text Alignment Model for HOI detection"""
    
    def __init__(self, verb_object_indices, args):
        super().__init__()
        self.verb_object_indices = verb_object_indices
        self.args = args
        
        # Initialize vision encoder
        if '__pretrained__' in args.vision_encoder.lower():
            # use open_clip model
            # import open_clip
            self.use_vision_open_clip = True
            vision_model_name, pretrained = args.vision_encoder.split('__pretrained__')
            clip_model, _, self.vision_processor = open_clip.create_model_and_transforms(vision_model_name, pretrained=pretrained)
            self.vision_model = clip_model.visual
            self.vision_model.config = open_clip.get_model_config(vision_model_name)
        else:
            self.use_vision_open_clip = False
            if 'clip' in args.vision_encoder.lower():
                self.vision_model = CLIPModel.from_pretrained(args.vision_encoder).vision_model
                self.vision_processor = CLIPProcessor.from_pretrained(args.vision_encoder)
            else:
                self.vision_model = AutoModel.from_pretrained(args.vision_encoder)
                self.vision_processor = AutoProcessor.from_pretrained(args.vision_encoder)
                if 'siglip' in args.vision_encoder.lower():
                    self.vision_model = self.vision_model.vision_model

        if args.vision_ft_method is not None:
            if args.vision_ft_method == 'lora':
                from peft import LoraConfig, get_peft_model
                if 'siglip' in args.vision_encoder.lower():
                    # target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
                    if args.ft_start_layer is not None:
                        raise ValueError("FT start layer is not supported for siglip")
                    else:
                        target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
                elif 'clip' in args.vision_encoder.lower():
                    # target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
                    if args.ft_start_layer is not None:
                        raise ValueError("FT start layer is not supported for clip")
                    else:
                        target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
                elif 'dinov2' in args.vision_encoder.lower():
                    num_layers = self.vision_model.config.num_hidden_layers
                    ft_start_layer = args.ft_start_layer if args.ft_start_layer is not None else 0
                    ft_end_layer = args.ft_end_layer if args.ft_end_layer is not None else num_layers - 1
                    
                    target_modules = []
                    
                    base_name = "encoder.layer"
                    for layer_idx in range(ft_start_layer, ft_end_layer+1):
                        target_modules.append(f"{base_name}.{layer_idx}.attention.attention.query")
                        target_modules.append(f"{base_name}.{layer_idx}.attention.attention.key")
                        target_modules.append(f"{base_name}.{layer_idx}.attention.attention.value")
                        target_modules.append(f"{base_name}.{layer_idx}.attention.output.dense")
                    
                else:
                    raise ValueError(f"Unsupported vision encoder: {args.vision_encoder}")
                
                peft_config = LoraConfig(
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    target_modules=target_modules,
                    bias="none"
                )
                self.vision_model = get_peft_model(self.vision_model, peft_config)
                print("training LoRA")
                print(f"target_modules: {target_modules}")
                print(f"r: {args.lora_r}, alpha: {args.lora_alpha}, dropout: {args.lora_dropout}")
                print(self.vision_model.print_trainable_parameters())
        else:
            # Freeze encoders if specified
            if args.freeze_vision_encoder:
                for param in self.vision_model.parameters():
                    param.requires_grad = False
            else:
                for name, param in self.vision_model.named_parameters():
                    if 'post_layernorm' in name:
                        param.requires_grad = False
                    if 'attnpool' in name:
                        param.requires_grad = False
                        
        if self.use_vision_open_clip:
            pt_image_size = self.vision_model.config['vision_cfg']['image_size']
            vision_dim = get_clip_vision_hidden_dim(args.vision_encoder.split('__pretrained__')[0])
        else:
            pt_image_size = getattr(self.vision_model.config, 'image_size', 224)
            try:
                vision_dim = self.vision_model.config.hidden_size
            except:
                vision_dim = get_clip_vision_hidden_dim(args.vision_encoder)
        print(f"Vision model pretrained image size: {pt_image_size}, fine-tuning image size: {args.input_resolution}")
        if pt_image_size != args.input_resolution:
            print(f"Need to interpolate positional embeddings")
            print(f"If vision models is frozen, position embedding interpolation will affect the performance")
            self.interpolate_positional_embeddings = True
        else:
            self.interpolate_positional_embeddings = False
                    
        self.class_embeddings = None
        self.seperate_normalization = args.seperate_normalization
        self.use_seperate_classifier = args.use_seperate_classifier
        if args.use_text_embeddings:
            assert args.freeze_text_encoder, "Freezing text encoder is required when using pre-extracted text embeddings"
            if  args.attention_type == 'ml_decoder' and args.ml_decoder_query_type == 'object':
                print(f"Using object embeddings for ML decoder")
                warnings.warn(
                    "When using object query type for ML decoder, "
                    "Input texts are not adequately handled for online text feature extraction. "
                    "This will need to be fixed, such as passing the text form of verb and object to the model.",
                    UserWarning
                )
                verb_list, object_list = verb_object_indices
                verb_embed_dir = os.path.join('embeddings', args.dataset_name, args.text_encoder, 'verb.pt')
                object_embed_dir = os.path.join('embeddings', args.dataset_name, args.text_encoder, f'{args.ml_decoder_query_ckpt}.pt')
                verb_embeddings = torch.load(verb_embed_dir, map_location='cpu')['embeddings']
                object_embeddings = torch.load(object_embed_dir, map_location='cpu')['embeddings']
                
                self.object_embeddings = object_embeddings
                self.class_embeddings = verb_embeddings
                text_dim = self.class_embeddings.shape[1]
                # obj_text_dim = self.object_embeddings.shape[1]
                
            else:
                if self.use_seperate_classifier:
                    print(f"Using seperate classifier for HOI classes")
                    verb_list, object_list = verb_object_indices
                    verb_embed_dir = os.path.join('embeddings', args.dataset_name, args.text_encoder, 'verb.pt')
                    object_embed_dir = os.path.join('embeddings', args.dataset_name, args.text_encoder, f'{args.ml_decoder_query_ckpt}.pt')
                    verb_embeddings = torch.load(verb_embed_dir, map_location='cpu')['embeddings']
                    object_embeddings = torch.load(object_embed_dir, map_location='cpu')['embeddings']
                    verb_to_hoi_embeddings = verb_embeddings[verb_list]
                    object_to_hoi_embeddings = object_embeddings[object_list]
                    self.class_embeddings = torch.cat([verb_to_hoi_embeddings, object_to_hoi_embeddings], dim=-1)
                else:
                    embed_dir = os.path.join('embeddings', args.dataset_name, args.text_encoder, f'{args.target_type}.pt')
                    print(f"Using text embeddings from {embed_dir}")
                    class_infos = torch.load(embed_dir, map_location='cpu')
                    self.class_embeddings = class_infos['embeddings']
            # classes = class_infos['labels']
            text_dim = self.class_embeddings.shape[1]
        else:
            # Initialize text encoder
            if '__pretrained__' in args.text_encoder.lower():
                # use open_clip model
                self.use_text_open_clip = True
                text_model_name, pretrained = args.text_encoder.split('__pretrained__')
                clip_model, _, _ = open_clip.create_model_and_transforms(text_model_name, pretrained=pretrained, precision='fp16')
                self.text_model = clip_model.transformer
                self.text_processor = open_clip.get_tokenizer(text_model_name)
                self.text_model.config = open_clip.get_model_config(text_model_name)
                
                # self.text_model = self.text_model.half()
                self.text_model.eval()
                self.text_model.cuda()
                
                self.clip_token_embedding = clip_model.token_embedding.to(next(self.text_model.parameters()).device)
                self.clip_positional_embedding = clip_model.positional_embedding.to(next(self.text_model.parameters()).device)
                self.clip_ln_final = clip_model.ln_final.to(next(self.text_model.parameters()).device)
                self.clip_text_projection = clip_model.text_projection.to(next(self.text_model.parameters()).device)
                self.clip_text_pool_type = clip_model.text_pool_type
                self.clip_text_eos_id = getattr(clip_model, "text_eos_id", None)
                self.clip_attn_mask = clip_model.attn_mask.to(next(self.text_model.parameters()).device)
                
                # Freeze all CLIP components to prevent gradient flow
                # self.clip_positional_embedding.requires_grad = False
                for param in self.clip_token_embedding.parameters():
                    param.requires_grad = False
                for param in self.clip_ln_final.parameters():
                    param.requires_grad = False
                if hasattr(self.clip_text_projection, 'parameters'):
                    for param in self.clip_text_projection.parameters():
                        param.requires_grad = False
                
            else:
                self.use_text_open_clip = False
                if 'clip' in args.text_encoder.lower():
                    self.text_model = CLIPTextModelWithProjection.from_pretrained(args.text_encoder)
                    self.text_processor = AutoTokenizer.from_pretrained(args.text_encoder)
                elif 'siglip' in args.text_encoder.lower():
                    self.text_model = SiglipTextModel.from_pretrained(args.text_encoder)
                    self.text_processor = AutoTokenizer.from_pretrained(args.text_encoder)
                else:
                    self.text_model = AutoModel.from_pretrained(args.text_encoder, trust_remote_code=True)
                    self.text_processor = AutoProcessor.from_pretrained(args.text_encoder)
                if 'bert' not in args.text_encoder.lower():
                    self.text_model = self.text_model.half()
                self.text_model.eval()
                self.text_model.cuda()
            
            
            # text_dim = self.text_model.config.hidden_size
            if self.use_text_open_clip:
                text_dim = self.text_model.config['embed_dim']
            else:
                try:
                    text_dim = self.text_model.config.projection_dim
                except:
                    text_dim = self.text_model.config.hidden_size
            
            if args.attention_type == 'ml_decoder':
                if args.ml_decoder_query_type == 'object':
                    # we use pre-extracted object embeddings for ml decoder
                    # raise NotImplementedError("TODO")
                    class_labels, _ = get_class_labels(args.dataset_name, args.ml_decoder_query_ckpt)
                    with torch.no_grad():
                        object_embeddings = self.encode_text(class_labels, precompute=True)
                    
                    # object_embed_dir = os.path.join('embeddings', args.dataset_name, args.text_encoder, f'{args.ml_decoder_query_ckpt}.pt')
                    # object_embeddings = torch.load(object_embed_dir, map_location='cpu')['embeddings']
                    self.object_embeddings = object_embeddings
                    from hico_list import hico_verbs_sentence
                    from vcoco_list import vcoco_verbs_sentence
                    if args.dataset_name == 'hico':
                        self.verb_texts = hico_verbs_sentence
                    elif args.dataset_name == 'vcoco':
                        self.verb_texts = vcoco_verbs_sentence
                    elif args.dataset_name == 'swig':
                        self.verb_texts = get_class_labels(args.dataset_name, 'verb')[0]
                    else:
                        raise ValueError(f"Unsupported dataset: {args.dataset_name}")
                elif args.ml_decoder_query_type == 'triplet':
                    self.verb_texts = get_class_labels(args.dataset_name, 'hoi')[0]
                else:
                    raise ValueError(f"Unsupported ML decoder query type: {args.ml_decoder_query_type}")
            
                # for i in range(len(self.verb_texts)):
                #     self.verb_texts[i] = self.verb_texts[i].replace("a photo of a ", "")
            
            # Apply PEFT to text encoder if specified
            if args.text_ft_method is not None:
                    
                if args.text_ft_method == 'lora':
                    from peft import LoraConfig, get_peft_model
                    text_dim = self.text_model.config.hidden_size if '__pretrained__' not in args.text_encoder.lower() else self.text_model.config['embed_dim']
                    if 'clip' in args.text_encoder.lower() or 'siglip' in args.text_encoder.lower():
                        num_layers = self.text_model.config.num_hidden_layers
                        ft_start_layer = args.text_ft_start_layer if args.text_ft_start_layer is not None else 0
                        ft_end_layer = args.text_ft_end_layer if args.text_ft_end_layer is not None else num_layers - 1
                        
                        target_modules = []
                        
                        base_name = "encoder.layers"
                        for layer_idx in range(ft_start_layer, ft_end_layer+1):
                            target_modules.append(f"{base_name}.{layer_idx}.self_attn.query")
                            target_modules.append(f"{base_name}.{layer_idx}.self_attn.key")
                            target_modules.append(f"{base_name}.{layer_idx}.self_attn.value")
                            target_modules.append(f"{base_name}.{layer_idx}.self_attn.out_proj")
                    elif 'roberta' in args.text_encoder.lower():
                        base_name = "encoder.layer"
                        num_layers = self.text_model.config.num_hidden_layers if hasattr(self.text_model.config, 'num_hidden_layers') else self.text_model.config.num_layers
                        ft_start_layer = args.text_ft_start_layer if args.text_ft_start_layer is not None else 0
                        ft_end_layer = args.text_ft_end_layer if args.text_ft_end_layer is not None else num_layers - 1
                        target_modules = []
                        for layer_idx in range(ft_start_layer, ft_end_layer+1):
                            target_modules.append(f"{base_name}.{layer_idx}.attention.self.query")
                            target_modules.append(f"{base_name}.{layer_idx}.attention.self.key")
                            target_modules.append(f"{base_name}.{layer_idx}.attention.self.value")
                            target_modules.append(f"{base_name}.{layer_idx}.attention.self.output.dense")
                    elif '__pretrained__' in args.text_encoder.lower():
                        base_name = "resblocks"
                        num_layers = self.text_model.config['text_cfg']['layers']
                        ft_start_layer = args.text_ft_start_layer if args.text_ft_start_layer is not None else 0
                        ft_end_layer = args.text_ft_end_layer if args.text_ft_end_layer is not None else num_layers - 1
                        target_modules = []
                        for layer_idx in range(ft_start_layer, ft_end_layer+1):
                            target_modules.append(f"{base_name}.{layer_idx}.attn")
                            # target_modules.append(f"{base_name}.{layer_idx}")
                            # target_modules.append(f"{base_name}.{layer_idx}")
                            # target_modules.append(f"{base_name}.{layer_idx}")
                    else:
                        raise ValueError(f"Unsupported text encoder for LoRA: {args.text_encoder}")
                    
                    peft_config = LoraConfig(
                        r=args.text_lora_r,
                        lora_alpha=args.text_lora_alpha,
                        lora_dropout=args.text_lora_dropout,
                        target_modules=target_modules,
                        bias="none"
                    )
                    self.text_model = get_peft_model(self.text_model, peft_config)
                    print("Training text encoder with LoRA")
                    print(f"target_modules: {target_modules}")
                    print(f"r: {args.text_lora_r}, alpha: {args.text_lora_alpha}, dropout: {args.text_lora_dropout}")
                    print(self.text_model.print_trainable_parameters())
                    
                elif args.text_ft_method == 'prompt_tuning':
                    # TODO: currently bug in forward pass... need to fix
                    from peft import PromptTuningConfig, PromptTuningInit, get_peft_model, TaskType, PromptEncoder
                    
                    peft_config = PromptTuningConfig(
                        task_type=TaskType.FEATURE_EXTRACTION,
                        prompt_tuning_init=PromptTuningInit.RANDOM,
                        num_virtual_tokens=args.text_num_prompt_tokens,
                        tokenizer_name_or_path=args.text_encoder,
                    )
                    # if 'siglip' in args.text_encoder.lower() or 'clip' in args.text_encoder.lower():
                    #     self.text_model.device = next(self.text_model.parameters()).device
                    self.text_model = get_peft_model(self.text_model, peft_config)
                    print("Training text encoder with Prompt Tuning")
                    print(f"num_virtual_tokens: {args.text_num_prompt_tokens}")
                    print(self.text_model.print_trainable_parameters())
                else:
                    raise ValueError(f"Unsupported text fine-tuning method: {args.text_ft_method}")
            else:
                # Freeze text encoder if specified and no PEFT method
                if args.freeze_text_encoder:
                    for param in self.text_model.parameters():
                        param.requires_grad = False
    
        # Initialize projectors with separate configurations
                
        # Vision projector configuration
        vision_projector_type = args.vision_projector_type if args.vision_projector_type is not None else args.projector_type
        vision_hidden_dim = args.vision_hidden_dim if args.vision_hidden_dim is not None else args.hidden_dim
        
        # Text projector configuration  
        text_projector_type = args.text_projector_type if args.text_projector_type is not None else args.projector_type
        text_hidden_dim = args.text_hidden_dim if args.text_hidden_dim is not None else args.hidden_dim
        
        if args.projection_dim == -1:
            args.projection_dim = text_dim
            
        self.text_projector = Projector(
            input_dim=text_dim,
            output_dim=args.projection_dim,
            projector_type=text_projector_type,
            hidden_dim=text_hidden_dim,
            skip_connection=args.text_skip_connection,
            # feat_size=self.feat_size
        )
        # Print text projector configuration for debugging
        print(f"Text projector: type={text_projector_type}, hidden_dim={text_hidden_dim}, input_dim={text_dim}, output_dim={args.projection_dim}")
        
        # Initialize attention pooling if specified and configure vision projector accordingly
        self.use_attention_pooling = getattr(args, 'use_attention_pooling', False)
        if self.use_attention_pooling:
            args.attention_pool_dim = vision_dim if vision_projector_type == 'identity' else getattr(args, 'attention_pool_dim', 512)
            args.attention_pool_heads = self.vision_model.config.num_attention_heads if vision_projector_type == 'identity' else getattr(args, 'attention_pool_heads', 8)
            pool_type = getattr(args, 'pool_type', 'token')
            
            # Auto-calculate feat_size from vision model config
            self.feat_size = self._get_vision_feat_size(args.input_resolution)
            
            self.use_det_results = False # set True for detection
            
            # Vision projector outputs to attention_pool_dim
            self.vision_projector = Projector(
                input_dim=vision_dim,
                output_dim=args.attention_pool_dim,  # Output to attention dim
                projector_type=vision_projector_type,
                hidden_dim=vision_hidden_dim,
                skip_connection=args.vision_skip_connection,
                feat_size=self.feat_size,
                args=args
            )
            
            # Use AttentionPool2d for attention pooling
            if args.attention_type == 'self':
                self.attention_pooling = SelfAttentionPool(
                    in_features=args.attention_pool_dim,   # Input from vision projector
                    feat_size=self.feat_size,
                    out_features=args.projection_dim, # Output to final projection dim
                    embed_dim=args.attention_pool_dim,
                    num_heads=args.attention_pool_heads,
                    pool_type=pool_type,
                    # class_token=True  # Use existing cls token from vision model
                    args=args
                )
            elif args.attention_type == 'cross':
                self.attention_pooling = CrossAttentionPool(
                    in_features=args.attention_pool_dim,   # Input from vision projector
                    out_features=args.projection_dim, # Output to final projection dim
                    embed_dim=args.attention_pool_dim,
                    num_heads=args.attention_pool_heads,
                    feat_size=self.feat_size,
                    pool_type=pool_type,
                    args=args
                )
            elif args.attention_type == 'cross_v2':
                self.attention_pooling = CrossAttentionPool_v2(
                    in_features=args.attention_pool_dim,
                    feat_size=self.feat_size,
                    out_features=args.projection_dim,
                    embed_dim=args.attention_pool_dim,
                    num_heads=args.attention_pool_heads,
                    args=args
                )
            elif args.attention_type == 'ml_decoder':
                self.attention_pooling = CrossAttentionPool_ml_decoder(
                    in_features=args.attention_pool_dim,
                    feat_size=self.feat_size,
                    out_features=args.projection_dim,
                    embed_dim=args.attention_pool_dim,
                    pos_embed=args.vis_pos_embed_type,
                    language_dim=self.object_embeddings.shape[1] if args.ml_decoder_query_type == 'object' else args.projection_dim,
                    num_heads=args.attention_pool_heads,
                    args=args
                )
                
            elif args.attention_type == 'dinotext':
                self.attention_pooling = DINOTXThead(
                    input_dim=args.attention_pool_dim,
                    embed_dim=args.projection_dim,
                    num_heads=args.attention_pool_heads,
                    num_blocks=1,
                    blocks_drop_path=args.blocks_drop_path,
                    use_class_token=True,
                    use_patch_tokens=False,
                    use_linear_projection=True,
                    feat_size=self.feat_size,
                    pool_type=pool_type,
                    args=args
                )
                
            if args.prefix_type == 'original' and 'siglip' in args.vision_encoder.lower():
                raise ValueError("Siglip does not support original prefix type since it does not have a prefix (e.g., CLS) token")
            
                
            print(f"Vision projector: {vision_dim} → {args.attention_pool_dim}")
            print(f"AttentionPool2d: {args.attention_pool_dim} → {args.projection_dim}, embed_dim={args.attention_pool_dim}, heads={args.attention_pool_heads}")
            print(f"Auto-calculated feat_size: {self.feat_size}, pool_type: {pool_type}")
        else:
            # Vision projector outputs directly to projection_dim
            self.vision_projector = Projector(
                input_dim=vision_dim,
                output_dim=args.projection_dim,  # Direct output to final dim
                projector_type=vision_projector_type,
                hidden_dim=vision_hidden_dim,
                args=args
            )
            self.attention_pooling = None
            print(f"Vision projector (no attention pooling): {vision_dim} → {args.projection_dim}")
        
        self.use_internal_projector = args.use_internal_projector
        if self.use_internal_projector:
            assert 'clip' in args.vision_encoder.lower() or 'clip' in args.text_encoder.lower(), "Internal projector is only supported for CLIP models"
            clip_model = CLIPModel.from_pretrained(args.vision_encoder or args.text_encoder)
            self.vision_projector = clip_model.visual_projection
            self.text_projector = clip_model.text_projection
            print(f"Using internal vision/text projector")
        
        self._custom_weight_initialization()
        
        # Scale parameters
        if not (args.attention_type == 'ml_decoder' and args.ml_decoder_query_type == 'triplet'):
            self.scale_logit = torch.nn.Parameter(torch.ones(1)*np.log(args.scale_logit))
            if not args.no_bias:
                self.scale_bias = torch.nn.Parameter(torch.ones(1)*args.scale_bias)
            else:
                self.scale_bias = 0.
        self.normalize_embeddings = args.normalize_embeddings

        # Print all learnable parameters
        print("All learnable parameters:")
        total_params = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"{name} : {param.numel()}")
                total_params += param.numel()
        print(f"Total learnable parameters: {total_params}")
        
        self.precomputed_text_embeddings = None
        
    def _custom_weight_initialization(self):
        """Custom weight initialization for the model"""
        if self.args.custom_weight_initialization:
            for name, m in self.named_modules():
                # leaf only: no children and at least one direct parameter with requires_grad=True
                if len(list(m.children())) == 0 and any(p.requires_grad for p in m.parameters(recurse=False)):
                    if isinstance(m, nn.Linear):
                        print(f'{name} is a Linear layer')
                        if m.weight.requires_grad:
                            print(f"Initializing {name} weight with xavier_uniform")
                            nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None and m.bias.requires_grad:
                            print(f"Initializing {name} bias with zeros")
                            nn.init.zeros_(m.bias)
                    elif isinstance(m, nn.LayerNorm):
                        print(f'{name} is a LayerNorm')
                        if m.weight.requires_grad:
                            print(f"Initializing {name} weight with ones")
                            nn.init.ones_(m.weight)
                        if m.bias is not None and m.bias.requires_grad:
                            print(f"Initializing {name} bias with zeros")
                            nn.init.zeros_(m.bias)
    
    def _get_vision_feat_size(self, input_resolution):
        """Calculate feature size from vision model configuration"""
        config = self.vision_model.config
        
        # Get image size and patch size from config        
        image_size = input_resolution 
        
        if hasattr(config, 'patch_size'):
            patch_size = config.patch_size
            feat_size = image_size // patch_size
        elif 'rn50' in self.args.vision_encoder.lower() or 'resnet' in self.args.vision_encoder.lower():
            feat_size = 7 * input_resolution //224
            patch_size = 224 // feat_size
        elif 'vision_cfg' in config and 'patch_size' in config['vision_cfg']:
            patch_size = config['vision_cfg']['patch_size']
            feat_size = image_size // patch_size
        else:
            # Default patch size for most models
            patch_size = 16
        
            # Calculate feature map size
            feat_size = image_size // patch_size
        print(f"Auto-calculated feat_size: {feat_size} (image_size: {image_size}, patch_size: {patch_size})")
        return feat_size

    def visualize_label_weight(self, images, instance_to_patch_scores, instance_scores, patch_to_instance_scores=None, score_threshold=None, visualize_class_index=None, smoothing=False):
        # input
        # images: [bs, 3, H, W]
        # instance_to_patch_scores: [bs, num_patches, num_class_labels]
        # instance_scores: [bs, num_class_labels]
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Denormalize images
        mean = self.vision_processor.image_mean if hasattr(self.vision_processor, 'image_mean') else self.vision_processor.image_processor.image_mean
        std = self.vision_processor.image_std if hasattr(self.vision_processor, 'image_std') else self.vision_processor.image_processor.image_std
        mean = torch.tensor(mean).view(1, 3, 1, 1).to(images.device)
        std = torch.tensor(std).view(1, 3, 1, 1).to(images.device)
        denorm_images = images * std + mean
        denorm_images = torch.clamp(denorm_images, 0, 1)
        
        bs, num_patches, num_class_labels = instance_to_patch_scores.shape
        
        # Reshape to spatial dimensions
        feat_size = int(np.sqrt(num_patches))
        assert feat_size * feat_size == num_patches, f"num_patches {num_patches} is not a perfect square"
        
        # Visualize each image in batch
        import os
        
        # Get max value across all classes for consistent scaling
        max_value = instance_to_patch_scores.max().item()
        
        from utils.hico_text_label import hico_obj_text_label, hico_text_label
        from hico_list import hico_verbs_sentence
        from vcoco_text_label import vcoco_obj_text_label, vcoco_hoi_text_label
        from vcoco_list import vcoco_verbs_sentence
        if self.args.dataset_name == 'hico':
            query_name = hico_obj_text_label
            class_name = hico_verbs_sentence
        elif self.args.dataset_name == 'vcoco':
            query_name = vcoco_obj_text_label
            class_name = vcoco_verbs_sentence
        else:
            raise ValueError(f"Unsupported dataset: {self.args.dataset_name}")
        
        # visualize specific class
        if visualize_class_index is not None and patch_to_instance_scores is not None:
            import matplotlib.pyplot as plt
            import os
            
            # Get scores for the specific class
            # patch_to_instance_scores: [bs, num_patches, num_class_labels]
            class_scores = patch_to_instance_scores[:, :, visualize_class_index]  # [bs, num_patches]
            
            for i in range(bs):
                # Create directory for this batch item
                batch_dir = f'label_weight_vis/batch_{i}'
                os.makedirs(batch_dir, exist_ok=True)
                
                # Get the original image for this batch
                if denorm_images.shape[0] != bs:
                    img = denorm_images[0].permute(1, 2, 0).detach().cpu().numpy()
                else:
                    img = denorm_images[i].permute(1, 2, 0).detach().cpu().numpy()
                
                # Get scores for this batch and reshape to 2D
                scores = class_scores[i].reshape(feat_size, feat_size).detach().cpu().numpy()
                
                # Upsample scores to match image size
                from scipy.ndimage import zoom
                h, w = img.shape[:2]
                zoom_factor_h = h / feat_size
                zoom_factor_w = w / feat_size
                scores_upsampled = zoom(scores, (zoom_factor_h, zoom_factor_w), order=1)
                
                # Normalize scores to [0, 1] for visualization
                scores_min = scores_upsampled.min()
                scores_max = scores_upsampled.max()
                if scores_max > scores_min:
                    scores_normalized = (scores_upsampled - scores_min) / (scores_max - scores_min)
                else:
                    scores_normalized = np.zeros_like(scores_upsampled)
                
                # Create overlay image
                overlay_img = img.copy()
                
                # Apply threshold if provided
                if score_threshold is not None:
                    # Create mask: True for scores above threshold
                    mask = scores_upsampled > score_threshold
                    
                    # For pixels below threshold, convert to grayscale (darker)
                    gray_img = np.mean(img, axis=2, keepdims=True)
                    gray_img = np.repeat(gray_img, 3, axis=2)
                    gray_img = gray_img * 0.5  # Make gray darker
                    
                    # Apply mask: keep original where score > threshold, gray otherwise
                    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
                    
                    # Blend based on normalized score (for areas above threshold)
                    # Higher score = more original image, lower score = more gray
                    alpha = scores_normalized[:, :, np.newaxis]
                    overlay_img = np.where(mask_3d, 
                                          img * alpha + gray_img * (1 - alpha),
                                          gray_img)
                else:
                    # No threshold: blend based on score only
                    # Higher score = more original image, lower score = more gray
                    gray_img = np.mean(img, axis=2, keepdims=True)
                    gray_img = np.repeat(gray_img, 3, axis=2)
                    gray_img = gray_img * 0.5  # Make gray darker
                    
                    alpha = scores_normalized[:, :, np.newaxis]
                    overlay_img = img * alpha + gray_img * (1 - alpha)
                
                # Create figure with original and overlay
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                
                # Original image
                axes[0].imshow(img)
                axes[0].set_title('Original Image')
                axes[0].axis('off')
                
                # Overlay image
                axes[1].imshow(np.clip(overlay_img, 0, 1))
                if score_threshold is not None:
                    axes[1].set_title(f'Class {visualize_class_index} Score Overlay (threshold={score_threshold:.3f})')
                else:
                    axes[1].set_title(f'Class {visualize_class_index} Score Overlay')
                axes[1].axis('off')
                
                plt.tight_layout()
                
                # Save figure
                save_path = os.path.join(batch_dir, f'class_{visualize_class_index}_score_overlay.png')
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"Saved class {visualize_class_index} visualization to {save_path}")
                
                # Also save score heatmap for reference
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                im = ax.imshow(scores, cmap='jet', interpolation='nearest')
                ax.set_title(f'Class {visualize_class_index} Score Heatmap')
                plt.colorbar(im, ax=ax)
                ax.axis('off')
                
                heatmap_path = os.path.join(batch_dir, f'class_{visualize_class_index}_score_heatmap.png')
                plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"Saved class {visualize_class_index} heatmap to {heatmap_path}")
        
        # Visualize patch_to_instance_scores if available
        if patch_to_instance_scores is not None:
            import random
            
            # Generate random colors for each class (excluding background)
            np.random.seed(42)  # For reproducible colors
            colors = []
            for _ in range(num_class_labels):
                colors.append([np.random.rand(), np.random.rand(), np.random.rand()])
            colors.append([1.0, 1.0, 1.0])  # White for background (last class)
            
            for i in range(bs):
                # Create directory for this batch item
                batch_dir = f'label_weight_vis/batch_{i}'
                os.makedirs(batch_dir, exist_ok=True)
                
                # Get the original image for this batch
                if denorm_images.shape[0] != bs:
                    img = denorm_images[0].permute(1, 2, 0).detach().cpu().numpy()
                else:
                    img = denorm_images[i].permute(1, 2, 0).detach().cpu().numpy()
                
                # Get predicted labels for each patch (argmax)
                patch_predictions = torch.argmax(patch_to_instance_scores[i], dim=-1)  # [num_patches]
                patch_predictions_2d = patch_predictions.reshape(feat_size, feat_size).detach().cpu().numpy()
                
                # Get max scores for each patch to apply threshold
                max_scores = torch.max(patch_to_instance_scores[i], dim=-1)[0]  # [num_patches]
                max_scores_2d = max_scores.reshape(feat_size, feat_size).detach().cpu().numpy()
                
                # Find unique classes actually present in the prediction (only above threshold)
                if score_threshold is not None:
                    # Only consider patches above threshold
                    valid_mask = max_scores_2d > score_threshold
                    unique_classes = np.unique(patch_predictions_2d[valid_mask])
                else:
                    unique_classes = np.unique(patch_predictions_2d)
                
                # Create color map
                color_map = np.zeros((feat_size, feat_size, 3))
                for h in range(feat_size):
                    for w in range(feat_size):
                        if score_threshold is not None and max_scores_2d[h, w] <= score_threshold:
                            # Set to white for patches below threshold
                            color_map[h, w] = [1.0, 1.0, 1.0]
                        else:
                            if visualize_class_index is not None and visualize_class_index != patch_predictions_2d[h, w]:
                                
                                color_map[h, w] = [1.0, 1.0, 1.0]
                            else:
                                pred_class = patch_predictions_2d[h, w]
                                base_color = colors[pred_class]
                                
                                # Apply smoothing based on score if enabled
                                if smoothing:
                                    # Normalize score to [0, 1] range for alpha blending
                                    score = max_scores_2d[h, w]
                                    # Use score as alpha to blend with white background
                                    alpha = min(1.0, max(0.0, score))
                                    # Blend: color * alpha + white * (1 - alpha)
                                    blended_color = [
                                        base_color[0] * alpha + 1.0 * (1 - alpha),
                                        base_color[1] * alpha + 1.0 * (1 - alpha),
                                        base_color[2] * alpha + 1.0 * (1 - alpha)
                                    ]
                                    color_map[h, w] = blended_color
                                else:
                                    color_map[h, w] = base_color
                
                # Create figure with original image, color map, and legend only for present classes
                # Calculate required height for the legend based on number of unique classes
                num_unique_classes = len(unique_classes)
                y_step = 0.08
                required_height = num_unique_classes * y_step + 0.1  # Add some padding
                
                # Adjust figure size if legend is too tall
                if required_height > 1.0:
                    # Scale the y_step to fit within the available space
                    y_step = 0.9 / num_unique_classes
                    fig_height = max(6, 6 * (required_height / 1.0))  # Scale figure height if needed
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, fig_height))
                else:
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
                
                # Left: Original image
                ax1.imshow(img)
                ax1.set_title(f'Original Image (Batch {i})', fontsize=14)
                ax1.axis('off')
                
                # Center: Color-coded prediction map
                ax2.imshow(color_map)
                ax2.set_title(f'Patch-to-Instance Predictions (Batch {i})', fontsize=14)
                ax2.axis('off')
                
                # Right: Legend only for classes present in the feature map
                ax3.axis('off')
                ax3.set_title('Present Classes', fontsize=14)
                
                # Create legend only for unique classes found in the prediction
                y_start = 0.95
                
                for idx, class_idx in enumerate(unique_classes):
                    y_pos = y_start - idx * y_step
                    
                    # Skip if y_pos would be below visible area
                    if y_pos < 0.05:
                        break
                    
                    if class_idx < num_class_labels:
                        color = colors[class_idx]
                        class_name_text = query_name[class_idx]
                    else:
                        color = [1.0, 1.0, 1.0]  # Background
                        class_name_text = 'Background'
                    
                    # Create small color patch
                    rect = plt.Rectangle((0.05, y_pos - 0.02), 0.1, 0.04, 
                                       facecolor=color, edgecolor='black', transform=ax3.transAxes)
                    ax3.add_patch(rect)
                    ax3.text(0.2, y_pos, f'{class_idx}: {class_name_text}', 
                            fontsize=11, va='center', transform=ax3.transAxes)
                
                ax3.set_xlim(0, 1)
                ax3.set_ylim(0, 1)
                plt.tight_layout()
                plt.savefig(f'{batch_dir}/patch_to_instance_predictions.png', 
                           dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                if visualize_class_index is not None:
                    patch_importance_scores = instance_to_patch_scores[i,:,visualize_class_index].detach().cpu().numpy()
                    patch_importance_scores_2d = patch_importance_scores.reshape(feat_size, feat_size)
                    max_value = patch_importance_scores_2d.max()
                    fig, ax = plt.subplots(1, 1, figsize=(15, 6))
                    ax.imshow(patch_importance_scores_2d, cmap='gray', vmin=0, vmax = max_value)
                    ax.set_title(f'Patch Importance Scores for Class {visualize_class_index}')
                    ax.axis('off')
                    plt.tight_layout()
                    plt.savefig(f'{batch_dir}/patch_importance_scores_class_id:{visualize_class_index}.png', dpi=150, bbox_inches='tight')
                    plt.close(fig)
        
        if instance_to_patch_scores is not None:
            for i in range(bs):
                batch_dir = f'label_weight_vis/batch_{i}'
                os.makedirs(batch_dir, exist_ok=True)
                
                # Get denormalized image for this batch
                if denorm_images.shape[0] != bs:
                    img = denorm_images[0].permute(1, 2, 0).detach().cpu().numpy()
                else:
                    img = denorm_images[i].permute(1, 2, 0).detach().cpu().numpy()
                
                # If specific class is specified, visualize only that class
                if visualize_class_index is not None:
                    class_attention = instance_to_patch_scores[i, :, visualize_class_index].detach().cpu().numpy()  # [num_patches]
                    class_attention_2d = class_attention.reshape(feat_size, feat_size)
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # Left: Original image
                    ax1.imshow(img)
                    ax1.set_title(f'Image {i}')
                    ax1.axis('off')
                    
                    # Right: Attention map for specified class
                    im = ax2.imshow(class_attention_2d, cmap='hot', vmin=0, vmax=class_attention_2d.max())
                    ax2.set_title(f'Class {visualize_class_index}: {query_name[visualize_class_index]} Attention Map')
                    ax2.axis('off')
                    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
                    
                    plt.tight_layout()
                    plt.savefig(f'{batch_dir}/class_{visualize_class_index}_attention.png', dpi=150, bbox_inches='tight')
                    plt.close(fig)
                else:
                    # Visualize all classes
                    for class_idx in range(num_class_labels):
                        class_attention = instance_to_patch_scores[i, :, class_idx].detach().cpu().numpy()  # [num_patches]
                        class_attention_2d = class_attention.reshape(feat_size, feat_size)
                        
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                        
                        # Left: Original image
                        ax1.imshow(img)
                        ax1.set_title(f'Image {i}')
                        ax1.axis('off')
                        
                        # Right: Attention map for this class
                        im = ax2.imshow(class_attention_2d, cmap='hot', vmin=0, vmax=class_attention_2d.max())
                        ax2.set_title(f'Class {class_idx}: {query_name[class_idx]} Attention Map')
                        ax2.axis('off')
                        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
                        
                        plt.tight_layout()
                        plt.savefig(f'{batch_dir}/class_{class_idx}_{query_name[class_idx]}.png', dpi=150, bbox_inches='tight')
                        plt.close(fig)
        # for i in range(bs):
        #     # Create directory for this batch item
        #     batch_dir = f'label_weight_vis/batch_{i}'
        #     os.makedirs(batch_dir, exist_ok=True)
            
        #     # Denormalize image for this batch
        #     if denorm_images.shape[0] != bs:
        #         img = denorm_images[0].permute(1, 2, 0).detach().cpu().numpy()
        #     else:
        #         img = denorm_images[i].permute(1, 2, 0).detach().cpu().numpy()
            
        #     # Process each class separately
        #     for class_idx in range(num_class_labels):
        #         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
        #         # Left: Original image
        #         ax1.imshow(img)
        #         ax1.set_title(f'Image {i} - Class {class_idx}')
        #         ax1.axis('off')
                
        #         # Right: Attention map for this class (matrix visualization only)
        #         class_attention = instance_to_patch_scores[i, :, class_idx].detach().cpu().numpy()  # [num_patches]
        #         class_attention_2d = class_attention.reshape(feat_size, feat_size)
        #         class_attention_sum = class_attention.sum().item()
                
        #         # Visualize attention matrix directly without overlaying on image
        #         im_overlay = ax2.imshow(class_attention_2d, cmap='hot', vmin=0, vmax=max_value)
                
        #         # Add instance score as title
        #         instance_score = instance_scores[i, class_idx].item()
        #         ax2.set_title(f'Class {class_idx} Attention Matrix (Score: {instance_score:.3f}, Patch sum: {class_attention_sum:.3f})')
        #         ax2.axis('off')                
                
        #         # Add colorbar
        #         plt.colorbar(im_overlay, ax=ax2, fraction=0.046, pad=0.04)
                
        #         plt.tight_layout()
                
        #         # Save each class separately
        #         plt.savefig(f'{batch_dir}/{query_name[class_idx]}.png', 
        #                    dpi=150, bbox_inches='tight')
        #         plt.close(fig)  

    def visualize_attention_weight(self, images, attn_weight, logits, score_threshold=0.3, draw_all_queries=False):
        """Visualize attention weights overlaid on images"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Denormalize images
        mean = self.vision_processor.image_mean if hasattr(self.vision_processor, 'image_mean') else self.vision_processor.image_processor.image_mean
        std = self.vision_processor.image_std if hasattr(self.vision_processor, 'image_std') else self.vision_processor.image_processor.image_std
        mean = torch.tensor(mean).view(1, 3, 1, 1).to(images.device)
        std = torch.tensor(std).view(1, 3, 1, 1).to(images.device)
        denorm_images = images * std + mean
        denorm_images = torch.clamp(denorm_images, 0, 1)
        
        # Process attention weights
        # attn_weight shape: [bs, heads, 1, num_tokens]
        bs, heads, _, num_tokens = attn_weight.shape
        
        # # Apply softmax
        # attn_weight = torch.softmax(attn_weight, dim=-1)
        
        # Average across heads
        attn_weight = attn_weight.mean(dim=1)  # [bs, 1, num_tokens]
        attn_weight = attn_weight.squeeze(1) if attn_weight.shape[1] == 1 else attn_weight   # [bs, num_queries, num_tokens] or [bs, num_tokens]
        
        # Handle multiple queries case            
        if len(attn_weight.shape) == 3:
            # Multiple queries: [bs, num_queries, num_tokens]
            bs, num_queries, num_tokens = attn_weight.shape
            
            # Calculate feat_size from num_tokens
            feat_size = int(np.sqrt(num_tokens))
            assert feat_size * feat_size == num_tokens, f"num_tokens {num_tokens} is not a perfect square"
            
            # Reshape to spatial dimensions
            attn_weight = attn_weight.reshape(bs, num_queries, feat_size, feat_size)  # [bs, num_queries, feat_size, feat_size]
            if self.args.attention_type == 'ml_decoder':
                from utils.hico_text_label import hico_obj_text_label, hico_text_label
                from hico_list import hico_verbs_sentence
                from vcoco_text_label import vcoco_obj_text_label, vcoco_hoi_text_label
                from vcoco_list import vcoco_verbs_sentence
                if self.args.dataset_name == 'hico':
                    query_name = hico_obj_text_label
                    class_name = hico_verbs_sentence
                elif self.args.dataset_name == 'vcoco':
                    query_name = vcoco_obj_text_label
                    class_name = vcoco_verbs_sentence
                else:
                    raise ValueError(f"Unsupported dataset: {self.args.dataset_name}")
            else:
                query_name = None
                class_name = None

            # Visualize each image with attention overlay for each query
            for i in range(bs):
                logit = logits[i].sigmoid()  # [num_queries, num_classes]
                threshold = score_threshold  # Set threshold
                if len(logit.shape) == 1:
                    logit = logit.unsqueeze(0)
                # Plot attention maps for each query separately
                for q in range(num_queries):
                    if (logit[q]<threshold).all() and not draw_all_queries:
                        continue
                    attn_map = attn_weight[i, q].detach().cpu().numpy()
                    
                    # Get image
                    if denorm_images.shape[0] != attn_weight.shape[0]:
                        img = denorm_images[0].permute(1, 2, 0).detach().cpu().numpy()
                    else:
                        img = denorm_images[i].permute(1, 2, 0).detach().cpu().numpy()
                    
                    # Save individual query attention as separate image
                    fig_single, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
                    
                    # Original image
                    ax1.imshow(img)
                    ax1.set_title(f'Original Image')
                    ax1.axis('off')
                    
                    # Attention map
                    im2 = ax2.imshow(attn_map, cmap='hot', interpolation='bilinear')
                    if query_name is not None:
                        ax2.set_title(f'{query_name[q]} Attention Map')
                    else:
                        ax2.set_title(f'Query {q} Attention Map')
                    ax2.axis('off')
                    plt.colorbar(im2, ax=ax2)
                    
                    # Overlay
                    from scipy.ndimage import zoom
                    img_h, img_w = img.shape[:2]
                    attn_resized = zoom(attn_map, (img_h/feat_size, img_w/feat_size), order=1)
                    ax3.imshow(img)
                    im3 = ax3.imshow(attn_resized, cmap='hot', alpha=0.6, interpolation='bilinear')
                    if query_name is not None:
                        ax3.set_title(f'{query_name[q]} Image + Attention Overlay')
                    else:
                        ax3.set_title(f'Query {q} Image + Attention Overlay')
                    ax3.axis('off')
                    
                    # Logit predictions with threshold
                    query_logits = logit[q].detach().cpu().numpy()  # [num_classes]
                    # threshold = 0.5  # Set threshold
                    
                    # Get indices and values above threshold, sorted by logit value
                    above_threshold = query_logits > threshold
                    if above_threshold.any():
                        high_indices = np.where(above_threshold)[0]
                        high_values = query_logits[high_indices]
                        sorted_idx = np.argsort(high_values)[::-1]  # Sort in descending order
                        
                        text_content = "Predictions > threshold:\n"
                        for idx in sorted_idx:
                            class_idx = high_indices[idx]
                            class_score = high_values[idx]
                            text_content += f"{class_name[class_idx]}: {class_score:.3f}\n"
                    else:
                        text_content = f"No predictions > {threshold}"
                    
                    ax4.text(0.05, 0.95, text_content, transform=ax4.transAxes, 
                            verticalalignment='top', fontsize=10, fontfamily='monospace')
                    if query_name is not None:
                        ax4.set_title(f'{query_name[q]} Predictions')
                    else:
                        ax4.set_title(f'Query {q} Predictions')
                    ax4.axis('off')
                    
                    plt.tight_layout()
                    os.makedirs(f'attention_weight/attention_weight_img{i}', exist_ok=True)
                    if query_name is not None:
                        plt.savefig(f'attention_weight/attention_weight_img{i}/{query_name[q]}.png', dpi=150, bbox_inches='tight')
                    else:
                        plt.savefig(f'attention_weight/attention_weight_img{i}/query{q}.png', dpi=150, bbox_inches='tight')
                    plt.close(fig_single)  # Close the individual query figure
                
        else:
            # Single query: [bs, num_tokens]
            bs, num_tokens = attn_weight.shape
            
            # Calculate feat_size from num_tokens
            feat_size = int(np.sqrt(num_tokens))
            assert feat_size * feat_size == num_tokens, f"num_tokens {num_tokens} is not a perfect square"
            
            # Reshape to spatial dimensions
            attn_weight = attn_weight.reshape(bs, feat_size, feat_size)  # [bs, feat_size, feat_size]
            
            # Visualize each image with attention overlay
            for i in range(bs):
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                
                # Original image
                if denorm_images.shape[0] != attn_weight.shape[0]:
                    img = denorm_images[0].permute(1, 2, 0).detach().cpu().numpy()
                else:
                    img = denorm_images[i].permute(1, 2, 0).detach().cpu().numpy()
                ax1.imshow(img)
                ax1.set_title('Original Image')
                ax1.axis('off')
                
                # Attention map
                attn_map = attn_weight[i].detach().cpu().numpy()
                im2 = ax2.imshow(attn_map, cmap='hot', interpolation='bilinear')
                ax2.set_title('Attention Map')
                ax2.axis('off')
                plt.colorbar(im2, ax=ax2)
                
                # Overlay
                ax3.imshow(img)
                # Resize attention map to match image size
                from scipy.ndimage import zoom
                img_h, img_w = img.shape[:2]
                attn_resized = zoom(attn_map, (img_h/feat_size, img_w/feat_size), order=1)
                im3 = ax3.imshow(attn_resized, cmap='hot', alpha=0.6, interpolation='bilinear')
                ax3.set_title('Image + Attention Overlay')
                ax3.axis('off')
                
                plt.tight_layout()
                os.makedirs('attention_weight', exist_ok=True)
                plt.savefig(f'attention_weight/attention_weight_{i}.png')
                plt.close(fig)  # Close the figure to free memory
    
    def visualize_instance_score(self, images, instance_scores, so_indices, meta_data):
        """Visualize instance scores"""
        import matplotlib.pyplot as plt
        import numpy as np
        import matplotlib.patches as patches
        
        # Denormalize images
        mean = self.vision_processor.image_mean if hasattr(self.vision_processor, 'image_mean') else self.vision_processor.image_processor.image_mean
        std = self.vision_processor.image_std if hasattr(self.vision_processor, 'image_std') else self.vision_processor.image_processor.image_std
        mean = torch.tensor(mean).view(1, 3, 1, 1).to(images.device)
        std = torch.tensor(std).view(1, 3, 1, 1).to(images.device)
        denorm_images = images * std + mean
        denorm_images = torch.clamp(denorm_images, 0, 1)
        
        from utils.hico_text_label import hico_obj_text_label
        from vcoco_text_label import vcoco_obj_text_label
        if self.args.dataset_name == 'hico':
            object_classes = hico_obj_text_label
        elif self.args.dataset_name == 'vcoco':
            object_classes = vcoco_obj_text_label
        else:
            raise ValueError(f"Unsupported dataset: {self.args.dataset_name}")
        humna_boxes = meta_data[0]['human_boxes'].cpu().numpy()
        object_boxes = meta_data[0]['object_boxes'].cpu().numpy()
        # Visualize instance scores
        for i,(h_box,o_box) in enumerate(zip(humna_boxes, object_boxes)):
            score = instance_scores[0, i, 0].item()
            if score<0.2:
                continue
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            img = denorm_images[0].permute(1, 2, 0).detach().cpu().numpy()
            ax.imshow(img)
            rect = patches.Rectangle((h_box[0], h_box[1]), h_box[2] - h_box[0], h_box[3] - h_box[1], linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            rect = patches.Rectangle((o_box[0], o_box[1]), o_box[2] - o_box[0], o_box[3] - o_box[1], linewidth=2, edgecolor='blue', facecolor='none')
            ax.add_patch(rect)
            
            # ax.text(h_box[0], h_box[1]-5, f'Human: {score:.3f}', color='red', fontsize=10, weight='bold')
            # ax.text(o_box[0], o_box[1]-5, f'Object: {score:.3f}', color='blue', fontsize=10, weight='bold')
            ax.set_title(f'Instance Scores - Image {i} (class {object_classes[so_indices[i]]}: score {score:.3f})')
            ax.axis('off')
            os.makedirs('instance_score_vis', exist_ok=True)
            
            plt.savefig(f'instance_score_vis/instance_score_img_{i}.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
        fig
    
    
    def visualize_dense_interactiveness(self,images,vision_out_dict, meta_data, so_indices, threshold=0.5):
        """Visualize dense interactiveness"""
        import matplotlib.pyplot as plt
        import numpy as np
        import matplotlib.patches as patches
        
        filename = meta_data[0]['filename']
        valid_vis_idx = [i for i,indice in enumerate(so_indices) if indice!=0]
        # Denormalize images
        mean = self.vision_processor.image_mean if hasattr(self.vision_processor, 'image_mean') else self.vision_processor.image_processor.image_mean
        std = self.vision_processor.image_std if hasattr(self.vision_processor, 'image_std') else self.vision_processor.image_processor.image_std
        mean = torch.tensor(mean).view(1, 3, 1, 1).to(images.device)
        std = torch.tensor(std).view(1, 3, 1, 1).to(images.device)
        denorm_images = images * std + mean
        denorm_images = torch.clamp(denorm_images, 0, 1)
        
        h = w = self.attention_pooling.feat_size
        # global interactiveness
        image_level_sub_patch_importance_score = vision_out_dict['image_level_sub_patch_importance_score'][0] # HW \times C
        image_level_obj_patch_importance_score = vision_out_dict['image_level_obj_patch_importance_score'][0] # HW \times C
        masked_sub_patch_importance_score = vision_out_dict['masked_sub_patch_importance_score'][:,0] # N
        masked_obj_patch_importance_score = vision_out_dict['masked_obj_patch_importance_score'][torch.arange(len(so_indices)),so_indices] # N
        patch_level_sub_instance_scores = vision_out_dict['patch_level_sub_instance_scores'][0] # HW \times C
        
        # local interactiveness
        masked_sub_instance_scores = vision_out_dict['masked_sub_instance_scores'][:,0] # N
        masked_obj_instance_scores = vision_out_dict['masked_obj_instance_scores'][torch.arange(len(so_indices)),so_indices] #N
        instance_level_sub_patch_importance_score = vision_out_dict['instance_level_sub_patch_importance_score'] # N \times HW \times C
        instance_level_obj_patch_importance_score = vision_out_dict['instance_level_obj_patch_importance_score'] # N \times HW \times C
        patch_level_obj_instance_scores = vision_out_dict['patch_level_obj_instance_scores'][0] # HW \times C
        
        sub_pair_masks = vision_out_dict['sub_pair_masks'] # NxHW
        obj_pair_masks = vision_out_dict['obj_pair_masks'] # NxHW
        sub_interactiveness_scores = masked_sub_instance_scores * masked_sub_patch_importance_score
        obj_interactiveness_scores = masked_obj_instance_scores * masked_obj_patch_importance_score
        combined_interactiveness_scores = (sub_interactiveness_scores * obj_interactiveness_scores)**0.5
        
        sub_boxes = meta_data[0]['human_boxes'] # Nx4
        obj_boxes = meta_data[0]['object_boxes'] # Nx4
        
        from utils.hico_text_label import hico_obj_text_label
        from vcoco_text_label import vcoco_obj_text_label
        if self.args.dataset_name == 'hico':
            object_classes = hico_obj_text_label
        elif self.args.dataset_name == 'vcoco':
            object_classes = vcoco_obj_text_label
        else:
            raise ValueError(f"Unsupported dataset: {self.args.dataset_name}")
        
        # Visualize dense interactiveness for each subject-object pair
        for i in range(len(sub_boxes)):
            if i not in valid_vis_idx:
                continue
            sub_box = sub_boxes[i].cpu().numpy()
            obj_box = obj_boxes[i].cpu().numpy()
            obj_class_idx = so_indices[i]
            
            # Create figure with 5 subplots (original + subject/object separate + global/local without overlay)
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Original image with boxes
            img = denorm_images[0].permute(1, 2, 0).detach().cpu().numpy()
            axes[0, 0].imshow(img)
            # Draw subject box (red)
            rect_sub = patches.Rectangle((sub_box[0], sub_box[1]), sub_box[2] - sub_box[0], sub_box[3] - sub_box[1], 
                                       linewidth=2, edgecolor='red', facecolor='none')
            axes[0, 0].add_patch(rect_sub)
            # Draw object box (blue)
            rect_obj = patches.Rectangle((obj_box[0], obj_box[1]), obj_box[2] - obj_box[0], obj_box[3] - obj_box[1], 
                                       linewidth=2, edgecolor='blue', facecolor='none')
            axes[0, 0].add_patch(rect_obj)
            # Add combined interactiveness score
            axes[0, 0].text(10, 30, f'Combined Score: {combined_interactiveness_scores[i]:.3f}', 
                        color='white', fontsize=12, weight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.8))
            axes[0, 0].set_title(f'Original Image - Pair {i}\nObject: {object_classes[obj_class_idx]}')
            axes[0, 0].axis('off')
            
            # Global interactiveness visualization with overlay
            axes[0, 1].imshow(img)
            # Overlay global patch importance scores
            global_sub_score = image_level_sub_patch_importance_score[:, 0].reshape(h, w).detach().cpu().numpy()
            global_obj_score = image_level_obj_patch_importance_score[:, obj_class_idx].reshape(h, w).detach().cpu().numpy()
            # Combine subject and object scores for visualization
            combined_global_score = (global_sub_score + global_obj_score) / 2
            # Resize to image size for overlay
            img_h, img_w = img.shape[:2]
            axes[0, 1].imshow(combined_global_score, alpha=0.5, cmap='hot', extent=[0, img_w, img_h, 0])
            # Draw boxes
            rect_sub = patches.Rectangle((sub_box[0], sub_box[1]), sub_box[2] - sub_box[0], sub_box[3] - sub_box[1], 
                                       linewidth=2, edgecolor='red', facecolor='none')
            axes[0, 1].add_patch(rect_sub)
            rect_obj = patches.Rectangle((obj_box[0], obj_box[1]), obj_box[2] - obj_box[0], obj_box[3] - obj_box[1], 
                                       linewidth=2, edgecolor='blue', facecolor='none')
            axes[0, 1].add_patch(rect_obj)
            # Add masked patch importance scores as text
            axes[0, 1].text(sub_box[0], sub_box[1]-10, f'Sub: {masked_sub_patch_importance_score[i]:.3f}', 
                        color='red', fontsize=10, weight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            axes[0, 1].text(obj_box[0], obj_box[1]-10, f'Obj: {masked_obj_patch_importance_score[i]:.3f}', 
                        color='blue', fontsize=10, weight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            axes[0, 1].set_title(f'Global Interactiveness (Overlay) - Pair {i}')
            axes[0, 1].axis('off')
            
            # Local interactiveness visualization with overlay
            axes[0, 2].imshow(img)
            # Overlay local patch importance scores
            local_sub_score = instance_level_sub_patch_importance_score[i,:, 0].reshape(h, w).detach().cpu().numpy()
            local_obj_score = instance_level_obj_patch_importance_score[i,:, obj_class_idx].reshape(h, w).detach().cpu().numpy()
            # Combine subject and object scores for visualization
            combined_local_score = (local_sub_score + local_obj_score) / 2
            axes[0, 2].imshow(combined_local_score, alpha=0.5, cmap='hot', extent=[0, img_w, img_h, 0])
            # Draw boxes
            rect_sub = patches.Rectangle((sub_box[0], sub_box[1]), sub_box[2] - sub_box[0], sub_box[3] - sub_box[1], 
                                       linewidth=2, edgecolor='red', facecolor='none')
            axes[0, 2].add_patch(rect_sub)
            rect_obj = patches.Rectangle((obj_box[0], obj_box[1]), obj_box[2] - obj_box[0], obj_box[3] - obj_box[1], 
                                       linewidth=2, edgecolor='blue', facecolor='none')
            axes[0, 2].add_patch(rect_obj)
            # Add masked instance scores as text
            axes[0, 2].text(sub_box[0], sub_box[1]-10, f'Sub: {masked_sub_instance_scores[i]:.3f}', 
                        color='red', fontsize=10, weight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            axes[0, 2].text(obj_box[0], obj_box[1]-10, f'Obj: {masked_obj_instance_scores[i]:.3f}', 
                        color='blue', fontsize=10, weight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            axes[0, 2].set_title(f'Local Interactiveness (Overlay) - Pair {i}')
            axes[0, 2].axis('off')
            
            # Subject and Object separate visualizations
            # Subject global score
            axes[1, 0].imshow(img)
            axes[1, 0].imshow(global_sub_score, alpha=0.5, cmap='hot', extent=[0, img_w, img_h, 0])
            # Draw boxes
            rect_sub = patches.Rectangle((sub_box[0], sub_box[1]), sub_box[2] - sub_box[0], sub_box[3] - sub_box[1], 
                                       linewidth=2, edgecolor='red', facecolor='none')
            axes[1, 0].add_patch(rect_sub)
            
            axes[1, 0].set_title(f'Subject Global Score: {masked_sub_patch_importance_score[i]:.3f}')
            axes[1, 0].axis('off')
            
            # Object global score
            axes[1, 1].imshow(img)
            axes[1, 1].imshow(global_obj_score, alpha=0.5, cmap='hot', extent=[0, img_w, img_h, 0])
            # Draw boxes
            
            rect_obj = patches.Rectangle((obj_box[0], obj_box[1]), obj_box[2] - obj_box[0], obj_box[3] - obj_box[1], 
                                       linewidth=2, edgecolor='blue', facecolor='none')
            axes[1, 1].add_patch(rect_obj)
            axes[1, 1].set_title(f'Object Global Score: {masked_obj_patch_importance_score[i]:.3f}')
            axes[1, 1].axis('off')
            
            # Combined local score without overlay
            axes[1, 2].imshow(combined_local_score, cmap='hot')
            axes[1, 2].set_title(f'Local Combined Score')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            os.makedirs('dense_interactiveness_vis', exist_ok=True)
            os.makedirs(f'dense_interactiveness_vis/{filename}/{i}', exist_ok=True)
            plt.savefig(f'dense_interactiveness_vis/{filename}/{i}/dense_interactiveness_pair.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # Create separate figure for subject and object local scores with patch level scores
            fig2, axes2 = plt.subplots(2, 3, figsize=(12, 12))
            
            # Subject local score
            axes2[0, 0].imshow(local_sub_score, cmap='hot')
            rect_sub = patches.Rectangle((sub_box[0], sub_box[1]), sub_box[2] - sub_box[0], sub_box[3] - sub_box[1], 
                                       linewidth=2, edgecolor='red', facecolor='none')
            axes2[0, 0].add_patch(rect_sub)
            axes2[0, 0].set_title(f'Subject Local Score: {masked_sub_instance_scores[i]:.3f}')
            axes2[0, 0].axis('off')
            
            # Subject patch level instance score
            patch_sub_score = patch_level_sub_instance_scores[:, 0].reshape(h, w).detach().cpu().numpy()
            # Apply thresholding - set values below 0.5 to 0
            patch_sub_score = np.where(patch_sub_score < threshold, 0, patch_sub_score)
            axes2[0, 1].imshow(patch_sub_score, cmap='Reds')
            axes2[0, 1].set_title(f'Subject Patch Level Instance Score')
            axes2[0, 1].axis('off')
            
            # Subject pair mask
            axes2[0, 2].imshow(sub_pair_masks[i].reshape(h, w).detach().cpu().numpy(), cmap='hot')
            axes2[0, 2].set_title(f'Subject Pair Mask')
            axes2[0, 2].axis('off')
            
            # Object local score
            axes2[1, 0].imshow(local_obj_score, cmap='hot')
            rect_obj = patches.Rectangle((obj_box[0], obj_box[1]), obj_box[2] - obj_box[0], obj_box[3] - obj_box[1], 
                                       linewidth=2, edgecolor='blue', facecolor='none')
            axes2[1, 0].add_patch(rect_obj)
            axes2[1, 0].set_title(f'Object Local Score: {masked_obj_instance_scores[i]:.3f}')
            axes2[1, 0].axis('off')
            
            # Object patch level instance score
            patch_obj_score = patch_level_obj_instance_scores[:, obj_class_idx].reshape(h, w).detach().cpu().numpy()
            # Apply thresholding - set values below 0.5 to 0
            patch_obj_score = np.where(patch_obj_score < threshold, 0, patch_obj_score)
            axes2[1, 1].imshow(patch_obj_score, cmap='Blues')
            axes2[1, 1].set_title(f'Object Patch Level Instance Score')
            axes2[1, 1].axis('off')
            
            # Object pair mask
            axes2[1, 2].imshow(obj_pair_masks[i].reshape(h, w).detach().cpu().numpy(), cmap='hot')
            axes2[1, 2].set_title(f'Object Pair Mask')
            axes2[1, 2].axis('off')
            
            plt.tight_layout()
            os.makedirs(f'dense_interactiveness_vis/{filename}/{i}', exist_ok=True)
            plt.savefig(f'dense_interactiveness_vis/{filename}/{i}/local_separate_pair.png', dpi=150, bbox_inches='tight')
            plt.close(fig2)
            
            # Create third figure for subject total interactiveness analysis
            fig3, axes3 = plt.subplots(2, 5, figsize=(20, 8))
            
            # Subject row
            # Subject total interactiveness (with original image overlay and boxes)
            axes3[0, 0].imshow(img)
            # Draw subject box (red)
            rect_sub = patches.Rectangle((sub_box[0], sub_box[1]), sub_box[2] - sub_box[0], sub_box[3] - sub_box[1], 
                                       linewidth=2, edgecolor='red', facecolor='none')
            axes3[0, 0].add_patch(rect_sub)
            # # Draw object box (blue) for reference
            # rect_obj = patches.Rectangle((obj_box[0], obj_box[1]), obj_box[2] - obj_box[0], obj_box[3] - obj_box[1], 
            #                            linewidth=2, edgecolor='blue', facecolor='none')
            # axes3[0, 0].add_patch(rect_obj)
            # Add subject total interactiveness score as text
            # axes3[0, 0].text(10, 30, f'Sub Total: {sub_interactiveness_scores[i]:.3f}', 
            #             color='white', fontsize=12, weight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.8))
            axes3[0, 0].set_title(f'Subject Total Interactiveness: {sub_interactiveness_scores[i]:.3f}')
            axes3[0, 0].axis('off')
            
            # Subject local interactiveness (instance level patch importance)
            axes3[0, 1].imshow(local_sub_score, cmap='hot')
            # Draw subject box (red)
            rect_sub = patches.Rectangle((sub_box[0], sub_box[1]), sub_box[2] - sub_box[0], sub_box[3] - sub_box[1], 
                                       linewidth=2, edgecolor='red', facecolor='none')
            axes3[0, 1].add_patch(rect_sub)
            # Draw object box (blue) for reference
            rect_obj = patches.Rectangle((obj_box[0], obj_box[1]), obj_box[2] - obj_box[0], obj_box[3] - obj_box[1], 
                                       linewidth=2, edgecolor='blue', facecolor='none')
            axes3[0, 1].add_patch(rect_obj)
            axes3[0, 1].set_title(f'Subject Instance Level Patch Importance')
            axes3[0, 1].axis('off')
            
            # Subject patch level instance score
            axes3[0, 2].imshow(patch_sub_score, cmap='Reds')
            # Draw subject box (red)
            rect_sub = patches.Rectangle((sub_box[0], sub_box[1]), sub_box[2] - sub_box[0], sub_box[3] - sub_box[1], 
                                       linewidth=2, edgecolor='red', facecolor='none')
            axes3[0, 2].add_patch(rect_sub)
            # Draw object box (blue) for reference
            rect_obj = patches.Rectangle((obj_box[0], obj_box[1]), obj_box[2] - obj_box[0], obj_box[3] - obj_box[1], 
                                       linewidth=2, edgecolor='blue', facecolor='none')
            axes3[0, 2].add_patch(rect_obj)
            axes3[0, 2].set_title(f'Subject Patch Level Instance Score')
            axes3[0, 2].axis('off')
            
            # Subject global interactiveness (image level patch importance)
            axes3[0, 3].imshow(global_sub_score, cmap='hot')
            # Draw subject box (red)
            rect_sub = patches.Rectangle((sub_box[0], sub_box[1]), sub_box[2] - sub_box[0], sub_box[3] - sub_box[1], 
                                       linewidth=2, edgecolor='red', facecolor='none')
            axes3[0, 3].add_patch(rect_sub)
            # Draw object box (blue) for reference
            rect_obj = patches.Rectangle((obj_box[0], obj_box[1]), obj_box[2] - obj_box[0], obj_box[3] - obj_box[1], 
                                       linewidth=2, edgecolor='blue', facecolor='none')
            axes3[0, 3].add_patch(rect_obj)
            axes3[0, 3].set_title(f'Subject Image Level Patch Importance')
            axes3[0, 3].axis('off')
            
            # Subject pair mask
            axes3[0, 4].imshow((~(sub_pair_masks[i].bool())).float().reshape(h, w).detach().cpu().numpy(), cmap='hot')
            # Draw subject box (red)
            rect_sub = patches.Rectangle((sub_box[0], sub_box[1]), sub_box[2] - sub_box[0], sub_box[3] - sub_box[1], 
                                       linewidth=2, edgecolor='red', facecolor='none')
            axes3[0, 4].add_patch(rect_sub)
            # Draw object box (blue) for reference
            rect_obj = patches.Rectangle((obj_box[0], obj_box[1]), obj_box[2] - obj_box[0], obj_box[3] - obj_box[1], 
                                       linewidth=2, edgecolor='blue', facecolor='none')
            axes3[0, 4].add_patch(rect_obj)
            axes3[0, 4].set_title(f'Subject Pair Mask')
            axes3[0, 4].axis('off')
            
            # Object row
            # Object total interactiveness (with original image overlay and boxes)
            axes3[1, 0].imshow(img)
            # # Draw subject box (red) for reference
            # rect_sub = patches.Rectangle((sub_box[0], sub_box[1]), sub_box[2] - sub_box[0], sub_box[3] - sub_box[1], 
            #                            linewidth=2, edgecolor='red', facecolor='none')
            # axes3[1, 0].add_patch(rect_sub)
            # Draw object box (blue)
            rect_obj = patches.Rectangle((obj_box[0], obj_box[1]), obj_box[2] - obj_box[0], obj_box[3] - obj_box[1], 
                                       linewidth=2, edgecolor='blue', facecolor='none')
            axes3[1, 0].add_patch(rect_obj)
            # Add object total interactiveness score as text
            # axes3[1, 0].text(10, 30, f'Obj Total: {obj_interactiveness_scores[i]:.3f}', 
            #             color='white', fontsize=12, weight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.8))
            axes3[1, 0].set_title(f'Object Total Interactiveness: {obj_interactiveness_scores[i]:.3f}')
            axes3[1, 0].axis('off')
            
            # Object local interactiveness (instance level patch importance)
            axes3[1, 1].imshow(local_obj_score, cmap='hot')
            # Draw subject box (red) for reference
            rect_sub = patches.Rectangle((sub_box[0], sub_box[1]), sub_box[2] - sub_box[0], sub_box[3] - sub_box[1], 
                                       linewidth=2, edgecolor='red', facecolor='none')
            axes3[1, 1].add_patch(rect_sub)
            # Draw object box (blue)
            rect_obj = patches.Rectangle((obj_box[0], obj_box[1]), obj_box[2] - obj_box[0], obj_box[3] - obj_box[1], 
                                       linewidth=2, edgecolor='blue', facecolor='none')
            axes3[1, 1].add_patch(rect_obj)
            axes3[1, 1].set_title(f'Object Instance Level Patch Importance')
            axes3[1, 1].axis('off')
            
            # Object patch level instance score
            axes3[1, 2].imshow(patch_obj_score, cmap='Blues')
            # Draw subject box (red) for reference
            rect_sub = patches.Rectangle((sub_box[0], sub_box[1]), sub_box[2] - sub_box[0], sub_box[3] - sub_box[1], 
                                       linewidth=2, edgecolor='red', facecolor='none')
            axes3[1, 2].add_patch(rect_sub)
            # Draw object box (blue)
            rect_obj = patches.Rectangle((obj_box[0], obj_box[1]), obj_box[2] - obj_box[0], obj_box[3] - obj_box[1], 
                                       linewidth=2, edgecolor='blue', facecolor='none')
            axes3[1, 2].add_patch(rect_obj)
            axes3[1, 2].set_title(f'Object Patch Level Instance Score')
            axes3[1, 2].axis('off')
            
            # Object global interactiveness (image level patch importance)
            axes3[1, 3].imshow(global_obj_score, cmap='hot')
            # Draw subject box (red) for reference
            rect_sub = patches.Rectangle((sub_box[0], sub_box[1]), sub_box[2] - sub_box[0], sub_box[3] - sub_box[1], 
                                       linewidth=2, edgecolor='red', facecolor='none')
            axes3[1, 3].add_patch(rect_sub)
            # Draw object box (blue)
            rect_obj = patches.Rectangle((obj_box[0], obj_box[1]), obj_box[2] - obj_box[0], obj_box[3] - obj_box[1], 
                                       linewidth=2, edgecolor='blue', facecolor='none')
            axes3[1, 3].add_patch(rect_obj)
            axes3[1, 3].set_title(f'Object Image Level Patch Importance')
            axes3[1, 3].axis('off')
            
            # Object pair mask
            axes3[1, 4].imshow((~(obj_pair_masks[i].bool())).float().reshape(h, w).detach().cpu().numpy(), cmap='hot')
            # Draw subject box (red) for reference
            rect_sub = patches.Rectangle((sub_box[0], sub_box[1]), sub_box[2] - sub_box[0], sub_box[3] - sub_box[1], 
                                       linewidth=2, edgecolor='red', facecolor='none')
            axes3[1, 4].add_patch(rect_sub)
            # Draw object box (blue)
            rect_obj = patches.Rectangle((obj_box[0], obj_box[1]), obj_box[2] - obj_box[0], obj_box[3] - obj_box[1], 
                                       linewidth=2, edgecolor='blue', facecolor='none')
            axes3[1, 4].add_patch(rect_obj)
            axes3[1, 4].set_title(f'Object Pair Mask')
            axes3[1, 4].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'dense_interactiveness_vis/{filename}/{i}/total_interactiveness_analysis.png', dpi=150, bbox_inches='tight')
            plt.close(fig3)
            
    def encode_vision(self, images, attn_mask=None, text_embeddings=None, so_indices=None, meta_data=None):
        """Encode images to vision embeddings"""
        # If images are already tensors (pixel_values), use them directly
        if isinstance(images, torch.Tensor):
            pixel_values = images
        else:
            # Process images using the processor
            processed = self.vision_processor(images=images, return_tensors="pt")
            pixel_values = processed.pixel_values.to(next(self.vision_model.parameters()).device)
        
        kwargs = {}
        out_dict = {}
        # Set interpolate_pos_encoding for SigLIP and CLIP models
        if 'siglip' in self.args.vision_encoder.lower() or 'clip' in self.args.vision_encoder.lower():
            kwargs['interpolate_pos_encoding'] = self.interpolate_positional_embeddings
        # Get vision features
        if self.use_vision_open_clip:
            vision_outputs = self.vision_model.forward_intermediates(pixel_values, intermediates_only=True)
        else:
            vision_outputs = self.vision_model(pixel_values=pixel_values, **kwargs)

        attn_weight = None
        if self.use_attention_pooling:
            # Get all hidden states for attention pooling
            if hasattr(vision_outputs, 'last_hidden_state') or 'image_intermediates' in vision_outputs.keys():
                if hasattr(vision_outputs, 'last_hidden_state'):
                    all_features = vision_outputs.last_hidden_state  # [B, seq_len, hidden_dim]
                    if 'microsoft/resnet' in self.args.vision_encoder.lower():
                        all_features = all_features.flatten(2).transpose(1, 2)
                elif 'image_intermediates' in vision_outputs.keys():
                    all_features = vision_outputs['image_intermediates'][self.args.layer_idx]  # [B, seq_len, hidden_dim]
                    # if 'rn' in self.args.vision_encoder.lower():
                    all_features = all_features.flatten(2).transpose(1, 2)
                
                # Apply vision projector first to all tokens (including CLS)
                projected_features = self.vision_projector(all_features)  # [B, seq_len, attention_pool_dim]
                
                # # For AttentionPool2d, need to reshape to spatial format
                # B, seq_len, attention_dim = projected_features.shape
                
                # feat_size = self.feat_size
                
                # # Exclude prefix tokens and reshape to spatial
                # spatial_features = projected_features[:, -feat_size*feat_size:]  # [B, spatial_tokens, attention_dim]
                # spatial_features = spatial_features.reshape(B, feat_size, feat_size, attention_dim)
                # spatial_features = spatial_features.permute(0, 3, 1, 2)  # [B, attention_dim, H, W]
                
                # Apply AttentionPool2d (outputs to projection_dim)
                if self.args.attention_type != 'ml_decoder' and self.use_det_results:
                    projected_features = projected_features.repeat(attn_mask.shape[0], 1, 1)
                kwargs = {}
                if text_embeddings is not None:
                    kwargs['language_query'] = text_embeddings
                if so_indices is not None:
                    kwargs['so_indices'] = so_indices
                if meta_data is not None:
                    kwargs['meta_data'] = meta_data
                vision_embeddings, attn_weight, extra_out_dict = self.attention_pooling(projected_features, attn_mask=attn_mask, **kwargs)
                out_dict['vision_embeddings'] = vision_embeddings
                out_dict['attn_weight'] = attn_weight
                for k,v in extra_out_dict.items():
                    out_dict[k] = v
                # if 'instance_scores' in extra_out_dict:
                #     out_dict['instance_scores'] = extra_out_dict['instance_scores']
                # if 'instance_to_patch_scores' in extra_out_dict:
                #     out_dict['instance_to_patch_scores'] = extra_out_dict['instance_to_patch_scores']
                # if 'patch_to_instance_scores' in extra_out_dict:
                #     out_dict['patch_to_instance_scores'] = extra_out_dict['patch_to_instance_scores']
                # if 'patch_instance_logits' in extra_out_dict:
                #     out_dict['patch_instance_logits'] = extra_out_dict['patch_instance_logits']
                
            else:
                raise ValueError("Cannot extract last_hidden_state for attention pooling")
        else:
            # Original behavior: extract pooled features first, then project
            if hasattr(vision_outputs, 'pooler_output') and vision_outputs.pooler_output is not None:
                vision_features = vision_outputs.pooler_output
            elif hasattr(vision_outputs, 'last_hidden_state'): 
                # Use CLS token (first token) if no pooler output
                vision_features = vision_outputs.last_hidden_state[:, 0]
            else:
                raise ValueError("Cannot extract vision features from model output")
            
            # Apply vision projector
            vision_embeddings = self.vision_projector(vision_features)
            out_dict['vision_embeddings'] = vision_embeddings
        
        return out_dict
    
    def clip_encode_text(self, texts, normalize: bool = False):
        # https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/model.py
        cast_dtype = self.text_model.get_cast_dtype()

        x = self.clip_token_embedding(texts).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.clip_positional_embedding.to(cast_dtype)
        x = self.text_model(x, attn_mask=self.clip_attn_mask)
        x = self.clip_ln_final(x)  # [batch_size, n_ctx, transformer.width]
        x = text_global_pool(x, texts, self.clip_text_pool_type, eos_token_id=getattr(self, "clip_text_eos_id", None))
        if self.clip_text_projection is not None:
            if isinstance(self.clip_text_projection, nn.Linear):
                x = self.clip_text_projection(x)
            else:
                x = x @ self.clip_text_projection
        return F.normalize(x, dim=-1) if normalize else x
    
    def encode_text(self, texts, precompute=False):
        if self.class_embeddings is not None and not precompute:
            text_features = self.class_embeddings.to(device=next(self.vision_model.parameters()).device).float()
        else:
            """Encode texts to text embeddings"""
            if self.args.freeze_text_encoder or self.args.text_ft_method is not None:
                self.text_model.eval()
            # If texts are already tensors (input_ids), use them directly
            if isinstance(texts, dict) and 'input_ids' in texts:
                input_ids = texts['input_ids'].to(next(self.text_model.parameters()).device)
                attention_mask = texts.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(next(self.text_model.parameters()).device)
            else:
                # Process texts using the processor
                if hasattr(self, 'verb_texts') and not self.args.compute_latency:
                    texts = self.verb_texts
                if 'siglip' in self.args.text_encoder.lower():
                    try:
                        if 'siglip2' in self.args.text_encoder.lower():
                            inputs = self.text_processor(text=texts, return_tensors="pt").to(next(self.text_model.parameters()).device)
                        else:
                            inputs = self.text_processor(text=texts, return_tensors="pt", padding='max_length').to(next(self.text_model.parameters()).device)
                    except:
                        # siglip-so400m-patch14-224 has a maximum position embedding length of 16, requiring text truncation. Other models may avoid this limitation
                        inputs = self.text_processor(text=texts, return_tensors="pt", padding='max_length', truncation=True).to(next(self.text_model.parameters()).device)
                elif 'clip' in self.args.text_encoder.lower():
                    inputs = self.text_processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(next(self.text_model.parameters()).device)
                elif 'nv' in self.args.text_encoder.lower():
                    pass
                elif 'roberta' in self.args.text_encoder.lower():
                    inputs = self.text_processor(text=texts, return_tensors="pt", padding=True).to(next(self.text_model.parameters()).device)
                elif '__pretrained__' in self.args.text_encoder.lower():
                    inputs = self.text_processor(texts).to(next(self.text_model.parameters()).device)
                else:
                    raise ValueError(f"Unsupported text encoder: {self.args.text_encoder}")
            # Get text features
            if 'nv' in self.args.text_encoder.lower():
                text_outputs = self.text_model.encode(texts, max_length=1024)
            elif '__pretrained__' in self.args.text_encoder.lower():
                with torch.cuda.amp.autocast():
                    text_outputs = self.clip_encode_text(inputs)
            else:
                text_outputs = self.text_model(**inputs)
            
            # Extract pooled features
            # if precompute:
            #     if 'clip' in self.args.text_encoder.lower():
            #         text_features = text_outputs.text_embeds
            #     elif 'siglip' in self.args.text_encoder.lower():
            #         text_features = text_outputs.pooler_output
            #     else:
            #         raise ValueError(f"Unsupported text encoder: {self.args.text_encoder}")
            # else:
                # if hasattr(text_outputs, 'pooler_output') and text_outputs.pooler_output is not None:
                #     text_features = text_outputs.pooler_output
                # elif hasattr(text_outputs, 'last_hidden_state'):
                #     # Use CLS token (first token) if no pooler output
                #     text_features = text_outputs.last_hidden_state[:, 0]
                # else:
                #     raise ValueError("Cannot extract text features from model output")
            if 'clip' in self.args.text_encoder.lower():
                text_features = text_outputs.text_embeds
            elif 'siglip' in self.args.text_encoder.lower():
                text_features = text_outputs.pooler_output
            elif 'nv' in self.args.text_encoder.lower():
                text_features = text_outputs.last_hidden_state[:, 0]
            elif 'roberta' in self.args.text_encoder.lower():
                text_features = text_outputs.last_hidden_state[:, 0]
            elif '__pretrained__' in self.args.text_encoder.lower():
                text_features = text_outputs
            else:
                raise ValueError(f"Unsupported text encoder: {self.args.text_encoder}")
            
        # Project features
        if precompute:
            text_features = text_features.float()
            return text_features
        
        text_embeddings = self.text_projector(text_features.float())
        
        return text_embeddings
    
    @profile
    def forward(self, images, texts, attn_mask=None, so_indices=None, meta_data=None):
        """
        Forward pass for vision-text alignment
        
        Args:
            images: Input images (can be PIL images, tensors, etc.)
            texts: Input texts (can be strings, token dicts, etc.)
        
        Returns:
            logits: Alignment logits between images and texts
        """
        out_dict = {}
        
        # Encode vision and text
        kwargs = {'attn_mask': attn_mask, 'so_indices': so_indices, 'meta_data': meta_data}
        
        if self.training:
            # self.text_embeddings = None
            self.precomputed_text_embeddings = None # revised at 260127
            text_embeddings = self.encode_text(texts)
        else:
            if self.precomputed_text_embeddings is not None:
                text_embeddings = self.precomputed_text_embeddings
            else:
                text_embeddings = self.encode_text(texts)
                self.precomputed_text_embeddings = text_embeddings
                
        if self.args.attention_type == 'ml_decoder':
            if self.args.ml_decoder_query_type == 'triplet':
                if self.normalize_embeddings:
                    if self.use_seperate_classifier and self.seperate_normalization:
                        text_dim = text_embeddings.shape[-1]
                        verb_embeddings, object_embeddings = torch.split(text_embeddings, [text_dim//2, text_dim//2], dim=-1) # now support for only same dimensionality for verb and object
                        verb_embeddings = F.normalize(verb_embeddings, p=2, dim=-1)
                        object_embeddings = F.normalize(object_embeddings, p=2, dim=-1)
                        text_embeddings = torch.cat([verb_embeddings, object_embeddings], dim=-1)
                    else:
                        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
                kwargs['text_embeddings'] = text_embeddings
            elif self.args.ml_decoder_query_type == 'object':
                object_embeddings = self.object_embeddings.to(device=next(self.parameters()).device, dtype=text_embeddings.dtype)
                if self.args.normalize_object_embeddings:
                    object_embeddings = F.normalize(object_embeddings, p=2, dim=-1)
                kwargs['text_embeddings'] = object_embeddings
                
        vision_out_dict = self.encode_vision(images, **kwargs)
        vision_embeddings = vision_out_dict['vision_embeddings']
        attn_weight = vision_out_dict['attn_weight'] if 'attn_weight' in vision_out_dict else None
        instance_scores = vision_out_dict['instance_scores'] if 'instance_scores' in vision_out_dict else None
        if self.args.attention_type == 'ml_decoder' and self.args.ml_decoder_query_type == 'triplet':
            # vision embeddings are already logits for language query
            logits = vision_embeddings
            out_dict['test_logits'] = logits
        else:
            # L2 normalize embeddings if specified
            if self.normalize_embeddings:
                if self.use_seperate_classifier and self.seperate_normalization:
                    text_dim = text_embeddings.shape[-1]
                    verb_embeddings, object_embeddings = torch.split(text_embeddings, [text_dim//2, text_dim//2], dim=-1) # now support for only same dimensionality for verb and object
                    verb_embeddings = F.normalize(verb_embeddings, p=2, dim=-1)
                    object_embeddings = F.normalize(object_embeddings, p=2, dim=-1)
                    text_embeddings = torch.cat([verb_embeddings, object_embeddings], dim=-1)
                    
                    vision_dim = vision_embeddings.shape[-1]
                    verb_vision_embeddings, object_vision_embeddings = torch.split(vision_embeddings, [vision_dim//2, vision_dim//2], dim=-1) # now support for only same dimensionality for verb and object
                    verb_vision_embeddings = F.normalize(verb_vision_embeddings, p=2, dim=-1)
                    object_vision_embeddings = F.normalize(object_vision_embeddings, p=2, dim=-1)
                    vision_embeddings = torch.cat([verb_vision_embeddings, object_vision_embeddings], dim=-1)
                    
                else:
                    vision_embeddings = F.normalize(vision_embeddings, p=2, dim=-1)
                    text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
            
            # Compute dot product similarity
            # vision_embeddings: [batch_size, projection_dim]
            # text_embeddings: [batch_size, projection_dim] or [num_texts, projection_dim]
            
            logits = torch.matmul(vision_embeddings, text_embeddings.transpose(-2, -1))
            
            # Apply scale and bias
            logits = logits * self.scale_logit.exp() + self.scale_bias
            
            if self.args.attention_type == 'ml_decoder':
                if self.args.ml_decoder_query_type == 'learnable':
                    if self.args.ml_decoder_mil_type == 'max_logit':
                        logits, _ = torch.max(logits, dim=1)
                    else:
                        raise ValueError(f"Invalid MIL type: {self.args.ml_decoder_mil_type}")
                elif self.args.ml_decoder_query_type == 'object':
                    if instance_scores is not None:
                        if so_indices is not None:
                            if self.args.instance_score_scheme != 'image':
                                if so_indices.shape[0]!=instance_scores.shape[0]:
                                    instance_scores = torch.cat([instance_scores, instance_scores], dim=0)
                                inst_scores = instance_scores[torch.arange(instance_scores.shape[0]), so_indices][None].unsqueeze(-1)
                                # inst_scores = instance_scores[:,so_indices].unsqueeze(-1)
                            else:
                                inst_scores = instance_scores[:,so_indices].unsqueeze(-1)
                            logits = logits.sigmoid() * inst_scores**self.args.instance_prior_factor
                        else:
                            if self.args.use_seperate_interactiveness_loss:
                                out_dict['instance_scores'] = inverse_sigmoid(instance_scores)                                
                                instance_scores = instance_scores.clone().detach() \
                                    if not self.args.no_detach_gradient else instance_scores
                            
                            logits = logits.sigmoid() * instance_scores.unsqueeze(-1)**self.args.instance_prior_factor
                        logits = inverse_sigmoid(logits)
                    
                    if self.args.vis_instance_score:
                        self.visualize_instance_score(images, inst_scores, so_indices, meta_data)
                    if self.args.vis_label_weight:
                        # assert 'instance_to_patch_scores' in vision_out_dict.keys()
                        # self.args.vis_label_weight
                        # if self.args.use_seperate_so:
                        # if  meta_data[0]['filename'] in ['HICO_test2015_00000048.jpg']:
                        instance_to_patch_scores = vision_out_dict['sub_instance_to_patch_scores']
                        patch_to_instance_scores = vision_out_dict['sub_patch_to_instance_scores'] if 'sub_patch_to_instance_scores' in vision_out_dict else None
                            # instance_to_patch_scores = vision_out_dict['obj_instance_to_patch_scores']
                            # patch_to_instance_scores = vision_out_dict['obj_patch_to_instance_scores']
                        # else:
                        #     instance_to_patch_scores = vision_out_dict['instance_to_patch_scores']
                        #     patch_to_instance_scores = vision_out_dict['patch_to_instance_scores'] if 'patch_to_instance_scores' in vision_out_dict else None
                        # self.visualize_label_weight(images, instance_to_patch_scores, instance_scores, patch_to_instance_scores, score_threshold=0.5, smoothing=True, visualize_class_index=0)
                        # self.visualize_label_weight(images, instance_to_patch_scores, instance_scores, patch_to_instance_scores, score_threshold=0.8, visualize_class_index=0)
                        
                        
                        # local and global vis
                        # if  meta_data[0]['filename'] in ['HICO_test2015_00000048.jpg', 'HICO_test2015_00000012.jpg', 'HICO_test2015_00000063.jpg']:
                        # if  meta_data[0]['filename'] in ['HICO_test2015_00000048.jpg']:
                        #     self.visualize_dense_interactiveness(images, vision_out_dict, meta_data, so_indices, threshold=0.3)
                    
                    if self.args.return_attention_weight:
                        self.visualize_attention_weight(images, attn_weight, logits, score_threshold=0.2, draw_all_queries=False)
                        import pdb; pdb.set_trace()
                    
                        
                        
                    if not self.use_det_results and not self.args.compute_latency:
                            
                        out_dict['test_logits'] = logits[:, self.verb_object_indices[1], self.verb_object_indices[0]]
                            
                        if self.args.target_type == 'hoi':
                            logits = logits[:, self.verb_object_indices[1], self.verb_object_indices[0]]
                        
        out_dict['logits'] = logits
        if self.use_det_results:
            return logits
        else:
            return out_dict
            
