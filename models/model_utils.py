import torch
from typing import Optional

CLIP_VISION_HIDDEN_DIMS = {
    # OpenAI CLIP
    "RN50": 2048,
    "RN101": 2048,
    "RN50x4": 2560,
    "RN50x16": 3072,
    "RN50x64": 4096,
    "ViT-B-32": 768,
    "ViT-B-16": 768,
    "ViT-L-14": 1024,
    "ViT-L-14-336": 1024,


    # OpenCLIP public (similar naming)
    "openai/clip-vit-base-patch32": 768,
    "openai/clip-vit-base-patch16": 768,
    "openai/clip-vit-large-patch14": 1024,
    "openai/clip-vit-large-patch14-336": 1024,
    # "openai/clip-rn50": 1024,
    # "openai/clip-rn101": 512,

    # LAION CLIP-like
    "laion/clip-convnext_base_w": 1024,
    "laion/clip-convnext_base_d": 1024,
    "laion/clip-convnext_large_d": 1536,
    "laion/clip-convnext_xxlarge_d_320": 2048,

    # SigLIP
    "ViT-B-16-SigLIP-512": 768,
    "ViT-L-16-SigLIP-512": 1024,
    "ViT-B-16-SigLIP": 768,
    "ViT-L-16-SigLIP": 1024,
    
    "microsoft/resnet-50": 2048,
    "microsoft/resnet-101": 2048,
}

def get_clip_vision_hidden_dim(model_name: str):
    """
    Return the vision encoder hidden dimension for a given CLIP/SigLIP model name.

    Args:
        model_name (str): Model identifier key.

    Returns:
        int: The hidden dim. Raises KeyError if unknown.
    """
    if model_name not in CLIP_VISION_HIDDEN_DIMS:
        raise ValueError(f"Unknown or unsupported CLIP model name: {model_name}")
    return CLIP_VISION_HIDDEN_DIMS[model_name]

def text_global_pool(
        x: torch.Tensor,
        text: Optional[torch.Tensor] = None,
        pool_type: str = 'argmax',
        eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    if pool_type == 'first':
        pooled = x[:, 0]
    elif pool_type == 'last':
        pooled = x[:, -1]
    elif pool_type == 'argmax':
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        assert text is not None
        pooled = x[torch.arange(x.shape[0], device=x.device), text.argmax(dim=-1)]
    elif pool_type == 'eos':
        # take features from tokenizer specific eos
        assert text is not None
        assert eos_token_id is not None
        idx = (text == eos_token_id).int().argmax(dim=-1)
        pooled = x[torch.arange(x.shape[0], device=x.device), idx]
    else:
        pooled = x

    return pooled