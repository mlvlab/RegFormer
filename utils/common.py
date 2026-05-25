import torch

def hard_softmax(logits, dim):
    # code from https://github.com/NVlabs/GroupViT
    y_soft = logits.softmax(dim)
    # Straight through.
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft

    return ret

def gumbel_softmax(logits: torch.Tensor, tau: float = 1, use_straight_through: bool = False, dim: int = -1, is_training=False, gumbel=False) -> torch.Tensor:
    # code from https://github.com/NVlabs/GroupViT
    # _gumbels = (-torch.empty_like(
    #     logits,
    #     memory_format=torch.legacy_contiguous_format).exponential_().log()
    #             )  # ~Gumbel(0,1)
    # more stable https://github.com/pytorch/pytorch/issues/41663
    if is_training and gumbel:
        gumbel_dist = torch.distributions.gumbel.Gumbel(
            torch.tensor(0., device=logits.device, dtype=logits.dtype),
            torch.tensor(1., device=logits.device, dtype=logits.dtype))
        gumbels = gumbel_dist.sample(logits.shape)

        gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
        y_soft = gumbels.softmax(dim)
    else:
        y_soft = (logits/tau).softmax(dim)


    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    if use_straight_through:
        # Straight through.
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_hard
    return ret