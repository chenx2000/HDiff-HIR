"""
Model registry for HSI reconstruction methods.

Supported methods (see Table I in paper):
    - 'hdiff-s': HDiff-HIR-S  {N1,N2,N3}={1,1,2}, ~4.86M params
    - 'hdiff-m': HDiff-HIR-M  {N1,N2,N3}={2,2,2}, ~5.50M params
    - 'hdiff-l': HDiff-HIR-L  {N1,N2,N3}={2,4,6}, ~13.80M params
"""

# Model variant configurations: (num_blocks, condition_blocks)
HDIFF_VARIANTS = {
    'hdiff-s': {'num_blocks': (1, 1, 2), 'condition_blocks': (2, 2)},
    'hdiff-m': {'num_blocks': (2, 2, 2), 'condition_blocks': (2, 2)},
    'hdiff-l': {'num_blocks': (2, 4, 6), 'condition_blocks': (2, 2)},
}


def model_generator(method, pretrained_model_path=None):
    """
    Create a model instance by method name.

    Args:
        method: Method name string (e.g., 'hdiff-s', 'hdiff-m', 'hdiff-l').
        pretrained_model_path: Optional path to pretrained checkpoint.

    Returns:
        nn.Module instance.
    """
    if method in HDIFF_VARIANTS:
        from .unet import UNet
        cfg = HDIFF_VARIANTS[method]
        model = UNet(
            num_blocks=cfg['num_blocks'],
            condition_blocks=cfg['condition_blocks'],
        )
    else:
        raise NotImplementedError(
            f"Method '{method}' is not implemented. "
            f"Available: {list(HDIFF_VARIANTS.keys())}"
        )

    if pretrained_model_path is not None:
        import torch
        checkpoint = torch.load(pretrained_model_path)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(
                {k.replace('module.', ''): v for k, v in checkpoint.items()},
                strict=False,
            )

    return model
