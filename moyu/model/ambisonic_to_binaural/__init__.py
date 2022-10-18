def get_model_class(model_type):
    r"""Get model.

    Args:
        model_type: str, e.g., 'ResUNet'

    Returns:
        nn.Module
    """
    if model_type == "ResUNet":
        from moyu.model.ambisonic_to_binaural.resunet import ResUNet
        return ResUNet

    elif model_type == "UNet":
        from moyu.model.ambisonic_to_binaural.unet import UNet
        return UNet

    elif model_type == "DNN":
        from moyu.model.ambisonic_to_binaural.others import DNN
        return DNN

    elif model_type == "GRU":
        from moyu.model.ambisonic_to_binaural.others import GRU
        return GRU
    
    else:
        raise NotImplementedError
