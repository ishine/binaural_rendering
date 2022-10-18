from typing import Callable


def get_loss_function(loss_type: str) -> Callable:
    r"""Get loss function.

    Args:
        loss_type: str

    Returns:
        loss function: Callable
    """

    if loss_type == "l1_wav":
        from moyu.loss.math import l1
        return l1

    elif loss_type == "l2_wav":
        from moyu.loss.math import l2
        return l2
    
    elif loss_type == "l1_sp":
        from moyu.loss.wave import L1_sp
        return L1_sp()

    elif loss_type == "l1_wav_l1_sp":
        from moyu.loss.wave import L1_Wav_L1_Sp
        return L1_Wav_L1_Sp()

    elif loss_type == "l2_wav_l2_sp":
        from moyu.loss.wave import L2_Wav_L2_Sp
        return L2_Wav_L2_Sp()
    else:
        raise NotImplementedError("Loss type: {}".format(loss_type))