import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

hrir_path = "./trivial/hrir/hrir.wav"

hrir, _ = sf.read(hrir_path)


def decode_ambisonic_to_binaural(ambisonic):
    """Decode a first order of ambisonic to binaural signals.

    Args:
        ambisonic (_type_): (N, 4, L) tensor
    """
    # the origin SP_HRIR provided by SAMI is (kernel_size, input_channels)=(256, 4) 
    # we append a extra 0 to the kernel_size, such that the kernel_size is odd
    # the processed sp_hrir for torch is (1, 4, 257)
    sp_hrir = torch.tensor(hrir)
    sp_hrir = torch.cat([sp_hrir, torch.zeros((1, 4))], dim=0).T[None, ...]
    
    left = F.conv1d(ambisonic, sp_hrir, bias=None, stride=1, padding=128)
    
    ambisonic[:, 1] = -ambisonic[:, 1]
    right = F.conv1d(ambisonic, sp_hrir, bias=None, stride=1, padding=128)
    return torch.cat([left, right], dim=1)


if __name__ == "__main__":
    # input_path = "resource/360 Workstation Ambisonic Panning AmbiX.wav"
    # input, sr = sf.read(input_path)
    # print("sample rate: {}".format(sr))
    input = np.random.randn(53061843, 4)
    output = trivial_binaural_rendering(input)
    print(output.shape)

    # sf.write("./result/test/trival_output.wav", data=output, samplerate=sr)