import torch
import torch.nn as nn
import math


def init_gru(rnn):
    """Initialize a GRU layer. """

    def _concat_init(tensor, init_funcs):
        (length, fan_out) = tensor.shape
        fan_in = length // len(init_funcs)

        for (i, init_func) in enumerate(init_funcs):
            init_func(tensor[i * fan_in : (i + 1) * fan_in, :])

    def _inner_uniform(tensor):
        fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
        nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))

    for i in range(rnn.num_layers):
        _concat_init(
            getattr(rnn, 'weight_ih_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, _inner_uniform]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_ih_l{}'.format(i)), 0)

        _concat_init(
            getattr(rnn, 'weight_hh_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, nn.init.orthogonal_]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_hh_l{}'.format(i)), 0)


def init_linear_or_conv(layer):
    if hasattr(layer, "bias") and layer.bias is not None:
        nn.init.constant_(layer.bias, 0)
    nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))


def init_module(module):
    if isinstance(module, nn.Linear):
        init_linear_or_conv(module)

    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        init_linear_or_conv(module)

    if isinstance(module, nn.GRU):
        init_gru(module)
