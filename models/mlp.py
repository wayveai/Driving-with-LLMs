from typing import List, Optional, Sequence

import torch
import torch.nn as nn
from einops import rearrange


class MLP(nn.Sequential):
    """
    Multi layer perceptron with sane defaults
    """

    _out_features: int

    def __init__(
        self,
        input_size: int,
        hidden_size: Sequence[int],
        output_size: Optional[int] = None,
        activation: str = "relu",
        norm: Optional[str] = None,
        dropout_rate: float = 0.0,
        output_bias: bool = True,
        output_activation: bool = False,
        pre_norm: bool = False,
        norm_mode: str = "before",
    ):
        layers: List[nn.Module] = []
        size = input_size
        for next_size in hidden_size:
            if (
                norm is not None
                and norm_mode == "before"
                and (len(layers) > 0 or pre_norm)
            ):
                layers.append(NORM_FACTORY[norm](size))
            layers.append(nn.Linear(size, next_size))
            size = next_size
            if norm is not None and norm_mode == "after":
                layers.append(NORM_FACTORY[norm](size))
            layers.append(ACTIVATION_FACTORY[activation]())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))
        if output_size is not None:
            if (
                norm is not None
                and norm_mode == "before"
                and (len(layers) > 0 or pre_norm)
            ):
                layers.append(NORM_FACTORY[norm](size))
            layers.append(nn.Linear(size, output_size, bias=output_bias))
            size = output_size
            if output_activation:
                layers.append(ACTIVATION_FACTORY[activation]())
        super().__init__(*layers)
        self._out_features = size

    def forward(self, x):
        y = x.flatten(0, -2)
        y = super().forward(y)
        y = y.view(x.shape[:-1] + (self._out_features,))
        return y

    @property
    def out_features(self):
        return self._out_features


class Sine(nn.Module):
    """
    Sine activation function,
    read more: https://arxiv.org/abs/2006.09661
    """

    def forward(self, x):
        return torch.sin(x)


class BLCBatchNorm(nn.BatchNorm1d):
    """
    Batch norm that accepts shapes
    (batch, sequence, channel)
    or (batch, channel)
    """

    def forward(self, x):
        if x.dim() == 2:
            return super().forward(x)
        if x.dim() == 3:
            x = rearrange(x, "B L C -> B C L")
            x = super().forward(x)
            x = rearrange(x, "B C L -> B L C")
            return x
        raise ValueError("Only 2d or 3d tensors are supported")


ACTIVATION_FACTORY = {
    "relu": lambda: nn.ReLU(inplace=True),
    "sine": Sine,
    "gelu": nn.GELU,
}


NORM_FACTORY = {"layer_norm": nn.LayerNorm, "batch_norm": BLCBatchNorm}
