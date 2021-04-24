import torch
import numbers
from torch.nn.parameter import Parameter
from torch.nn import Module
from torch.nn import functional as F
from torch.nn import init
from torch import nn

from torch import Tensor, Size
from typing import Union, List
_shape_t = Union[int, List[int], Size]

class ConditionalLayerNorm(Module):
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: _shape_t
    eps: float
    elementwise_affine: bool

    def __init__(self, normalized_shape: _shape_t, conditional_size: int ,eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        super(ConditionalLayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.Tensor(*normalized_shape))
            self.bias = Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.conditional_size = conditional_size
        self.weight_dense = nn.Linear(conditional_size,self.normalized_shape[0],bias=False)
        self.bias_dense = nn.Linear(conditional_size, self.normalized_shape[0],bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)
            init.zeros_(self.weight_dense.weight)
            init.zeros_(self.bias_dense.weight) 

    def forward(self, input: Tensor,conditional: Tensor) -> Tensor:
        conditional = torch.unsqueeze(conditional, 1)
        add_weight =  self.weight_dense(conditional)
        add_bias = self.bias_dense(conditional)
        weight = self.weight + add_weight
        bias = self.bias + add_bias
        outputs = input
        mean = torch.mean(outputs, axis=-1, keepdims=True)
        outputs = outputs - mean
        variance = torch.mean(torch.square(outputs), axis=-1, keepdims=True)
        std = torch.sqrt(variance + self.eps)
        outputs = outputs / std
        outputs = outputs * weight
        outputs = outputs + bias
        return outputs
