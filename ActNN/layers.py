# The code is compatible with PyTorch 1.6/1.7
from typing import List, Optional
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed
from torch import Tensor

from qscheme import QScheme
from qbnscheme import QBNScheme
from conf import config
from ops import linear, batch_norm
import cpp_extension.quantization as ext_quantization

class QLinear(nn.Linear):
    num_layers = 0

    def __init__(self, input_features, output_features, bias=True, group=0):
        super(QLinear, self).__init__(input_features, output_features, bias)
        if config.adaptive_conv_scheme:
            self.scheme = QScheme(self, group=group)
        else:
            self.scheme = None

    def forward(self, input):
        if config.training:
            return linear.apply(input, self.weight, self.bias, self.scheme)
        else:
            return super(QLinear, self).forward(input)


class QBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, group=0):
        super(QBatchNorm1d, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        if config.adaptive_bn_scheme:
            self.scheme = QBNScheme(group=group)
        else:
            self.scheme = None

    def forward(self, input):
        if not config.training:
            return super(QBatchNorm1d, self).forward(input)

        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        """ Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        """Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        return batch_norm.apply(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight, self.bias, bn_training, exponential_average_factor, self.eps, self.scheme)

class QReLU(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return ext_quantization.act_quantized_relu(input)


class QDropout(nn.Dropout):
    def __init__(self, p=0.5):
        super().__init__(p=p)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            return ext_quantization.act_quantized_dropout(input, self.p)
        else:
            return super(QDropout, self).forward(input)
