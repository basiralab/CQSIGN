from collections import namedtuple
import os
import time

import numpy as np
import torch
from torch.autograd.function import Function
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple
from torch.utils.cpp_extension import load

from conf import config
from utils import get_memory_usage, compute_tensor_bytes, empty_cache, swap_to_cpu
import cpp_extension.quantization as ext_quantization
import cpp_extension.minimax as ext_minimax
import cpp_extension.backward_func as ext_backward_func

QParams = namedtuple('QParams', ['range', 'zero_point', 'num_bits'])


def quantize_and_pack(data, bits, mn, mx):
    if config.simulate:
        N = data.shape[0]
        output = data   # N, groups, group_dim

        if isinstance(bits, int):  # Handle the case when config.adaptive_scheme is False
            bits = torch.ones(N, dtype=torch.int32, device='cuda') * bits

        B = (2 ** bits - 1).view(N, 1, 1)
        mn = mn - 1e-6
        mx = mx + 1e-6
        scale = B / (mx - mn)     # N, groups, 1
        output = (output - mn) * scale

        if config.stochastic:
            noise = output.new(output.shape).uniform_(-0.5, 0.5)
            output.add_(noise)

        output = F.relu(output)
        output = torch.min(output, B.float()).round_().int()
    else:
        # Pack to bitstream
        if isinstance(bits, int):
            pack_func = ext_quantization.pack_single_precision
        else:
            pack_func = ext_quantization.pack_mixed_precision
        output, scale = pack_func(data, mn, mx, bits, config.stochastic)
        if config.swap:
            output = swap_to_cpu(output)

    return output, scale


def dequantize_and_unpack(data, shape, bits, scale, mn):
    if config.simulate:
        data = data / scale + mn
    else:
        if config.swap:
            data = data.cuda(non_blocking=True)

        # Pad to group_size
        N = shape[0]
        num_features = int(np.prod(shape[1:]))
        group_size = config.group_size
        num_features = (num_features + (group_size - num_features % group_size) % group_size)

        # Unpack bitstream
        if isinstance(bits, int):
            unpack_func = ext_quantization.unpack_single_precision
        else:
            unpack_func = ext_quantization.unpack_mixed_precision
        data = unpack_func(data, bits, scale, mn, N, num_features // group_size, group_size)
    return data

def no_scheme_compute_quantization_bits(input):
    N = input.shape[0]
    D = input.shape[1]
    input_flatten = input.view(N, -1)
    num_features = input_flatten.shape[1]
    num_pixels = num_features // D
    
    # Compute min, max by groups
    if num_features % config.group_size != 0:
        # Padding
        new_num_features = (num_features // config.group_size + 1) * config.group_size
        delta = new_num_features - num_features
        input_flatten = torch.cat([input_flatten,
                                   torch.zeros([N, delta], dtype=input.dtype, device=input.device)], 1)

    input_groups = input_flatten.view(-1, config.group_size)
    mn, mx = ext_minimax.minimax(input_groups)

    b = config.activation_compression_bits[0]
    return input_groups.view(N, -1, config.group_size), b, mn.view(N, -1, 1), mx.view(N, -1, 1)


def quantize_activation(input, scheme):
    if not config.compress_activation:
        if config.swap:
            input = swap_to_cpu(input)

        return input, None, None, None

    N = input.shape[0]
    if scheme:
        input_groups, q_bits, q_min, mx = scheme.compute_quantization_bits(input)
    else:
        input_groups, q_bits, q_min, mx = no_scheme_compute_quantization_bits(input)

    q_input, q_scale = quantize_and_pack(input_groups, q_bits, q_min, mx)

    # TODO convert q_bits to int8
    if input.dtype == torch.float32:
        return q_input, q_bits, q_scale.to(torch.bfloat16), q_min.to(torch.bfloat16)
    else:
        return q_input, q_bits, q_scale, q_min


def dequantize_activation(quantized, q_input_shape):
    if not config.compress_activation:
        ret = quantized[0]
        if config.swap:
            ret = ret.cuda(non_blocking=True)
        return ret

    q_input, q_bits, q_scale, q_min = quantized
    if q_scale.dtype == torch.bfloat16:
        q_scale = q_scale.to(torch.float32)
        q_min = q_min.to(torch.float32)
    input = dequantize_and_unpack(q_input, q_input_shape, q_bits, q_scale, q_min)

    # Remove padding
    N = q_input_shape[0]
    num_features = np.prod(q_input_shape[1:])
    input = input.view(N, -1)[:, :num_features]
    input = input.view(*q_input_shape)
    return input.contiguous()

conv2d_layer_ct = 0
bn_layer_ct = 0
total_act_mem = 0

class linear(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, scheme=None):
        quantized = quantize_activation(input, scheme)

        empty_cache(config.empty_cache_threshold)

        ctx.scheme = scheme
        ctx.saved = quantized, weight, bias
        ctx.other_args = input.shape

        return F.linear(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.scheme:
            ctx.scheme.set_scale(grad_output)

        quantized, weight, bias = ctx.saved
        q_input_shape = ctx.other_args

        input = dequantize_activation(quantized, q_input_shape)
        del quantized, ctx.saved

        empty_cache(config.empty_cache_threshold)

        # TODO: the following implementation might not be optimal
        C_in = input.shape[-1]
        C_out = grad_output.shape[-1]
        # rank = len(grad_output.shape)

        grad_output_flatten = grad_output.view(-1, C_out)
        input_flatten = input.view(-1, C_in)
        # print(grad_output_flatten.shape, weight.shape)
        grad_input = grad_output_flatten.mm(weight)
        grad_weight = grad_output_flatten.t().mm(input_flatten)

        # grad_input = grad_output.mm(weight)
        # grad_weight = grad_output.t().mm(input)
        if bias is not None:
            # grad_bias = grad_output.sum(0)
            grad_bias = grad_output_flatten.sum(0)
        else:
            grad_bias = None
            
        del input, grad_output
        return grad_input, grad_weight, grad_bias, None


class batch_norm(Function):
    @staticmethod
    def forward(ctx, input, running_mean, running_var, weight, bias,
                training, exponential_average_factor, eps, scheme):
        # if not ctx.needs_input_grad[3]:
        #     assert not ctx.needs_input_grad[0] and not ctx.needs_input_grad[4]
        #     return ext_backward_func.cudnn_batch_norm(
        #         input, weight, bias, running_mean, running_var, training, exponential_average_factor, eps)[0]
        quantized = quantize_activation(input, scheme)

        empty_cache(config.empty_cache_threshold)

        if config.debug_memory_op_forward:
            global bn_layer_ct, total_act_mem
            print("========== bn forward %d ==========" % bn_layer_ct)
            get_memory_usage(True)
            bn_layer_ct += 1
            total_act_mem += compute_tensor_bytes(quantized)
            print("Act mem: %.2f MB" % (total_act_mem / 1024 ** 2))

        if training:
            output, save_mean, save_var, reserve = ext_backward_func.cudnn_batch_norm(
                input, weight, bias, running_mean, running_var, training, exponential_average_factor, eps)
        else:
            output, save_mean, save_var = ext_backward_func.native_batch_norm(
                input, weight, bias, running_mean, running_var, training, exponential_average_factor, eps)
            reserve = None

        ctx.scheme = scheme
        ctx.other_args = input.shape
        ctx.saved = (quantized, weight, running_mean, running_var, save_mean, save_var, training, eps, reserve)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # if not ctx.needs_input_grad[3]:
        #     assert not ctx.needs_input_grad[0] and not ctx.needs_input_grad[4]
        #     return None, None, None, None, None, None, None, None, None
        quantized, weight, running_mean, running_var, save_mean, save_var, training, eps, reserve = ctx.saved

        q_input_shape = ctx.other_args

        input = dequantize_activation(quantized, q_input_shape)
        del quantized, ctx.saved

        empty_cache(config.empty_cache_threshold)

        if config.debug_memory_op_backward:
            global bn_layer_ct
            print("========== bn backward %d ==========" % bn_layer_ct)
            get_memory_usage(True)
            bn_layer_ct += 1

        if training:
            input = input.contiguous()
            grad_input, grad_weight, grad_bias = ext_backward_func.cudnn_batch_norm_backward(
                input, grad_output, weight, running_mean, running_var, save_mean, save_var, eps, reserve)
        else:
            grad_input, grad_weight, grad_bias = ext_backward_func.native_batch_norm_backward(
                grad_output, input, weight, running_mean, running_var, save_mean, save_var, training, eps,
                [ctx.needs_input_grad[0], ctx.needs_input_grad[3], ctx.needs_input_grad[4]]
            )

        return grad_input, None, None, grad_weight, grad_bias, None, None, None, None

