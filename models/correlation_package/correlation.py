"""Module providing GPU-accelerated correlation and its autograd function."""

import torch
from torch.autograd import Function
from torch.nn.modules.module import Module

from . import correlation_cuda


class CorrelationFunction(Function):
    """New-style autograd Function for computing correlation with CUDA acceleration."""

    @staticmethod
    def forward(ctx, input1, input2, pad_size, kernel_size,
                max_displacement, stride1, stride2, corr_multiply):
        """Compute forward correlation and save context for backward."""
        ctx.pad_size = pad_size
        ctx.kernel_size = kernel_size
        ctx.max_displacement = max_displacement
        ctx.stride1 = stride1
        ctx.stride2 = stride2
        ctx.corr_multiply = corr_multiply

        ctx.save_for_backward(input1, input2)

        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()
            output = input1.new()

            correlation_cuda.forward(
                input1, input2,
                rbot1, rbot2, output,
                pad_size, kernel_size,
                max_displacement, stride1,
                stride2, corr_multiply
            )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Compute gradients of correlation with respect to inputs."""
        input1, input2 = ctx.saved_tensors
        pad_size = ctx.pad_size
        kernel_size = ctx.kernel_size
        max_displacement = ctx.max_displacement
        stride1 = ctx.stride1
        stride2 = ctx.stride2
        corr_multiply = ctx.corr_multiply

        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()
            grad_input1 = input1.new()
            grad_input2 = input2.new()

            correlation_cuda.backward(
                input1, input2,
                rbot1, rbot2,
                grad_output, grad_input1,
                grad_input2,
                pad_size, kernel_size,
                max_displacement, stride1,
                stride2, corr_multiply
            )

        return grad_input1, grad_input2, None, None, None, None, None, None


class Correlation(Module):
    """PyTorch module wrapping the CorrelationFunction."""

    def __init__(self, pad_size=0, kernel_size=0,
                 max_displacement=0, stride1=1,
                 stride2=2, corr_multiply=1):
        super().__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def forward(self, input1, input2):
        """Apply correlation to two feature maps."""
        return CorrelationFunction.apply(
            input1, input2,
            self.pad_size,
            self.kernel_size,
            self.max_displacement,
            self.stride1,
            self.stride2,
            self.corr_multiply
        )
