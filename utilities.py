from functools import reduce

import torch
from math import floor
from torch import nn

import numpy as np
import operator as op
import torch.nn.functional as F


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)

    from https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/5
    """
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
    w = floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
    return h, w


def calc_params(model):
    # from torch forums: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/8
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    return num_params


def conv2d_factory(input_shape, out_channels=None, kernel_size=3, stride=1, padding=1,
                   dilation=1, groups=1, bias=True, filter_multiplier=1):
    """
    Factory to create a convolutional layer from an input shape and return statistics about the layer
    Args:
        input_shape: the 3-d input shape (channels, height, width)
        out_channels: The number of output channels. if left out, multiply by filter_multiplier to determine the next output channels
        kernel_size (int or tuple) – Size of the convolving kernel
        stride (int or tuple, optional) – Stride of the convolution. Default: 1
        padding (int or tuple, optional) – Zero-padding added to both sides of the input. Default: 1 for same
        dilation (int or tuple, optional) – Spacing between kernel elements. Default: 1
        groups (int, optional) – Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional) – If True, adds a learnable bias to the output. Default: True
        filter_multiplier: how much to multiply the input filters by to find the output filters. Default: 1 (same)

    Returns:

    """
    # channels first
    in_channels = input_shape[0]
    if out_channels is None:
        out_channels = int(in_channels * filter_multiplier)
    height_width = input_shape[1:]
    output_shape = conv_output_shape(height_width, kernel_size, stride, padding, dilation)
    output_shape = (out_channels, *output_shape)
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    num_params = calc_params(conv)

    return conv, output_shape, num_params


class GenericConvolutionalEncoder(nn.Module):
    def __init__(self, input_shape, max_final_width=8, activation=F.relu):
        """

        Args:
            input_shape: a 3-element tuple with the number of channels, the height, and the width. (channels first)
            max_final_width: the maximum desired final width of the network.
            activation: The activation function used by each layer
        """
        super(GenericConvolutionalEncoder, self).__init__()

        next_shape = input_shape
        self.activation = activation
        self.output_shape = next_shape

        self.conv_layers = nn.ModuleList([])
        # add filters with stride until our output size is less than 6
        while min(next_shape[1:]) >= max_final_width:
            conv, next_shape, layer_params = conv2d_factory(next_shape, stride=2,
                                                            padding=1, filter_multiplier=2)
            self.conv_layers.append(conv)
            self.output_shape = next_shape

    def forward(self, x):
        for layer in self.conv_layers:
            x = self.activation(layer(x))
        return x


class GenericFullyConnected(nn.Module):
    def __init__(self, input_shape, shape, depth, activation=F.relu):
        """

        Args:
            input_shape: a 3-element tuple with the number of channels, the height, and the width.
            max_final_width: the maximum desired final width of the network.
            activation: The activation function used by each layer
        """
        super(GenericFullyConnected, self).__init__()

        self.activation = activation

        self.layers = nn.ModuleList([])
        for i in range(depth):
            layer = nn.Linear(input_shape, shape)
            self.layers.append(layer)
            input_shape = shape

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return x


def flat_shape(shape):
    return np.prod(shape)


def flatten(mod: nn.Module):
    return mod.view(mod.size(0), -1)


def unflatten(mod: nn.Module):
    return mod.view(mod.size(0), mod.size(1), 1, 1)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def format_screen(screen, device):
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return screen.unsqueeze(0).to(device)


def detect_channel_index(input_shape):
    dimensions = len(input_shape)
    if dimensions == 4:
        # batch shape or video, we're doomed
        raise NotImplementedError
    elif dimensions == 3:
        # we can actually do this shape
        # find index of min size
        index = np.argmin(input_shape)
        return index


def to_torch_channels(array: np.ndarray):
    if array is not None and len(array.shape) > 2:
        return np.moveaxis(array, 0, -1)
    else:
        return array


def to_batch_shape(array: np.ndarray):
    return np.reshape(array, (1, *array.shape))


def image_batch_to_device_and_format(screen_batch, device, dtype=torch.float32):
    """
    Convert a batch of images from torch.uint8 to the correct format for the net
    Args:
        screen_batch:
        device:
        dtype:

    Returns:

    """
    # Convert to float, rescale, convert to torch tensor
    # transfer it as a uint8 to save bandwidth
    screen = torch.from_numpy(screen_batch).to(device, dtype=dtype) / 255
    return screen.to(device)


def get_tensors(gpu_only=True):
    import gc
    for obj in gc.get_objects():
        # noinspection PyBroadException
        try:
            if torch.is_tensor(obj):
                tensor = obj
            elif hasattr(obj, 'data') and torch.is_tensor(obj.data):
                tensor = obj.data
            else:
                continue

            if tensor.is_cuda:
                yield tensor
        except:
            pass


def log_tensors(logger):
    tensor_data_list = []
    tensor_count_by_size = {}
    total_size = 0
    for tensor in get_tensors():
        tensor_tuple = (reduce(op.mul, tensor.size())
                        if len(tensor.size()) > 0 else 0,
                        type(tensor), tensor.size())
        tensor_data_list.append(tensor_tuple)
        tensor_size = tensor_tuple[0]
        total_size += tensor_size
        if tensor_size in tensor_count_by_size:
            tensor_count_by_size[tensor_size] += 1
        else:
            tensor_count_by_size[tensor_size] = 1
    logger.debug(tensor_data_list)
    logger.debug(len(tensor_data_list))
    logger.debug(tensor_count_by_size)
    logger.debug("Total Size: %s" % total_size)

