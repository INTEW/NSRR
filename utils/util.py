import json
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict

import time
import torch
import torch.nn.functional as F
import numpy as np

from threading import Lock, Thread
from typing import Union, Tuple, List

from pytorch_colors import rgb_to_hsv


def backward_warp_motion(img: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
    # see: https://discuss.pytorch.org/t/image-warping-for-backward-flow-using-forward-flow-matrix-optical-flow/99298
    # input image is: [batch, channel, height, width]
    index_batch, number_channels, height, width = img.size()
    grid_x = torch.arange(width).view(1, -1).repeat(height, 1)
    grid_y = torch.arange(height).view(-1, 1).repeat(1, width)
    grid_x = grid_x.view(1, 1, height, width).repeat(index_batch, 1, 1, 1)
    grid_y = grid_y.view(1, 1, height, width).repeat(index_batch, 1, 1, 1)
    ##
    grid = torch.cat((grid_x, grid_y), 1).float()
    # grid is: [batch, channel (2), height, width]
    vgrid = grid + motion
    # Grid values must be normalised positions in [-1, 1]
    vgrid_x = vgrid[:, 0, :, :]
    vgrid_y = vgrid[:, 1, :, :]
    vgrid[:, 0, :, :] = (vgrid_x / width) * 2.0 - 1.0
    vgrid[:, 1, :, :] = (vgrid_y / height) * 2.0 - 1.0
    # swapping grid dimensions in order to match the input of grid_sample.
    # that is: [batch, output_height, output_width, grid_pos (2)]
    vgrid = vgrid.permute((0, 2, 3, 1))
    output = F.grid_sample(img, vgrid, mode='bilinear', align_corners=False)
    return output


def optical_flow_to_motion(rgb_flow: torch.Tensor, sensitivity: float = 0.5) -> torch.Tensor:
    """
    Returns motion vectors as a [batch, 2, height, width]
    with [:, 0, :, :] the abscissa and [:, 1, :, :] the ordinate.
    """
    # flow is: batch x 3-channel x height x width
    # todo: rgb_to_hsv  is extremely slow (from pytorch_color)
    # around 300ms on my machine
    # with [1, 3, 1080, 1920] tensor: single 1920 x 1080 image...
    hsv_flow = rgb_to_hsv(rgb_flow)
    motion_length = hsv_flow[:, 2, :, :] / sensitivity
    motion_angle = (hsv_flow[:, 0, :, :] - 0.5) * (2.0 * np.pi)
    motion_x = - motion_length * torch.cos(motion_angle)
    motion_y = - motion_length * torch.sin(motion_angle)
    motion_x.unsqueeze_(1)
    motion_y.unsqueeze_(1)
    motion = torch.cat((motion_x, motion_y), dim=1)
    # motion is: batch x 2-channel x height x width
    return motion


def upsample_zero_2d(img: torch.Tensor,
                     size: Union[Tuple[int, int], None] = None,
                     scale_factor: Union[Tuple[int, int], List[int], int, None] = None) \
        -> torch.Tensor:
    """
    IMPORTANT: we only support integer scaling factors for now!!
    """
    # input shape is: batch x channels x height x width
    # output shape is:
    if size is not None and scale_factor is not None:
        raise ValueError("Should either define both size and scale_factor!")
    if size is None and scale_factor is None:
        raise ValueError("Should either define size or scale_factor!")
    input_size = torch.tensor(img.size(), dtype=torch.int)
    input_image_size = input_size[2:]
    data_size = input_size[:2]
    if size is None:
        # Get the last two dimensions -> height x width
        # compare to given scale factor
        b_ = np.asarray(scale_factor)
        b = torch.tensor(b_)
        # check that the dimensions of the tuples match.
        if len(input_image_size) != len(b):
            raise ValueError("scale_factor should match input size!")
        output_image_size = (input_image_size * b).type(torch.int)
    else:
        output_image_size = size
    if scale_factor is None:
        scale_factor = output_image_size / input_image_size
    else:
        scale_factor = torch.tensor(np.asarray(scale_factor), dtype=torch.int)
    ##
    output_size = torch.cat((data_size, output_image_size))
    output = torch.zeros(tuple(output_size.tolist()))
    ##
    # todo: use output.view(...) instead.
    output[:, :, ::scale_factor[0], ::scale_factor[1]] = img
    return output


# bit dirty.
def get_scale_factor(a: torch.Tensor, b: torch.Tensor) -> Tuple:
    """
    Computes scaling factor (s_h, s_w) from a to b two 4D or (s_d, s_h, s_w) from 5D tensors of images, returned as a tuple of floats
    """
    # bit dirty.
    return tuple(np.asarray(b.size()[2:]) / np.asarray(a.size()[2:]).astype(int).tolist())


def get_downscaled_size(x: torch.Tensor, downscale_factor: Tuple) -> Tuple:
    """
    Computes (h, w) size of a 4D or (d, h, w) of 5D tensor of images downscaled by a scaling factor (tuple of int).
    """
    return tuple((np.asarray(x.size()[2:]) / np.asarray(downscale_factor)).astype(int).tolist())


class SingletonPattern(type):
    """
    see: https://refactoring.guru/fr/design-patterns/singleton/python/example
    """
    _instances = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]

class Timer:
    """
    see: https://saladtomatonion.com/blog/2014/12/16/mesurer-le-temps-dexecution-de-code-en-python/
    """
    def __init__(self):
        self.start_time = None
        self.interval = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        self.interval = None
        self.start_time = time.time()

    def stop(self):
        if self.start_time is not None:
            self.interval = time.time() - self.start_time
            self.start_time = None

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
