
import os
from base import BaseDataLoader

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import SequentialSampler
import torchvision.transforms as tf

from PIL import Image

from typing import Union, Tuple, List

from utils import get_downscaled_size


class NSRRDataLoader(BaseDataLoader):
    """

    """
    def __init__(self,
                 data_dir: str,
                 view_dirname: str,
                 depth_dirname: str,
                 flow_dirname: str,
                 batch_size: int,
                 shuffle: bool = False,
                 validation_split: float = 0.0,
                 num_workers: int = 1,
                 downscale_factor: Union[Tuple[int, int], List[int], int] = (4, 4)
                 ):
        dataset = NSRRDataset(data_dir,
                              view_dirname=view_dirname,
                              depth_dirname=depth_dirname,
                              flow_dirname=flow_dirname,
                              downscale_factor=downscale_factor
                              )
        # sampler = SequentialSampler(dataset)
        super(NSRRDataLoader, self).__init__(dataset=dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             validation_split=validation_split,
                                             num_workers=num_workers,
                                             )


class NSRRDataset(Dataset):
    """
    Requires that corresponding view, depth and motion frames share the same name.
    """
    def __init__(self,
                 data_dir: str,
                 view_dirname: str,
                 depth_dirname: str,
                 flow_dirname: str,
                 downscale_factor: Union[Tuple[int, int], List[int], int] = (4, 4),
                 transform: nn.Module = None,
                 ):
        super(NSRRDataset, self).__init__()

        self.data_dir = data_dir
        self.view_dirname = view_dirname
        self.depth_dirname = depth_dirname
        self.flow_dirname = flow_dirname

        if type(downscale_factor) == int:
            downscale_factor = (downscale_factor, downscale_factor)
        self.downscale_factor = tuple(downscale_factor)

        if transform is None:
            self.transform = tf.ToTensor()
        self.view_listdir = os.listdir(os.path.join(self.data_dir, self.view_dirname))
        self.view_listdir.sort(key=lambda x:int(x.split('.')[0]))
        self.view_listdir.reverse()

    def __getitem__(self, index):
        # view
        #print(index)
        image_name = self.view_listdir[index]
        view_path = os.path.join(self.data_dir, self.view_dirname, image_name)
        depth_path = os.path.join(self.data_dir, self.depth_dirname, image_name)
        flow_path = os.path.join(self.data_dir, self.flow_dirname, image_name)

        trans = self.transform

        img_view_truth = trans(Image.open(view_path))

        downscaled_size = get_downscaled_size(img_view_truth.unsqueeze(0), self.downscale_factor)

        trans_downscale = tf.Resize(downscaled_size)
        trans = tf.Compose([trans_downscale, trans])

        img_view = trans_downscale(img_view_truth)
        # depth data is in a single-channel image.
        img_depth = trans(Image.open(depth_path).convert(mode="L"))
        img_flow = trans(Image.open(flow_path))

        return img_view, img_depth, img_flow, img_view_truth

    def __len__(self) -> int:
        return len(self.view_listdir)

