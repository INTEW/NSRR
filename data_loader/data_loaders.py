import imp
import os
from base import BaseDataLoader
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import SequentialSampler
# import torchvision.transforms as tf
import torchvision as tv
from torchvision import transforms
from torchvision.transforms import functional
from utils import optical_flow_to_motion

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
        ## TODO: move `idx_to_seq` to the input param of NSRRDataset
        self.idx_to_seq = lambda x: [x+i for i in range(6)]
        if type(downscale_factor) == int:
            downscale_factor = (downscale_factor, downscale_factor)
        self.downscale_factor = tuple(downscale_factor)
        if transform is None:
            self.transform = tv.transforms.ToTensor()
        self.view_listdir = os.listdir(os.path.join(self.data_dir, self.view_dirname))
        self.view_listdir.sort(key=lambda x:int(x.split('.')[0]))
        self.view_listdir.reverse()
        print(self.view_listdir)
    def __getitem__(self, index):
        seq_list = self.idx_to_seq(index)
        ## TODO: multiprocess to read img
        img_view_list, img_depth_list, img_flow_list, img_view_truth_list = [], [], [], []
        for i in seq_list:
            image_name = self.view_listdir[i]
            view_path = os.path.join(self.data_dir, self.view_dirname, image_name)
            depth_path = os.path.join(self.data_dir, self.depth_dirname, image_name)
            flow_path = os.path.join(self.data_dir, self.flow_dirname, image_name)
            # print(f"view_path = {view_path}\ndepth_path = {depth_path}\nflow_path = {flow_path}")
            # trans = self.transform
            # img_view_truth = trans(Image.open(view_path))
            img_view_truth = tv.io.read_image(view_path, mode=tv.io.ImageReadMode.RGB) / 255
            img_depth = tv.io.read_image(depth_path, mode=tv.io.ImageReadMode.RGB)[0, :, :].unsqueeze(0) / 255
            img_flow = tv.io.read_image(flow_path, mode=tv.io.ImageReadMode.RGB) / 255
            C, H, W = img_view_truth.shape
            HH = int(H / self.downscale_factor[0])
            WW = int(W / self.downscale_factor[1])
            # downscaled_size = get_downscaled_size(img_view_truth.unsqueeze(0), self.downscale_factor)
            # trans = tf.Compose([trans_downscale, trans])
            img_view = tv.transforms.functional.resize(img_view_truth, [HH, WW])
            img_depth = tv.transforms.functional.resize(img_depth, [HH, WW])
            img_flow = tv.transforms.functional.resize(img_flow, [HH, WW])
            img_flow.unsqueeze_(0)
            img_flow = optical_flow_to_motion(img_flow)
            img_flow.squeeze_(0)
            img_view_list.append(img_view.unsqueeze(1))
            img_depth_list.append(img_depth.unsqueeze(1))
            img_flow_list.append(img_flow.unsqueeze(1))
            img_view_truth_list.append(img_view_truth.unsqueeze(1))
        img_view = torch.cat(img_view_list, dim=1)
        img_depth = torch.cat(img_depth_list, dim=1)
        img_flow = torch.cat(img_flow_list, dim=1)
        img_view_truth = torch.cat(img_view_truth_list, dim=1)
        # print("img_view.shape = ", img_view.shape)
        # print("img_depth.shape = ", img_depth.shape)
        # print("img_flow.shape = ", img_flow.shape)
        # print("img_view_truth.shape = ", img_view_truth.shape)
        return img_view, img_depth, img_flow, img_view_truth

    def __len__(self) -> int:
        return len(self.view_listdir) - 6 + 1

