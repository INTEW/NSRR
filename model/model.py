from numpy import pad
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import pytorch_colors as colors
from torchvision import utils as vutils

from typing import Union, List, Tuple, Callable, Any

from utils import upsample_zero_2d, backward_warp_motion, optical_flow_to_motion

class NSRRModel(BaseModel):
    def __init__(self):
        super(NSRRModel, self).__init__()
        # [N, C, D, H, W]
        # [N, C, H, W]
        self.kernel_size = (3, 3)
        self.padding = (1, 1)
        self.scale_factor = (4, 4)
        self.flow_sensitivity = 0.5
        self.number_previous_frames = 5
        self.UpsamplingBilinear2d = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
        self.featureExtractSeq = nn.ModuleList([nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=self.kernel_size, padding=self.padding),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=self.kernel_size, padding=self.padding),
            nn.ReLU(),
            nn.Conv2d(32, 8, kernel_size=self.kernel_size, padding=self.padding),
        ) for _ in range(1+self.number_previous_frames)])
        self.feature_reweighting_model = NSRRFeatureReweightingModel()
        self.reconstructionModel = NSRRReconstructionModel()
    def feature_extract_model(self, rgbd):
        # rgbd: [B, C=3+1, D=6, W, H]
        rgbd_out = list(None for _ in range(1+self.number_previous_frames))
        for i in range(1+self.number_previous_frames):
            rgbd_out[i] = self.featureExtractSeq[i](rgbd[:, :, i, :, :]).unsqueeze_(dim=2)
        rgbd_out = torch.cat(rgbd_out, dim=2)
        # return: [B, C=12, D=6, W, H]
        return torch.cat((rgbd, rgbd_out), dim=1)
    def zero_upsampling_function(self, x):
        B, C, D, H, W = x.shape
        x_out = [None for i in range(D)]
        for i in range(D):
            # we don't use zero upsample, instead, we use interpolate upsample because Pytorch has API
            x_out[i] = nn.functional.upsample(x[:, :, i, :, :], scale_factor=self.scale_factor).unsqueeze(2)
        x_out = torch.cat(x_out, dim=2)
        return x_out
    def motion_upsampling_function(self, x):
        B, C, D, H, W = x.shape
        x_out = [None for _ in range(D)]
        for i in range(D):
            x_out[i] = self.UpsamplingBilinear2d(x[:, :, i, :, :]).unsqueeze_(dim=2)
        x_out = torch.cat(x_out, dim=2)
        return x_out

    def motion_warping_function(self, img, motion):
        return backward_warp_motion(img, motion)
    def forward(self, x_view, x_depth, x_flow):
        # 1°) extract features
        all_rgbd = torch.cat((x_view, x_depth), dim=1)
        all_features = self.feature_extract_model(all_rgbd)

        # feature.shape =  torch.Size([1, 12, 6, 120, 180])
        # 2°) upsample features
        all_features_upsampled = self.zero_upsampling_function(all_features)
        all_features_upsampled_for_reweighting = self.zero_upsampling_function(all_rgbd)
        
        # 3°) we need to convert from optical flow
        all_flow_upsampled = self.motion_upsampling_function(x_flow)

        # 4°) warp previous features and motion recursively
        list_previous_motion_from_current = [all_flow_upsampled[:, :, 0, :, :]]
        # back warp motion
        for i in range(1, self.number_previous_frames):
            list_previous_motion_from_current.append(
                self.motion_warping_function(
                    all_flow_upsampled[:,:,i,:,:],
                    list_previous_motion_from_current[-1]
                )
            )

        # back warp feature
        list_previous_features_warped = [] # nn.paramList
        for i in range(self.number_previous_frames):
            list_previous_features_warped.append(
                self.motion_warping_function(
                    all_features_upsampled[:,:,i,:,:],
                    list_previous_motion_from_current[i]
                )
            )

        # 5°) reweight features of previous frames
        list_previous_features_reweighted = self.feature_reweighting_model.forward(
            all_features_upsampled_for_reweighting[:,:,0,:,:],
            list_previous_features_warped
        )

        # 6°) reconstruction 
        target = self.reconstructionModel.forward(all_features_upsampled[:,:,0,:,:], list_previous_features_reweighted)
        return target

class NSRRFeatureExtractionModel(BaseModel):
    """
    """

    def __init__(self):
        super(NSRRFeatureExtractionModel, self).__init__()
        kernel_size = 3
        # Adding padding here so that we do not lose width or height because of the convolutions.
        # The input and output must have the same image dimensions so that we may concatenate them
        padding = 1
        process_seq = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(32, 8, kernel_size=kernel_size, padding=padding),
            nn.ReLU()
        )
        self.add_module("featuring", process_seq)

    def forward(self, colour_images: torch.Tensor, depth_images: torch.Tensor) -> torch.Tensor:
        # From each 3-channel image and 1-channel image, we construct a 4-channel input for our model.
        x = torch.cat((colour_images, depth_images), dim=1)
        x_features = self.featuring(x)
        # We concatenate the original input that 'skipped' the network.
        x = torch.cat((x, x_features), dim=1)
        return x


class NSRRFeatureReweightingModel(BaseModel):
    """
    """

    def __init__(self):
        super(NSRRFeatureReweightingModel, self).__init__()
        # According to the paper, rescaling in [0, 10] after the final tanh activation
        # gives accurate enough results.
        self.scale = 10
        kernel_size = 3
        padding = 1

        process_seq = nn.Sequential(
            nn.Conv2d(24, 40, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(40, 40, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(40, 5, kernel_size=kernel_size, padding=padding),
            nn.Tanh(),
            Remap([-1, 1], [0, self.scale])
        )
        self.add_module("weighting", process_seq)

    def forward(self,
                current_features_upsampled_for_reweighting:
                    torch.Tensor,
                list_previous_features_warped:
                    List[torch.Tensor]) -> List[torch.Tensor]:
        # Generating a pixel-wise weighting map for the current and each previous frames.
        # current_feature_upsampled[:, :4] are the RGBD of the current frame (upsampled).
        # previous_features_warped [:, :4] are the RGBD of the previous frame (upsampled then warped).
        # TODO cache the results

        # current_features_upsampled_for_reweighting = [7, 4, 480, 720]
        # list_previous_features_warped = [7, 5, 480, 720] * 5
        reweight_feed_in = current_features_upsampled_for_reweighting
        for previous_features_warped in list_previous_features_warped:
            reweight_feed_in = torch.cat((reweight_feed_in, previous_features_warped[:,:4]), dim=1)
        
        weighting_map = self.weighting(reweight_feed_in)
        # weighting_map = [7, 5, 480, 720]
        list_previous_features_reweighted = []
        for i in range(5):
            tmp = list_previous_features_warped[i]
            tmp2 = [None for _ in range(12)]
            for j in range(12):
                # print(f"tmp[j].shape = {tmp[:, j, :, :].shape}, weighting_map[i].shape = {weighting_map[:, i, :, :].shape}")
                tmp2[j] = torch.mul(tmp[:, j, :, :], weighting_map[:, i, :, :])
            
            result_list_previous = tmp2[0].unsqueeze(1)
            # result_list_previous = [7, 1, 480, 720]
            for j in range(1, 12):
                result_list_previous = torch.cat((result_list_previous, tmp2[j].unsqueeze(1)), dim=1)
            result_list_previous_final = result_list_previous
            list_previous_features_reweighted.append(result_list_previous_final)

        return list_previous_features_reweighted


class NSRRReconstructionModel(BaseModel):
    """
    Reconstruction Model based on U-Net structure
    https://arxiv.org/pdf/1505.04597.pdf https://github.com/usuyama/pytorch-unet/blob/master/pytorch_unet.py
    """

    def __init__(self, number_previous_frames: int = 5):
        super(NSRRReconstructionModel, self).__init__()
        padding = 1
        kernel_size = 3
        self.pooling = nn.MaxPool2d(2)

        assert(number_previous_frames > 0)
        # This is constant throughout the life of a model.
        self.number_previous_frames = number_previous_frames
        encoder1 = nn.Sequential(
            nn.Conv2d(72, 32, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
        )
        encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=kernel_size, padding=padding),
            nn.ReLU()
        )
        center = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        cat_1 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=kernel_size, padding=padding),
            nn.ReLU()
        )
        cat_2 = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=kernel_size, padding=padding),
            nn.ReLU()
        )
        decoder2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        decoder1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=kernel_size, padding=padding),
            nn.ReLU()
        )

        self.add_module("encoder_1", encoder1)
        self.add_module("encoder_2", encoder2)
        self.add_module("center", center)
        self.add_module("decoder_2", decoder2)
        self.add_module("decoder_1", decoder1)
        self.add_module("cat_1", cat_1)
        self.add_module("cat_2", cat_2)

    def crop_tensor(self, target: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        # https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
        diffY = actual.size()[2] - target.size()[2]
        diffX = actual.size()[3] - target.size()[3]
        x = F.pad(target, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        return x

    def forward(self, current_features: torch.Tensor,list_previous_features_reweighted: [torch.Tensor]) -> torch.Tensor:
        # Features of the current frame and the reweighted features
        # of previous frames are concatenated
        x = torch.cat((current_features, *list_previous_features_reweighted), 1)
        # Cache result to handle 'skipped' connection for encoder 1 & 2
        x_encoder_1 = self.encoder_1(x)
        x = self.pooling(x_encoder_1)
        x_encoder_2 = self.encoder_2(x)
        x = self.pooling(x_encoder_2)
        x = self.center(x)

        # Crop the skipped image to target dimension and concatenate for encoder 1 & 2
        #x_encoder_2 = self.crop_tensor(x_encoder_2, x)
        # (128, 64) = (192)
        # print("x",  x.shape)
        # print("x_encoder_2", x_encoder_2.shape)
        x = torch.cat((x, x_encoder_2), 1)
        x = self.cat_1(x)
        x = self.decoder_2(x)
        #x_encoder_1 = self.crop_tensor(x_encoder_1, x)
        x = torch.cat((x, x_encoder_1), 1)
        x = self.cat_2(x)
        x = self.decoder_1(x)
        return x


class Remap(BaseModel):
    """
    Basic layer for element-wise remapping of values from one range to another.
    """

    in_range: Tuple[float, float]
    out_range: Tuple[float, float]

    def __init__(self,
                 in_range: Union[Tuple[float, float], List[float]],
                 out_range: Union[Tuple[float, float], List[float]]
                 ):
        assert(len(in_range) == len(out_range) and len(in_range) == 2)
        super(BaseModel, self).__init__()
        self.in_range = tuple(in_range)
        self.out_range = tuple(out_range)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.div(
            torch.mul(torch.add(x, - self.in_range[0]), self.out_range[1] - self.out_range[0]),
            (self.in_range[1] - self.in_range[0]) + self.out_range[0])


class ZeroUpsample2D(BaseModel):
    """
    Basic layer for zero-upsampling of 2D images (4D tensors).
    """

    scale_factor: Tuple[int, int]

    def __init__(self, scale_factor: Union[Tuple[int, int], List[int], int]):
        super(ZeroUpsample2D, self).__init__()
        if type(scale_factor) == int:
            scale_factor = (scale_factor, scale_factor)
        assert(len(scale_factor) == 2)
        self.scale_factor = tuple(scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return upsample_zero_2d(x, scale_factor=self.scale_factor)


class BackwardWarp(BaseModel):
    """
    A model for backward warping 2D image tensors according to motion tensors.
    """

    def __init__(self):
        super(BackwardWarp, self).__init__()

    def forward(self, x_image: torch.Tensor, x_motion: torch.Tensor) -> torch.Tensor:
        return backward_warp_motion(x_image, x_motion)


class OpticalFlowToMotion(BaseModel):
    """
    A model for computing optical flow to motion vectors conversion, in RGB image tensors.
    """

    sensitivity: float

    def __init__(self, sensitivity: float):
        super(BaseModel, self).__init__()
        self.sensitivity = sensitivity

    def forward(self, x_flow_rgb) -> torch.Tensor:
        return optical_flow_to_motion(x_flow_rgb, self.sensitivity)


class LayerOutputModelDecorator(BaseModel):
    """
    A Decorator for a Model to output the output from an arbitrary set of layers.
    """

    def __init__(self, model: nn.Module, layer_predicate: Callable[[str, nn.Module], bool]):
        super(LayerOutputModelDecorator, self).__init__()
        self.model = model
        self.layer_predicate = layer_predicate

        self.output_layers = []

        def _layer_forward_func(layer_index: int) -> Callable[[nn.Module, Any, Any], None]:
            def _layer_hook(module_: nn.Module, input_, output) -> None:
                self.output_layers[layer_index] = output
            return _layer_hook
        self.layer_forward_func = _layer_forward_func

        for name, module in self.model.named_children():
            if self.layer_predicate(name, module):
                module.register_forward_hook(
                    self.layer_forward_func(len(self.output_layers)))
                self.output_layers.append(torch.Tensor())

    def forward(self, x) -> List[torch.Tensor]:
        self.model(x)
        return self.output_layers
