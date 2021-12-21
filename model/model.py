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
        self.kernel_size = 3
        self.padding = 1
        self.scale_factor = (4, 4)
        self.flow_sensitivity = 0.5
        self.number_previous_frames = 5
        ###############################################################
        self.featureExtractSeq1 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=self.kernel_size, padding=self.padding),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=self.kernel_size, padding=self.padding),
            nn.ReLU(),
            nn.Conv2d(32, 8, kernel_size=self.kernel_size, padding=self.padding),
        )
        self.featureExtractSeq2 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=self.kernel_size, padding=self.padding),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=self.kernel_size, padding=self.padding),
            nn.ReLU(),
            nn.Conv2d(32, 8, kernel_size=self.kernel_size, padding=self.padding),
        )
        self.featureExtractSeq3 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=self.kernel_size, padding=self.padding),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=self.kernel_size, padding=self.padding),
            nn.ReLU(),
            nn.Conv2d(32, 8, kernel_size=self.kernel_size, padding=self.padding),
        )
        self.featureExtractSeq4 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=self.kernel_size, padding=self.padding),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=self.kernel_size, padding=self.padding),
            nn.ReLU(),
            nn.Conv2d(32, 8, kernel_size=self.kernel_size, padding=self.padding),
        )
        self.featureExtractSeq5 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=self.kernel_size, padding=self.padding),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=self.kernel_size, padding=self.padding),
            nn.ReLU(),
            nn.Conv2d(32, 8, kernel_size=self.kernel_size, padding=self.padding),
        )
        self.featureExtractSeq6 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=self.kernel_size, padding=self.padding),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=self.kernel_size, padding=self.padding),
            nn.ReLU(),
            nn.Conv2d(32, 8, kernel_size=self.kernel_size, padding=self.padding),
        )
        self.motion_upsampling_model = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
        self.feature_reweighting_model = NSRRFeatureReweightingModel()
        self.reconstructionModel = NSRRReconstructionModel()
    def feature_extract_model1(self, color_images, depth_images):
        x = torch.cat((color_images, depth_images), dim=1)
        x_out = self.featureExtractSeq1(x)
        return torch.cat((x, x_out), dim=1)
    
    def feature_extract_model2(self, color_images, depth_images):
        x = torch.cat((color_images, depth_images), dim=1)
        x_out = self.featureExtractSeq2(x)
        return torch.cat((x, x_out), dim=1)
    
    def feature_extract_model3(self, color_images, depth_images):
        x = torch.cat((color_images, depth_images), dim=1)
        x_out = self.featureExtractSeq3(x)
        return torch.cat((x, x_out), dim=1)

    def feature_extract_model4(self, color_images, depth_images):
        x = torch.cat((color_images, depth_images), dim=1)
        x_out = self.featureExtractSeq4(x)
        return torch.cat((x, x_out), dim=1)

    def feature_extract_model5(self, color_images, depth_images):
        x = torch.cat((color_images, depth_images), dim=1)
        x_out = self.featureExtractSeq5(x)
        return torch.cat((x, x_out), dim=1)
    
    def feature_extract_model6(self, color_images, depth_images):
        x = torch.cat((color_images, depth_images), dim=1)
        x_out = self.featureExtractSeq6(x)
        return torch.cat((x, x_out), dim=1)

    
    def zero_upsampling_model(self, x):
        return upsample_zero_2d(x, scale_factor=self.scale_factor)
    def flow_motion_model(self, x):
        return optical_flow_to_motion(x, self.flow_sensitivity)
    def motion_warping_model(self, img, motion):
        return backward_warp_motion(img, motion)
    def forward(self, x_view, x_depth, x_flow):
        #print(x_view.shape)
        # print(x_depth.shape)
        #print(x_flow.shape)

        # [6:c:w:h]
        current_view = x_view[0].unsqueeze(0)
        current_depth = x_depth[0].unsqueeze(0)
        current_flow = x_flow[0].unsqueeze(0)
        # [1:c:w:h]
        list_previous_view = []
        list_previous_depth = []
        list_previous_flow = []
        
        #current_view_Ycbcr = colors.rgb_to_ycbcr(current_view)
        
        for i in range(1, self.number_previous_frames + 1):
            list_previous_view.append(x_view[i].unsqueeze(0))
            list_previous_depth.append(x_depth[i].unsqueeze(0))
            list_previous_flow.append(x_flow[i].unsqueeze(0))

        # 1°) extract features
        current_features = self.feature_extract_model1(current_view, current_depth)
        list_previous_features = []
        #for i in range(self.number_previous_frames):
        list_previous_features.append(
            self.feature_extract_model2(
                list_previous_view[0], list_previous_depth[0]
            )
        )
        list_previous_features.append(
            self.feature_extract_model3(
                list_previous_view[1], list_previous_depth[1]
            )
        )
        list_previous_features.append(
            self.feature_extract_model4(
                list_previous_view[2], list_previous_depth[2]
            )
        )
        list_previous_features.append(
            self.feature_extract_model5(
                list_previous_view[3], list_previous_depth[3]
            )
        )
        list_previous_features.append(
            self.feature_extract_model6(
                list_previous_view[4], list_previous_depth[4]
            )
        )

        # 2°) upsample features
        # print('c = ', current_features.shape)
        current_features_upsampled = self.zero_upsampling_model(current_features)
        current_rgbd = torch.cat((current_view, current_depth),dim=1)
        current_features_upsampled_for_reweighting = self.zero_upsampling_model(current_rgbd)
        
        list_previous_features_upsampled = []
        for i in range(self.number_previous_frames):
            list_previous_features_upsampled.append(
                self.zero_upsampling_model(list_previous_features[i])
            )

        # 3°) we need to convert from optical flow
        # to motion vectors,then upsample them.


        current_motion_upsampled = self.motion_upsampling_model(
            self.flow_motion_model(current_flow)
        )
        list_previous_motion_upsampled = []
        for i in range(self.number_previous_frames):
            list_previous_motion_upsampled.append(
                self.motion_upsampling_model(
                    self.flow_motion_model(list_previous_flow[i])
                )
            )

        # 4°) warp previous features and motion recursively
        # to align them with the current one.
        list_previous_motion_from_current = []
        list_previous_motion_from_current.append(list_previous_motion_upsampled[0])

        # back warp motion
        for i in range(1, self.number_previous_frames):
            list_previous_motion_from_current.append(
                self.motion_warping_model(
                    list_previous_motion_upsampled[i],
                    list_previous_motion_from_current[-1]
                )
            )

        # back warp feature
        list_previous_features_warped = []
        for i in range(self.number_previous_frames):
            list_previous_features_warped.append(
                self.motion_warping_model(
                    list_previous_features_upsampled[i],
                    list_previous_motion_from_current[i]
                )
            )

        # 5°) reweight features of previous frames


        list_previous_features_reweighted = self.feature_reweighting_model.forward(
            current_features_upsampled_for_reweighting,
            list_previous_features_warped
        )

        # 6°) reconstruction 
        # print("l = ", len(list_previous_features_reweighted))
        # print("c.shape = ", current_features_upsampled.shape)
        # print('-----------------------------------------')
        # for i in list_previous_features_reweighted:
        #     print(i.shape)
        # print('-----------------------------------------')
        target = self.reconstructionModel.forward(current_features_upsampled, list_previous_features_reweighted)

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
        # Adding padding here so that we do not lose width or height because of the convolutions.
        # The input and output must have the same image dimensions so that we may concatenate them
        padding = 1
        # todo: I'm unsure about what to feed the module here, from the paper:
        # "The feature reweighting module is a 3-layer convolutional neural network,
        # which takes the RGB-D of the zero-upsampled current frame
        # as well as the zero-upsampled, warped previous frames as input,
        # and generates a pixel-wise weighting map for each previous frame,
        # with values between 0 and 10."
        # So do we concatenated the RGBD of the current frame
        # to each previous RGBD frame? That's what I'm going to do for now.
        # So the input number of channels in the first 2D convolution is 8.
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
        # list_weighting_maps = []
        #print("current_upsample:", current_features_upsampled.shape)
        #list_previous_rgbd = []
        reweight_feed_in = current_features_upsampled_for_reweighting
        for previous_features_warped in list_previous_features_warped:
            reweight_feed_in = torch.cat((reweight_feed_in, previous_features_warped[:,:4]), dim=1)
        
        list_weighting_map = self.weighting(reweight_feed_in)
        list_weighting_map_cal = list_weighting_map.squeeze(0)
        list_previous_features_reweighted = []
        for i in range(5):
            #list_previous_features_reweighted.append(list_previous_features_warped[i])
            tmp = list_previous_features_warped[i].squeeze(0)
            tmp2 = [1]*12
            for j in range(12):
                tmp2[j] = torch.mul(tmp[j], list_weighting_map_cal[i])
            result_list_previous0 = tmp2[0].unsqueeze(0)
            result_list_previous1 = torch.cat((result_list_previous0, tmp2[1].unsqueeze(0)), dim=0)
            result_list_previous2 = torch.cat((result_list_previous1, tmp2[2].unsqueeze(0)), dim=0)
            result_list_previous3 = torch.cat((result_list_previous2, tmp2[3].unsqueeze(0)), dim=0)
            result_list_previous4 = torch.cat((result_list_previous3, tmp2[4].unsqueeze(0)), dim=0)
            result_list_previous5 = torch.cat((result_list_previous4, tmp2[5].unsqueeze(0)), dim=0)
            result_list_previous6 = torch.cat((result_list_previous5, tmp2[6].unsqueeze(0)), dim=0)
            result_list_previous7 = torch.cat((result_list_previous6, tmp2[7].unsqueeze(0)), dim=0)
            result_list_previous8 = torch.cat((result_list_previous7, tmp2[8].unsqueeze(0)), dim=0)
            result_list_previous9 = torch.cat((result_list_previous8, tmp2[9].unsqueeze(0)), dim=0)
            result_list_previous10 = torch.cat((result_list_previous9, tmp2[10].unsqueeze(0)), dim=0)
            result_list_previous11 = torch.cat((result_list_previous10, tmp2[11].unsqueeze(0)), dim=0)
            result_list_previous_final = result_list_previous11.unsqueeze(0)
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

        # Split the network into 5 groups of 2 layers to apply concat operation at each stage
        # todo: the first layer of the model would take
        # the concatenated features of all previous frames,
        # so the input number of channels of the first 2D convolution
        # would be 12 * self.number_previous_frames
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
        # print("x.shape = ", x.shape)
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
