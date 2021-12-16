import argparse
import collections

import torch.nn as nn

from parse_config import ConfigParser
from data_loader import NSRRDataLoader
from model import NSRRFeatureExtractionModel, \
    NSRRFeatureReweightingModel, NSRRReconstructionModel, \
    ZeroUpsample2D, OpticalFlowToMotion, BackwardWarp

from utils.unit_test import UnitTest


def main(config):
    downscale_factor = config['data_loader']['args']['downscale_factor']
    downscale_factor = [downscale_factor, downscale_factor]
    data_dir = config['data_loader']['args']['data_dir']
    view_dirname = config['data_loader']['args']['view_dirname']
    depth_dirname = config['data_loader']['args']['depth_dirname']
    flow_dirname = config['data_loader']['args']['flow_dirname']

    number_previous_frames = 5
    scale_factor = (2, 2)
    flow_sensitivity = 0.5

    batch_size = number_previous_frames + 1

    # UnitTest.dataloader_iteration(data_dir, batch_size)
    loader = NSRRDataLoader(data_dir=data_dir,
                            view_dirname=view_dirname,
                            depth_dirname=depth_dirname,
                            flow_dirname=flow_dirname,
                            batch_size=batch_size,
                            downscale_factor=downscale_factor)
    # get a single batch
    x_view, x_depth, x_flow, _ = next(iter(loader))
    print(x_view.shape)
    print(x_depth.shape)
    print(x_flow.shape)
    print(_.shape)
    # Test util functions
    UnitTest.backward_warping(x_view, x_flow, downscale_factor)
    UnitTest.nsrr_loss(x_view)
    UnitTest.zero_upsampling(x_view, downscale_factor)

    # UnitTest.feature_extraction(x_view, x_depth)
    # UnitTest.feature_reweighting(rgbd, rgbd, rgbd, rgbd, rgbd)
    # UnitTest.reconstruction(rgbd, rgbd)

    # Test whole neural network
    # [6:c:w:h]
    current_view = x_view[0].unsqueeze(0)
    current_depth = x_depth[0].unsqueeze(0)
    current_flow = x_flow[0].unsqueeze(0)
    # [1:c:w:h]
    list_previous_view = []
    list_previous_depth = []
    list_previous_flow = []

    for i in range(1, number_previous_frames + 1):
        list_previous_view.append(x_view[i].unsqueeze(0))
        list_previous_depth.append(x_depth[i].unsqueeze(0))
        list_previous_flow.append(x_flow[i].unsqueeze(0))

    # 1°) extract features
    feature_extraction_model = NSRRFeatureExtractionModel()

    current_features = feature_extraction_model.forward(current_view, current_depth)
    list_previous_features = []
    for i in range(number_previous_frames):
        list_previous_features.append(
            feature_extraction_model.forward(list_previous_view[i],
                                             list_previous_depth[i]))

    # 2°) upsample features
    zero_upsampling_model = ZeroUpsample2D(scale_factor=scale_factor)
    current_features_upsampled = zero_upsampling_model.forward(current_features)

    list_previous_features_upsampled = []
    for i in range(number_previous_frames):
        list_previous_features_upsampled.append(
           zero_upsampling_model.forward(list_previous_features[i])
        )

    # 3°) we need to convert from optical flow
    # to motion vectors,then upsample them.
    flow_motion_model = OpticalFlowToMotion(sensitivity=flow_sensitivity)
    motion_upsampling_model = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

    current_motion_upsampled = motion_upsampling_model.forward(
        flow_motion_model.forward(current_flow))
    list_previous_motion_upsampled = []
    for i in range(number_previous_frames):
        list_previous_motion_upsampled.append(
            motion_upsampling_model.forward(
                flow_motion_model.forward(current_flow)
        ))

    # 4°) warp previous features and motion recursively
    # to align them with the current one.
    motion_warping_model = BackwardWarp()

    list_previous_motion_from_current = [
        motion_warping_model.forward(
            # list_previous_motion_upsampled[0],
            # TODO:
            list_previous_motion_upsampled[0],
            current_motion_upsampled
        )
    ]
    
    # back warp motion
    for i in range(1, number_previous_frames):
        list_previous_motion_from_current.append(
            motion_warping_model.forward(
                list_previous_motion_upsampled[i],
                list_previous_motion_from_current[-1]
            )
        )

    # back warp feature
    list_previous_features_warped = []
    for i in range(number_previous_frames):
        list_previous_features_warped.append(
            motion_warping_model(
                list_previous_features_upsampled[i],
                list_previous_motion_from_current[i]
             )
        )

    # 5°) reweight features of previous frames
    feature_reweighting_model = NSRRFeatureReweightingModel()

    list_previous_features_reweighted = feature_reweighting_model.forward(
        current_features_upsampled,
        list_previous_features_warped
    )


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='NSRR Unit testing')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='unused here')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['-ds', '--downscale'], type=int, target=('data_loader', 'args', 'downsample'))
    ]

    config = ConfigParser.from_args(args, options)
    main(config)
