# Copyright (c) OpenMMLab. All rights reserved.
from cv2 import Mahalanobis
import torch
import torch.nn as nn
import numpy as np
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, normal_init)

from mmpose.core.evaluation import pose_pck_accuracy
from mmpose.core.evaluation.top_down_eval import _get_max_preds
from mmpose.models.utils.ops import resize
from ..builder import HEADS
from .topdown_heatmap_base_head import TopdownHeatmapBaseHead


@HEADS.register_module()
class TopdownUncertaintyHead(TopdownHeatmapBaseHead):
    """Top-down heatmap simple head. paper ref: Bin Xiao et al. ``Simple
    Baselines for Human Pose Estimation and Tracking``.

    TopdownHeatmapSimpleHead is consisted of (>=0) number of deconv layers
    and a simple conv2d layer.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        num_deconv_layers (int): Number of deconv layers.
            num_deconv_layers should >= 0. Note that 0 means
            no deconv layers.
        num_deconv_filters (list|tuple): Number of filters.
            If num_deconv_layers > 0, the length of
        num_deconv_kernels (list|tuple): Kernel sizes.
        in_index (int|Sequence[int]): Input feature index. Default: 0
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resized to the
                same size as the first one and then concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        loss_keypoint (dict): Config for keypoint loss. Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 in_index=0,
                 input_transform=None,
                 train_cfg=None,
                 test_cfg=None,):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self._init_inputs(in_channels, in_index, input_transform)
        self.uncertainty_head = self.build_uncertainty_head(in_channels, out_channels)


    def build_uncertainty_head(self, conv_channels, out_channels, kernel_size=3, padding='same'):
        layers = []
        layers.append(
            build_conv_layer(
                cfg=dict(type='Conv2d'),
                in_channels=conv_channels,
                out_channels=256,
                kernel_size=kernel_size,
                stride=1,
                padding=padding))
        layers.append(
            build_norm_layer(dict(type='BN'), 256)[1])
        layers.append(nn.ReLU(inplace=True))

        layers.append(
            build_conv_layer(
                cfg=dict(type='Conv2d'),
                in_channels=256,
                out_channels=256,
                kernel_size=kernel_size,
                stride=1,
                padding=padding))
        layers.append(
            build_norm_layer(dict(type='BN'), 256)[1])
        layers.append(nn.ReLU(inplace=True))
        layers.append(
            build_conv_layer(
                cfg=dict(type='Conv2d'),
                in_channels=256,
                out_channels=64,
                kernel_size=kernel_size,
                stride=1,
                padding=padding))
        layers.append(
            build_norm_layer(dict(type='BN'), 64)[1])
        layers.append(nn.ReLU(inplace=True))
        layers.append(
            build_conv_layer(
                cfg=dict(type='Conv2d'),
                in_channels=64,
                out_channels=16,
                kernel_size=kernel_size,
                stride=1,
                padding=padding))
        layers.append(
            build_norm_layer(dict(type='BN'), 16)[1])
        layers.append(nn.ReLU(inplace=True))

        layers.append(torch.nn.Flatten())
        layers.append(torch.nn.Linear(262144,(out_channels * 2 + 1) * out_channels))
        layers.append(nn.ELU(inplace=True))
        return  nn.Sequential(*layers)

    def _get_max_preds_tensor(self, heatmaps):
        """Get keypoint predictions from score maps.

        Note:
            batch_size: N
            num_keypoints: K
            heatmap height: H
            heatmap width: W

        Args:
            heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.

        Returns:
            tuple: A tuple containing aggregated results.

            - preds (np.ndarray[N, K, 2]): Predicted keypoint location.
            - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
        """
        N, K, _, W = heatmaps.shape
        heatmaps_reshaped = heatmaps.reshape((N, K, -1))
        # idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
        idx = torch.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
        # maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))
        maxvals = torch.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

        # preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
        preds = torch.tile(idx, (1, 1, 2))
        preds[:, :, 0] = preds[:, :, 0] % W
        preds[:, :, 1] = preds[:, :, 1] // W

        # preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
        preds = torch.where(torch.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
        return preds

    def get_loss(self, output_uncertainty, output, target, img_metas):

        losses = {"uncertainty_loss":0}
        batch_size = output.shape[0]

        preds = self._get_max_preds_tensor(output).cpu()
        gt = self._get_max_preds_tensor(target).cpu()
        preds = preds.reshape(batch_size, -1)
        gt = gt.reshape(batch_size, -1)

        # See: https://www.merl.com/publications/docs/TR2019-117.pdf UGLLI Face Alignment:Estimating Uncertainty with Gaussian Log-Likelihood Loss
        mask = torch.tril(torch.ones([self.out_channels*2, self.out_channels*2], dtype=torch.bool))
        L = torch.zeros([batch_size, self.out_channels*2, self.out_channels*2])
        L[:, mask] = output_uncertainty.cpu()

        sigma = torch.matmul(torch.transpose(L, 1, 2), L).cpu()
        sigma_loss = torch.log(torch.det(sigma))
        sigma_loss = sigma_loss.nan_to_num(10e10) #TODO: check why nan
        # print("sigma_loss", sigma_loss)

        for sample in range(batch_size):
            diff = (gt[sample]-preds[sample]).t().type(torch.FloatTensor)
            mahalanobis_loss = torch.matmul(diff, torch.inverse(sigma[sample]))
            mahalanobis_loss = torch.matmul(mahalanobis_loss, diff)
            # print("mahalanobis_loss", mahalanobis_loss)
            losses["uncertainty_loss"] += sigma_loss[sample] + mahalanobis_loss
        # print("uncertainty_loss", losses["uncertainty_loss"])
        losses["uncertainty_loss"] /=5000
        return losses

    # def get_accuracy(self, output, target, target_weight):
    #     """Calculate accuracy for top-down keypoint loss.

    #     Note:
    #         batch_size: N
    #         num_keypoints: K
    #         heatmaps height: H
    #         heatmaps weight: W

    #     Args:
    #         output (torch.Tensor[NxKxHxW]): Output heatmaps.
    #         target (torch.Tensor[NxKxHxW]): Target heatmaps.
    #         target_weight (torch.Tensor[NxKx1]):
    #             Weights across different joint types.
    #     """

    #     accuracy = dict()

    #     if self.target_type == 'GaussianHeatmap':
    #         _, avg_acc, _ = pose_pck_accuracy(
    #             output.detach().cpu().numpy(),
    #             target.detach().cpu().numpy(),
    #             target_weight.detach().cpu().numpy().squeeze(-1) > 0)
    #         accuracy['acc_pose'] = float(avg_acc)

    #     return accuracy

    def forward(self, x):
        """Forward function."""
        x = self._transform_inputs(x)
        x = self.uncertainty_head(x)
        return x

    def inference_model(self, x, flip_pairs=None):
        """Inference function.

        Returns:
            output_heatmap (np.ndarray): Output heatmaps.

        Args:
            x (torch.Tensor[NxKxHxW]): Input features.
            flip_pairs (None | list[tuple()):
                Pairs of keypoints which are mirrored.
        """
        output = self.forward(x)       
        mask = torch.tril(torch.ones([self.out_channels*2, self.out_channels*2], dtype=torch.bool))
        L = torch.zeros([output.shape[0], self.out_channels*2, self.out_channels*2])
        L[:, mask] = output.cpu()
        sigma = torch.matmul(torch.transpose(L, 1, 2), L).cpu()
        return sigma

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform is not None, in_channels and in_index must be
        list or tuple, with the same length.

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor] | Tensor): multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """
        if not isinstance(inputs, list):
            return inputs

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def init_weights(self):
        """Initialize model weights."""
        for _, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
