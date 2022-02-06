# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
import numpy as np
from ....core.post_processing import oks_nms, soft_oks_nms, oks_iou
from mmpose.core.evaluation.top_down_eval import keypoint_pck_accuracy, pose_pck_accuracy

from mmcv import Config
import random

from ...builder import DATASETS
from ..base import Kpt2dSviewRgbImgTopDownDataset


@DATASETS.register_module()
class TopDownChargerDataset(Kpt2dSviewRgbImgTopDownDataset):
    """CocoDataset dataset for top-down pose estimation.

    `Microsoft COCO: Common Objects in Context' ECCV'2014
    More details can be found in the `paper
    <https://arxiv.org/abs/1405.0312>`__ .

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    COCO keypoint indexes::

    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        dataset_info (DatasetInfo): A class containing all dataset info.
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """

    def __init__(self,
                 ann_dir,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 dataset_info=None,
                 test_mode=False):

        if dataset_info is None:
            warnings.warn(
                'dataset_info is missing. '
                'Check https://github.com/open-mmlab/mmpose/pull/663 '
                'for details.', DeprecationWarning)
            cfg = Config.fromfile('configs/_base_/datasets/coco.py')
            dataset_info = cfg._cfg_dict['dataset_info']

        super().__init__(
            None,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            test_mode=test_mode,
            coco_style=False)

        self.use_gt_bbox = data_cfg['use_gt_bbox']
        self.bbox_file = data_cfg['bbox_file']
        self.det_bbox_thr = data_cfg.get('det_bbox_thr', 0.0)
        self.use_nms = data_cfg.get('use_nms', True)
        self.soft_nms = data_cfg['soft_nms']
        self.nms_thr = data_cfg['nms_thr']
        self.oks_thr = data_cfg['oks_thr']
        self.vis_thr = data_cfg['vis_thr']
        self.slice_size = data_cfg['image_size']

        self.ann_dir = ann_dir
        self.annotations = os.listdir(ann_dir)
        random.shuffle(self.annotations)

        self.num_images = len(self.annotations)
        self.db = self._get_db()

        print(f'=> num_images: {self.num_images}')
        print(f'=> load {len(self.db)} samples')

    def _get_db(self):
        """Load dataset."""

        img_labels = []
        # Add images
        for a in self.annotations:
            image_file = os.path.join(self.img_prefix, a[:-4]+'.png')
            ann_path = os.path.join(self.ann_dir, a)
            tree = ET.parse(ann_path)
            root = tree.getroot()
            keypoints = []
            size = root.find('size')
            img_width = int(size.find('width').text)
            img_height = int(size.find('height').text)
            obj = root.findall('object')[0]
            kps = obj.find('keypoints')
            box = obj.find('bndbox')
            xmin = float(box.find('xmin').text)*img_width
            xmax = float(box.find('xmax').text)*img_width
            ymin = float(box.find('ymin').text)*img_height
            ymax = float(box.find('ymax').text)*img_height
            x, y, w, h = xmin, ymin, xmax-xmin, ymax-ymin
            center, scale = self._xywh2cs(x, y, w, h)
            bbox = [x,y,w,h]

            for i in range(self.ann_info['num_joints']):
                kp = kps.find('keypoint' + str(i))
                point_center = (
                    int((float(kp.find('x').text) * img_width)),
                    int((float(kp.find('y').text) * img_height)))
                keypoints.append(point_center)
            keypoints = np.array(keypoints)
            rec = {
                'image_file': image_file,
                'center': center,
                'scale': scale,
                'bbox': bbox,
                'rotation': 0,
                'joints_3d': keypoints,
                'joints_3d_visible': keypoints, # TODO: handle occluded kpts
                'dataset': self.dataset_name,
                'bbox_score': 1,
                'bbox_id': 0
            }
            img_labels.append(rec)
        return img_labels


    def evaluate(self, outputs, res_folder, metric='mAP', **kwargs):
        """Evaluate coco keypoint results. The pose prediction results will be
        saved in `${res_folder}/result_keypoints.json`.

        Note:
            batch_size: N
            num_keypoints: K
            heatmap height: H
            heatmap width: W

        Args:
            outputs (list(dict))
                :preds (np.ndarray[N,K,3]): The first two dimensions are
                    coordinates, score is the third dimension of the array.
                :boxes (np.ndarray[N,6]): [center[0], center[1], scale[0]
                    , scale[1],area, score]
                :image_paths (list[str]): For example, ['data/coco/val2017
                    /000000393226.jpg']
                :heatmap (np.ndarray[N, K, H, W]): model output heatmap
                :bbox_id (list(int)).
            res_folder (str): Path of directory to save the results.
            metric (str | list[str]): Metric to be performed. Defaults: 'mAP'.

        Returns:
            dict: Evaluation results for evaluation metric.
        """
        kpts = defaultdict(list)
        avg_acc = []
        avg_mse_loss = []

        for output in outputs:
            preds = output['preds']
            boxes = output['boxes']
            image_paths = output['image_paths']
            bbox_ids = output['bbox_ids']
            avg_mse_loss.append(output["mse_loss"].cpu().numpy())

            batch_size = len(image_paths)
            for i in range(batch_size):
                gt_kpt, bbox_h_w = self.read_annotation(image_paths[i].split('/')[-1][:-4]+'.txt')
                _, acc, _ = keypoint_pck_accuracy(np.expand_dims(preds[i, :, :2], 0), np.expand_dims(gt_kpt, 0), np.full((1,4), True), 0.01, np.array([self.slice_size]))
                avg_acc.append(acc)

        return {"acc":np.average(avg_acc), "mse_loss":np.average(avg_mse_loss)}

    def read_annotation(self, file_name):
        ann_path = os.path.join(self.ann_dir, file_name)
        tree = ET.parse(ann_path)
        root = tree.getroot()
        keypoints = []
        size = root.find('size')
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)
        obj = root.findall('object')[0]
        box = obj.find('bndbox')
        xmin = float(box.find('xmin').text)*img_width
        xmax = float(box.find('xmax').text)*img_width
        ymin = float(box.find('ymin').text)*img_height
        ymax = float(box.find('ymax').text)*img_height
        x, y, w, h = xmin, ymin, xmax-xmin, ymax-ymin
        kps = obj.find('keypoints')
        for i in range(self.ann_info['num_joints']):
            kp = kps.find('keypoint' + str(i))
            point_center = (
                int((float(kp.find('x').text) * img_width)),
                int((float(kp.find('y').text) * img_height)))
            keypoints.append(point_center)
        keypoints = np.array(keypoints)
        return keypoints, np.array([h, w])