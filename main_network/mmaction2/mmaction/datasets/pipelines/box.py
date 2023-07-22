##### Thanks to https://github.com/vt-vl-lab/video-data-aug/blob/01667cdbd1b952f2510af3422beeeb76e0d9e15a/mmaction/datasets/pipelines/box.py #####

import random

import mmcv
import numpy as np

from ..builder import PIPELINES
from .augmentations import Resize, RandomResizedCrop, Flip
from .formatting import FormatShape


## Loading
@PIPELINES.register_module()
class DetectionLoad(object):
    """Load human detection results with given indices.
    Required keys are "frame_dir", "frame_inds", and "all_detections",
    added or modified keys are "detections".
    Args:
        thres (float): Threshold for human detection.
        kwargs (dict, optional): Arguments for FileClient.
    """

    def __init__(self, thres=0.4, offset=0, **kwargs):
        self.kwargs = kwargs
        self.thres = thres
        self.offset = offset

    def __call__(self, results):
        """Perform the ``DetectionLoad`` to get detections given indices.
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """

        # directory = results['frame_dir']
        detections = list()

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        offset = results.get('offset', self.offset)

        for frame_idx in results['frame_inds']:
            frame_idx += offset
            frame_idx = min(len(results['all_detections'].keys()), frame_idx)
            cur_detections = results['all_detections'][frame_idx]
            sel = cur_detections[:, -1] > self.thres
            cur_detections = cur_detections[sel, :4]
            detections.append(cur_detections)

        results['detections'] = detections
        # Already have deep copy in the pipeline, we can safely delete it here
        del results['all_detections']

        return results


## Augmentation
@PIPELINES.register_module()
class BuildHumanMask(object):
    """Construct binary human mask from detection bbox.
    Required keys are "detections", added
    or modified keys are "human_masks".
    Args:
        kwargs (dict, optional): Arguments for FileClient.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, results):
        """Perform the ``BuildHumanMask`` augmentation.
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        # NOTE: If no human for the current clip, then treat the whole clip as human region
        box_nums = [det.shape[0] for det in results['detections']]
        num = len(results['detections'])
        if sum(box_nums) == 0:
            human_mask = [np.ones_like(results['imgs'][idx]) for idx in range(num)]
            results['human_mask'] = human_mask
            return results

        human_mask = [np.zeros_like(results['imgs'][idx]) for idx in range(num)]
        for idx in range(len(results['detections'])):
            cur_detections = results['detections'][idx]
            boxes = cur_detections.astype(int)

            # TODO: Speed up this
            num = boxes.shape[0]
            if num > 0:
                for i in range(num):
                    box = boxes[i, :]
                    human_mask[idx][box[1]:box[3], box[0]:box[2], :] = 1

        # Box info is no longer required after this operation
        del results['detections']

        results['human_mask'] = human_mask
        return results


@PIPELINES.register_module()
class ResizeWithBox(Resize):
    """Resize images (and detection boxes) to a specific size.
    Required keys are "imgs", "img_shape", "modality", "detections", added
    or modified keys are "imgs", "img_shape", "keep_ratio", "scale_factor",
    "lazy", "resize_size", "detections".
    Args:
        scale (float | Tuple[int]): If keep_ratio is True, it serves as scaling
            factor or maximum size:
            If it is a float number, the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, the image will
            be rescaled as large as possible within the scale.
            Otherwise, it serves as (w, h) of output size.
        keep_ratio (bool): If set to True, Images will be resized without
            changing the aspect ratio. Otherwise, it will resize images to a
            given size. Default: True.
        interpolation (str): Algorithm used for interpolation:
            "nearest" | "bilinear". Default: "bilinear".
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __call__(self, results):
        """Performs the ResizeWithBox augmentation.
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if 'scale_factor' not in results:
            results['scale_factor'] = np.array([1, 1], dtype=np.float32)
        img_h, img_w = results['img_shape']

        if self.keep_ratio:
            new_w, new_h = mmcv.rescale_size((img_w, img_h), self.scale)
        else:
            new_w, new_h = self.scale

        self.scale_factor = np.array([new_w / img_w, new_h / img_h],
                                     dtype=np.float32)

        results['img_shape'] = (new_h, new_w)
        results['keep_ratio'] = self.keep_ratio
        results['scale_factor'] = results['scale_factor'] * self.scale_factor

        if not self.lazy:
            results['imgs'] = [
                mmcv.imresize(
                    img, (new_w, new_h), interpolation=self.interpolation)
                for img in results['imgs']
            ]

            for idx in range(len(results['detections'])):
                cur_detections = results['detections'][idx]
                cur_detections[:, 0::2] = np.clip(cur_detections[:, 0::2]*self.scale_factor[0], 0, new_w)
                cur_detections[:, 1::2] = np.clip(cur_detections[:, 1::2]*self.scale_factor[1], 0, new_h)
                results['detections'][idx] = cur_detections
        else:
            raise NotImplementedError

        return results


@PIPELINES.register_module()
class RandomResizedCropWithBox(RandomResizedCrop):
    """Random crop that specifics the area and height-weight ratio range.
    Required keys in results are "imgs", "img_shape", "crop_bbox", "lazy",
    and "detections", added or modified keys are "imgs", "crop_bbox", "lazy",
    and "detections".
    Args:
        area_range (Tuple[float]): The candidate area scales range of
            output cropped images. Default: (0.08, 1.0).
        aspect_ratio_range (Tuple[float]): The candidate aspect ratio range of
            output cropped images. Default: (3 / 4, 4 / 3).
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __call__(self, results):
        """Performs the RandomResizeCropWithBox augmentation.
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        img_h, img_w = results['img_shape']

        left, top, right, bottom = self.get_crop_bbox(
            (img_h, img_w), self.area_range, self.aspect_ratio_range)
        new_h, new_w = bottom - top, right - left

        results['crop_bbox'] = np.array([left, top, right, bottom])
        results['img_shape'] = (new_h, new_w)

        if not self.lazy:
            results['imgs'] = [
                img[top:bottom, left:right] for img in results['imgs']
            ]

            for idx in range(len(results['detections'])):
                cur_detections = results['detections'][idx]
                cur_detections[:, 0::2] = np.clip(cur_detections[:, 0::2]-left, 0, new_w)
                cur_detections[:, 1::2] = np.clip(cur_detections[:, 1::2]-top, 0, new_h)
                results['detections'][idx] = cur_detections

        else:
            raise NotImplementedError

        return results


@PIPELINES.register_module()
class FlipWithBox(Flip):
    """Flip the input images with a probability.
    Reverse the order of elements in the given imgs with a specific direction.
    The shape of the imgs is preserved, but the elements are reordered.
    Required keys are "imgs", "img_shape", "modality", and "detections",
    added or modified keys are "imgs", "lazy", "flip_direction", and "detections".
    Args:
        flip_ratio (float): Probability of implementing flip. Default: 0.5.
        direction (str): Flip imgs horizontally or vertically. Options are
            "horizontal" | "vertical". Default: "horizontal".
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __call__(self, results):
        """Performs the Flip augmentation.
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        modality = results['modality']
        assert modality == 'RGB'

        if np.random.rand() < self.flip_ratio:
            flip = True
        else:
            flip = False

        results['flip'] = flip
        results['flip_direction'] = self.direction

        if not self.lazy:
            if flip:
                for i, img in enumerate(results['imgs']):
                    mmcv.imflip_(img, self.direction)

                img_h, img_w = results['img_shape']
                for idx in range(len(results['detections'])):
                    cur_detections = results['detections'][idx].copy()
                    if self.direction == 'horizontal':
                        cur_detections[:, 0] = img_w - results['detections'][idx][:, 2]
                        cur_detections[:, 2] = img_w - results['detections'][idx][:, 0]
                    else:
                        cur_detections[:, 1] = img_h - results['detections'][idx][:, 3]
                        cur_detections[:, 3] = img_h - results['detections'][idx][:, 1]
                    results['detections'][idx] = cur_detections

            else:
                results['imgs'] = list(results['imgs'])

        else:
            raise NotImplementedError

        return results


@PIPELINES.register_module()
class FormatDetShape(FormatShape):

    def __init__(self, input_format, collapse=False):
        super().__init__(input_format, collapse)

    def __call__(self, results):
        """Performs the FormatShape formatting.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if not isinstance(results['imgs'], np.ndarray):
            results['imgs'] = np.array(results['imgs'])
        imgs = results['imgs']
        if not isinstance(results['human_mask'], np.ndarray):
            results['human_mask'] = np.array(results['human_mask'])
        human_mask = results['human_mask']
        # [M x H x W x C]
        # M = 1 * N_crops * N_clips * L
        if self.collapse:
            assert results['num_clips'] == 1

        if self.input_format == 'NCTHW':
            num_clips = results['num_clips']
            clip_len = results['clip_len']

            imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
            # N_crops x N_clips x L x H x W x C
            imgs = np.transpose(imgs, (0, 1, 5, 2, 3, 4))
            # N_crops x N_clips x C x L x H x W
            imgs = imgs.reshape((-1, ) + imgs.shape[2:])
            # M' x C x L x H x W
            # M' = N_crops x N_clips

            human_mask = human_mask.reshape((-1, num_clips, clip_len) + human_mask.shape[1:])
            human_mask = np.transpose(human_mask, (0, 1, 5, 2, 3, 4))
            human_mask = human_mask.reshape((-1, ) + human_mask.shape[2:])
        elif self.input_format == 'NCHW':
            imgs = np.transpose(imgs, (0, 3, 1, 2))
            # M x C x H x W

            human_mask = np.transpose(human_mask, (0, 3, 1, 2))

        if self.collapse:
            assert imgs.shape[0] == 1
            imgs = imgs.squeeze(0)
            human_mask = human_mask.squeeze(0)

        results['imgs'] = imgs
        results['input_shape'] = imgs.shape
        results['human_mask'] = human_mask
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(input_format='{self.input_format}')"
        return repr_str