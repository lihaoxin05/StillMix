# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

import torch
import torch.nn.functional as F
from torch.distributions.beta import Beta

from .builder import BLENDINGS

import os
import pickle
import numpy as np
import decord
import cv2
import mmcv
from mmcv.fileio import FileClient
mmcv.use_backend('cv2')
file_client = FileClient('disk')

import kornia

__all__ = ['BaseMiniBatchBlending', 'MixupBlending', 'MixupBlending_fixbs', 'CutmixBlending', 'CutmixBlending_fixbs', 'ActorCutmixBlending', 'BEIntraBlending', 'FameBlending', 'StillMixRandomBlending', 'StillMixFrameBankBlending', 'StillMixFrameBankBlending_MixLabel', 'TemporalSelect']


class BaseMiniBatchBlending(metaclass=ABCMeta):
    """Base class for Image Aliasing."""

    def __init__(self, num_classes):
        self.num_classes = num_classes

    @abstractmethod
    def do_blending(self, imgs, label, **kwargs):
        pass

    def __call__(self, imgs, label, **kwargs):
        """Blending data in a mini-batch.

        Images are float tensors with the shape of (B, N, C, H, W) for 2D
        recognizers or (B, N, C, T, H, W) for 3D recognizers.

        Besides, labels are converted from hard labels to soft labels.
        Hard labels are integer tensors with the shape of (B, 1) and all of the
        elements are in the range [0, num_classes - 1].
        Soft labels (probablity distribution over classes) are float tensors
        with the shape of (B, 1, num_classes) and all of the elements are in
        the range [0, 1].

        Args:
            imgs (torch.Tensor): Model input images, float tensor with the
                shape of (B, N, C, H, W) or (B, N, C, T, H, W).
            label (torch.Tensor): Hard labels, integer tensor with the shape
                of (B, 1) and all elements are in range [0, num_classes).
            kwargs (dict, optional): Other keyword argument to be used to
                blending imgs and labels in a mini-batch.

        Returns:
            mixed_imgs (torch.Tensor): Blending images, float tensor with the
                same shape of the input imgs.
            mixed_label (torch.Tensor): Blended soft labels, float tensor with
                the shape of (B, 1, num_classes) and all elements are in range
                [0, 1].
        """
        one_hot_label = F.one_hot(label, num_classes=self.num_classes)

        mixed_imgs, mixed_label = self.do_blending(imgs, one_hot_label,
                                                   **kwargs)

        return mixed_imgs, mixed_label


@BLENDINGS.register_module()
class MixupBlending(BaseMiniBatchBlending):
    """Implementing Mixup in a mini-batch.

    This module is proposed in `mixup: Beyond Empirical Risk Minimization
    <https://arxiv.org/abs/1710.09412>`_.
    Code Reference https://github.com/open-mmlab/mmclassification/blob/master/mmcls/models/utils/mixup.py # noqa

    Args:
        num_classes (int): The number of classes.
        alpha (float): Parameters for Beta distribution.
    """

    def __init__(self, num_classes, alpha=.2):
        super().__init__(num_classes=num_classes)
        self.beta = Beta(alpha, alpha)

    def do_blending(self, imgs, label, **kwargs):
        """Blending images with mixup."""
        assert len(kwargs) == 0, f'unexpected kwargs for mixup {kwargs}'

        lam = self.beta.sample()
        batch_size = imgs.size(0)
        rand_index = torch.randperm(batch_size)

        mixed_imgs = lam * imgs + (1 - lam) * imgs[rand_index, :]
        mixed_label = lam * label + (1 - lam) * label[rand_index, :]

        return mixed_imgs, mixed_label


@BLENDINGS.register_module()
class MixupBlending_fixbs(BaseMiniBatchBlending):
    """Implementing Mixup in a mini-batch.

    This module is proposed in `mixup: Beyond Empirical Risk Minimization
    <https://arxiv.org/abs/1710.09412>`_.
    Code Reference https://github.com/open-mmlab/mmclassification/blob/master/mmcls/models/utils/mixup.py # noqa

    Args:
        num_classes (int): The number of classes.
        alpha (float): Parameters for Beta distribution.
    """

    def __init__(self, num_classes, alpha=.2, prob_aug=1.0):
        super().__init__(num_classes=num_classes)
        self.beta = Beta(alpha, alpha)
        self.prob_aug = prob_aug

    def do_blending(self, imgs, label, **kwargs):
        """Blending images with mixup."""
        assert len(kwargs) == 0, f'unexpected kwargs for mixup {kwargs}'

        lam = self.beta.sample()
        batch_size = imgs.size(0)
        rand_index = torch.randperm(batch_size)

        mixed_imgs = lam * imgs + (1 - lam) * imgs[rand_index, :]
        mixed_label = lam * label + (1 - lam) * label[rand_index, :]

         ## choose samples according to prob
        if self.prob_aug < 1:
            rand_batch = torch.rand(batch_size)
            aug_ind = torch.where(rand_batch < self.prob_aug)
            ori_ind = torch.where(rand_batch >= self.prob_aug)
            all_imgs = torch.cat([mixed_imgs[aug_ind], imgs[ori_ind]], dim=0).contiguous()
            all_label = torch.cat([mixed_label[aug_ind], label[ori_ind]], dim=0).contiguous()
        else:
            all_imgs = mixed_imgs
            all_label = mixed_label

        return all_imgs, all_label


@BLENDINGS.register_module()
class CutmixBlending(BaseMiniBatchBlending):
    """Implementing Cutmix in a mini-batch.

    This module is proposed in `CutMix: Regularization Strategy to Train Strong
    Classifiers with Localizable Features <https://arxiv.org/abs/1905.04899>`_.
    Code Reference https://github.com/clovaai/CutMix-PyTorch

    Args:
        num_classes (int): The number of classes.
        alpha (float): Parameters for Beta distribution.
    """

    def __init__(self, num_classes, alpha=.2):
        super().__init__(num_classes=num_classes)
        self.beta = Beta(alpha, alpha)

    @staticmethod
    def rand_bbox(img_size, lam):
        """Generate a random boudning box."""
        w = img_size[-1]
        h = img_size[-2]
        cut_rat = torch.sqrt(1. - lam)
        cut_w = torch.tensor(int(w * cut_rat))
        cut_h = torch.tensor(int(h * cut_rat))

        # uniform
        cx = torch.randint(w, (1, ))[0]
        cy = torch.randint(h, (1, ))[0]

        bbx1 = torch.clamp(cx - torch.div(cut_w, 2, rounding_mode='trunc'), 0, w)
        bby1 = torch.clamp(cy - torch.div(cut_h, 2, rounding_mode='trunc'), 0, h)
        bbx2 = torch.clamp(cx + torch.div(cut_w, 2, rounding_mode='trunc'), 0, w)
        bby2 = torch.clamp(cy + torch.div(cut_h, 2, rounding_mode='trunc'), 0, h)

        return bbx1, bby1, bbx2, bby2

    def do_blending(self, imgs, label, **kwargs):
        """Blending images with cutmix."""
        assert len(kwargs) == 0, f'unexpected kwargs for cutmix {kwargs}'

        batch_size = imgs.size(0)
        rand_index = torch.randperm(batch_size)
        lam = self.beta.sample()

        bbx1, bby1, bbx2, bby2 = self.rand_bbox(imgs.size(), lam)
        imgs[:, ..., bby1:bby2, bbx1:bbx2] = imgs[rand_index, ..., bby1:bby2,
                                                  bbx1:bbx2]
        lam = 1 - (1.0 * (bbx2 - bbx1) * (bby2 - bby1) /
                   (imgs.size()[-1] * imgs.size()[-2]))

        label = lam * label + (1 - lam) * label[rand_index, :]

        return imgs, label


@BLENDINGS.register_module()
class CutmixBlending_fixbs(BaseMiniBatchBlending):
    """Implementing Cutmix in a mini-batch.

    This module is proposed in `CutMix: Regularization Strategy to Train Strong
    Classifiers with Localizable Features <https://arxiv.org/abs/1905.04899>`_.
    Code Reference https://github.com/clovaai/CutMix-PyTorch

    Args:
        num_classes (int): The number of classes.
        alpha (float): Parameters for Beta distribution.
    """

    def __init__(self, num_classes, alpha=.2, prob_aug=1.0):
        super().__init__(num_classes=num_classes)
        self.beta = Beta(alpha, alpha)
        self.prob_aug = prob_aug

    @staticmethod
    def rand_bbox(img_size, lam):
        """Generate a random boudning box."""
        w = img_size[-1]
        h = img_size[-2]
        cut_rat = torch.sqrt(1. - lam)
        cut_w = torch.tensor(int(w * cut_rat))
        cut_h = torch.tensor(int(h * cut_rat))

        # uniform
        cx = torch.randint(w, (1, ))[0]
        cy = torch.randint(h, (1, ))[0]

        bbx1 = torch.clamp(cx - torch.div(cut_w, 2, rounding_mode='trunc'), 0, w)
        bby1 = torch.clamp(cy - torch.div(cut_h, 2, rounding_mode='trunc'), 0, h)
        bbx2 = torch.clamp(cx + torch.div(cut_w, 2, rounding_mode='trunc'), 0, w)
        bby2 = torch.clamp(cy + torch.div(cut_h, 2, rounding_mode='trunc'), 0, h)

        return bbx1, bby1, bbx2, bby2

    def do_blending(self, imgs, label, **kwargs):
        """Blending images with cutmix."""
        assert len(kwargs) == 0, f'unexpected kwargs for cutmix {kwargs}'

        original_imgs = imgs.clone()
        original_label = label.clone()

        batch_size = imgs.size(0)
        rand_index = torch.randperm(batch_size)
        lam = self.beta.sample()

        bbx1, bby1, bbx2, bby2 = self.rand_bbox(imgs.size(), lam)
        imgs[:, ..., bby1:bby2, bbx1:bbx2] = imgs[rand_index, ..., bby1:bby2,
                                                  bbx1:bbx2]
        lam = 1 - (1.0 * (bbx2 - bbx1) * (bby2 - bby1) /
                   (imgs.size()[-1] * imgs.size()[-2]))

        label = lam * label + (1 - lam) * label[rand_index, :]

        ## choose samples according to prob
        if self.prob_aug < 1:
            rand_batch = torch.rand(batch_size)
            aug_ind = torch.where(rand_batch < self.prob_aug)
            ori_ind = torch.where(rand_batch >= self.prob_aug)
            all_imgs = torch.cat([imgs[aug_ind], original_imgs[ori_ind]], dim=0).contiguous()
            all_label = torch.cat([label[aug_ind], original_label[ori_ind]], dim=0).contiguous()
        else:
            all_imgs = imgs
            all_label = label

        return all_imgs, all_label


@BLENDINGS.register_module()
class ActorCutmixBlending(BaseMiniBatchBlending):

    def __init__(self, num_classes, prob_aug=0.5):
        super().__init__(num_classes=num_classes)
        self.prob_aug = prob_aug

    def do_blending(self, imgs, label, **kwargs):
        assert 'human_mask' in kwargs
        human_mask = kwargs['human_mask']
        # Select clips with valid ActorCutMix augmentation
        full_size = human_mask[0].numel()
        mask_size = human_mask.sum(tuple([i for i in range(1,len(human_mask.shape))]))
        invalid = torch.logical_or(mask_size == full_size, mask_size == 0)
        # If batch size is an odd number, then the central one is always invalid
        if imgs.shape[0] % 2 != 0:
            invalid[imgs.shape[0]//2] = True
        # For invalid clips, set the whole clip as foreground so that we won't mess up the label
        human_mask[invalid] = 1

        view_shape = [-1] + [1 for i in range(1,len(imgs.shape))]
        invalid = invalid.float().view(view_shape)

        fg = imgs * human_mask
        bg = imgs * (1-human_mask)
        bg = torch.flip(bg, dims=[0]) * (1-human_mask)
        temp = fg + bg

        aug_imgs = invalid * imgs + (1-invalid) * temp

        temp_mask = human_mask.reshape(human_mask.shape[0], -1)
        ratio = temp_mask.sum(1) / (1. * temp_mask.shape[1])
        weight = -(ratio-1)**4+1
        weight = weight.unsqueeze(1).unsqueeze(1)

        aug_label = weight * label + (1-weight) * torch.flip(label, dims=[0])

        ## choose samples according to prob
        if self.prob_aug < 1:
            rand_batch = torch.rand(imgs.shape[0])
            aug_ind = torch.where(rand_batch < self.prob_aug)
            ori_ind = torch.where(rand_batch >= self.prob_aug)
            all_imgs = torch.cat([aug_imgs[aug_ind], imgs[ori_ind]], dim=0).contiguous()
            all_label = torch.cat([aug_label[aug_ind], label[ori_ind]], dim=0).contiguous()
        else:
            all_imgs = aug_imgs
            all_label = aug_label

        return all_imgs, all_label


@BLENDINGS.register_module()
class BEIntraBlending(BaseMiniBatchBlending):
    """
    videos (torch.Tensor): Model input videos, float tensor with the
        shape of (B, N, C, H, W) or (B, N, C, T, H, W).
    label (torch.Tensor): Labels are converted from hard labels to soft labels.
    Hard labels are integer tensors with the shape of (B, 1) and all of the
    elements are in the range [0, num_classes - 1].
    Soft labels (probablity distribution over classes) are float tensors
    with the shape of (B, 1, num_classes) and all of the elements are in
    the range [0, 1].

    kwargs (dict, optional): Other keyword argument to be used to
        blending imgs and labels in a mini-batch.
    
    Args:
        num_classes (int): The number of classes.
        alpha (float): Parameters for Beta distribution.
    """

    def __init__(self, num_classes, alpha=1.0, prob_aug=1.0):
        super().__init__(num_classes=num_classes)
        self.beta = Beta(alpha, alpha)
        self.prob_aug = prob_aug

    def do_blending(self, videos, label, **kwargs):
        assert len(kwargs) == 0, f'unexpected kwargs for mixup {kwargs}'

        lam = self.beta.sample() * 0.3
        indicator = len(videos.shape)
        if indicator == 5: ## 2D recognizer
            batch_size, num_clip, channel, h, w = videos.shape
            tmp_video = videos
            sample_from = num_clip
            num_sample = batch_size
            num_elements = channel*h*w
        elif indicator == 6: ## 3D recognizer
            batch_size, num_clip, channel, clip_len, h, w = videos.shape
            tmp_video = videos.permute(0,1,3,2,4,5).contiguous()
            sample_from = clip_len
            num_sample = batch_size*num_clip
            num_elements = channel*h*w
        else:
            assert False, "unsupport video size"

        ## randomly choose one frame for each video
        frame_sample_index = torch.stack([torch.randint(0, sample_from, (num_sample,)).unsqueeze(-1)]*num_elements, dim=-1).cuda()
        frames = tmp_video.view(num_sample, sample_from, num_elements).contiguous().gather(1,frame_sample_index).view(batch_size, num_sample//batch_size, 1, channel, h, w)

        if indicator == 5:
            sampled_frames = frames.squeeze(1)
        elif indicator == 6:
            sampled_frames = frames.permute(0,1,3,2,4,5).contiguous()

        ## mix
        mixed_videos = (1 - lam) * videos + lam * sampled_frames
        ## choose samples according to prob
        if self.prob_aug < 1:
            rand_batch = torch.rand(batch_size)
            aug_ind = torch.where(rand_batch < self.prob_aug)
            ori_ind = torch.where(rand_batch >= self.prob_aug)
            all_videos = torch.cat([mixed_videos[aug_ind], videos[ori_ind]], dim=0).contiguous()
            all_label = torch.cat([label[aug_ind], label[ori_ind]], dim=0).contiguous()
        else:
            all_videos = mixed_videos
            all_label = label

        return all_videos, all_label


@BLENDINGS.register_module()
class FameBlending(BaseMiniBatchBlending):
    """
    videos (torch.Tensor): Model input videos, float tensor with the
        shape of (B, N, C, H, W) or (B, N, C, T, H, W).
    label (torch.Tensor): Labels are converted from hard labels to soft labels.
    Hard labels are integer tensors with the shape of (B, 1) and all of the
    elements are in the range [0, num_classes - 1].
    Soft labels (probablity distribution over classes) are float tensors
    with the shape of (B, 1, num_classes) and all of the elements are in
    the range [0, 1].

    kwargs (dict, optional): Other keyword argument to be used to
        blending imgs and labels in a mini-batch.
    
    Args:
        num_classes (int): The number of classes.
        alpha (float): Parameters for Beta distribution.
    """

    def __init__(self, num_classes, frame_mean, frame_std, crop_size=224, beta=0.5, prob_aug=1.0):
        super().__init__(num_classes=num_classes)
        self.frame_mean = frame_mean
        self.frame_std = frame_std
        gauss_size = int(0.1 * crop_size) // 2 * 2 + 1
        self.gauss = kornia.filters.GaussianBlur2d(
            (gauss_size, gauss_size),
            (gauss_size / 3, gauss_size / 3))
        self.beta = beta
        self.prob_aug = prob_aug
        self.eps = 1e-8

    def norm_batch(self, matrix):
        # matrix : B*H*W
        B, H, W = matrix.shape
        matrix = matrix.flatten(start_dim=1)
        matrix -= matrix.min(dim=-1, keepdim=True)[0]
        matrix /= (matrix.max(dim=-1, keepdim=True)[0] + self.eps)
        return matrix.reshape(B, H, W)

    def batched_bincount(self, x, dim, max_value):
        target = torch.zeros(x.shape[0], max_value, dtype=x.dtype, device=x.device)
        values = torch.ones_like(x)
        target.scatter_add_(dim, x, values)
        return target
    
    def getSeg(self, mask, video_clips):
        # input mask:B, H, W; video_clips:B, C, T, H, W
        # return soft seg mask: B, H, W
        B, C, T, H, W = video_clips.shape
        video_clips_ = video_clips.mean(dim=2) # B, C, H, W
        img_hsv = kornia.color.rgb_to_hsv(video_clips_.reshape(-1, C, H, W))  # B, C, H, W
        sampled_fg_index = torch.topk(mask.reshape(B, -1), k=int(0.5 * H * W), dim=-1)[1]  # shape B * K
        sampled_bg_index = torch.topk(mask.reshape(B, -1), k=int(0.1 * H * W), dim=-1, largest=False)[1]  # shape B * K
        
        dimH, dimS, dimV = 10, 10, 10
        img_hsv = img_hsv.reshape(B, -1, H, W)  # B * C * H * W
        img_h = img_hsv[:, 0]
        img_s = img_hsv[:, 1]
        img_v = img_hsv[:, 2]
        hx = (img_s * torch.cos(img_h * 2 * np.pi) + 1) / 2
        hy = (img_s * torch.sin(img_h * 2 * np.pi) + 1) / 2
        h = torch.round(hx * (dimH - 1) + 1)
        s = torch.round(hy * (dimS - 1) + 1)
        v = torch.round(img_v * (dimV - 1) + 1)
        color_map = h + (s - 1) * dimH + (v - 1) * dimH * dimS  # B, H, W
        color_map = color_map.reshape(B, -1).long()
        col_fg = color_map.gather(index=sampled_fg_index, dim=-1)  # B * K
        col_bg = color_map.gather(index=sampled_bg_index, dim=-1)  # B * K
        dict_fg = self.batched_bincount(col_fg, dim=1, max_value=dimH * dimS * dimV)  # B * (dimH * dimS * dimV)
        dict_bg = self.batched_bincount(col_bg, dim=1, max_value=dimH * dimS * dimV)  # B * (dimH * dimS * dimV)
        dict_fg = dict_fg.float()
        dict_bg = dict_bg.float() + 1
        dict_fg /= (dict_fg.sum(dim=-1, keepdim=True) + self.eps)
        dict_bg /= (dict_bg.sum(dim=-1, keepdim=True) + self.eps)

        pr_fg = dict_fg.gather(dim=1, index=color_map)
        pr_bg = dict_bg.gather(dim=1, index=color_map)
        refine_mask = pr_fg / (pr_bg + pr_fg)
        mask = self.gauss(refine_mask.reshape(-1, 1, H, W))
        mask = self.norm_batch(mask.reshape(-1, H, W))
        num_fg = int(self.beta * H * W)
        sampled_index = torch.topk(mask.reshape(B, -1), k=num_fg, dim=-1)[1]
        mask = torch.zeros_like(mask).reshape(B, -1)
        b_index = torch.LongTensor([[i]*num_fg for i in range(B)])
        mask[b_index.view(-1), sampled_index.view(-1)] = 1
        return mask.reshape(B, H, W)

    def getmask(self, video_clips):
        # input video_clips: B, C, T, H, W
        # return soft seg mask: B, H, W
        B, C, T, H, W = video_clips.shape
        im_diff = (video_clips[:, :, 0:-1] - video_clips[:, :, 1:]).abs().sum(dim=1).mean(dim=1)  # B, H, W
        mask = self.gauss(im_diff.reshape(-1, 1, H, W))
        mask = self.norm_batch(mask.reshape(-1, H, W))  # B, H, W
        mask = self.getSeg(mask, video_clips)
        return mask
    
    def do_blending(self, videos, label, **kwargs):
        assert len(kwargs) == 0, f'unexpected kwargs for mixup {kwargs}'

        indicator = len(videos.shape)
        if indicator == 5: ## 2D recognizer
            batch_size, num_clip, channel, h, w = videos.shape
            tmp_video = videos.permute(0,2,1,3,4).contiguous()
        elif indicator == 6: ## 3D recognizer
            batch_size, num_clip, channel, clip_len, h, w = videos.shape
            tmp_video = videos.view(batch_size*num_clip, channel, clip_len, h, w).contiguous()
        else:
            assert False, "unsupport video size"

        tmp_video = tmp_video * torch.tensor(self.frame_std, device=tmp_video.device).reshape(1,3,1,1,1) + torch.tensor(self.frame_mean, device=tmp_video.device).reshape(1,3,1,1,1)
        tmp_video = tmp_video / 255
        mask = self.getmask(tmp_video) #B, H, W
        if indicator == 5: ## 2D recognizer
            mask = mask.unsqueeze(1).unsqueeze(1)
        elif indicator == 6: ## 3D recognizer
            mask = mask.unsqueeze(1).unsqueeze(1).view(batch_size, num_clip, 1, 1, h, w).contiguous()
        index = torch.randperm(batch_size, device=videos.device)
        video_fuse = videos[index] * (1 - mask) + videos * mask

        ## choose samples according to prob
        if self.prob_aug < 1:
            rand_batch = torch.rand(batch_size)
            aug_ind = torch.where(rand_batch < self.prob_aug)
            ori_ind = torch.where(rand_batch >= self.prob_aug)
            all_videos = torch.cat([video_fuse[aug_ind], videos[ori_ind]], dim=0).contiguous()
            all_label = torch.cat([label[aug_ind], label[ori_ind]], dim=0).contiguous()
        else:
            all_videos = video_fuse
            all_label = label

        return all_videos, all_label


@BLENDINGS.register_module()
class StillMixRandomBlending(BaseMiniBatchBlending):
    """
    videos (torch.Tensor): Model input videos, float tensor with the
        shape of (B, N, C, H, W) or (B, N, C, T, H, W).
    label (torch.Tensor): Labels are converted from hard labels to soft labels.
    Hard labels are integer tensors with the shape of (B, 1) and all of the
    elements are in the range [0, num_classes - 1].
    Soft labels (probablity distribution over classes) are float tensors
    with the shape of (B, 1, num_classes) and all of the elements are in
    the range [0, 1].

    kwargs (dict, optional): Other keyword argument to be used to
        blending imgs and labels in a mini-batch.
    
    Args:
        num_classes (int): The number of classes.
        alpha (float): Parameters for Beta distribution.
    """

    def __init__(self, num_classes, alpha1=200.0, alpha2=200.0, prob_aug=0.5):
        super().__init__(num_classes=num_classes)
        self.beta = Beta(alpha1, alpha2)
        self.prob_aug = prob_aug

    def do_blending(self, videos, label, **kwargs):
        assert len(kwargs) == 0, f'unexpected kwargs for mixup {kwargs}'

        lam = self.beta.sample()
        indicator = len(videos.shape)
        if indicator == 5: ## 2D recognizer
            batch_size, num_clip, channel, h, w = videos.shape
            tmp_video = videos
            sample_from = num_clip
            num_sample = batch_size
            num_elements = channel*h*w
        elif indicator == 6: ## 3D recognizer
            batch_size, num_clip, channel, clip_len, h, w = videos.shape
            tmp_video = videos.permute(0,1,3,2,4,5).contiguous()
            sample_from = clip_len
            num_sample = batch_size*num_clip
            num_elements = channel*h*w
        else:
            assert False, "Invalid video size."

        ## randomly choose one frame for each video
        frame_sample_index = torch.stack([torch.randint(0, sample_from, (num_sample,)).unsqueeze(-1)]*num_elements, dim=-1).cuda()
        frames = tmp_video.view(num_sample, sample_from, num_elements).contiguous().gather(1,frame_sample_index).view(batch_size, num_sample//batch_size, 1, channel, h, w)
        ## randomly choose one frame to mix
        frames_index = torch.stack([torch.arange(0,batch_size)]*batch_size, dim=0)
        frames_index = frames_index.view(batch_size**2)[:-1].view(batch_size-1, batch_size+1)[:,1:].contiguous().view(batch_size, batch_size-1)
        sample_index = torch.randint(0, batch_size-1, (batch_size, 1))
        sampled_frames_index = frames_index.gather(1, sample_index).view(-1)
        sampled_frames = frames[sampled_frames_index].view(batch_size, 1, num_sample//batch_size, 1, channel, h, w).squeeze(1)

        if indicator == 5:
            sampled_frames = sampled_frames.squeeze(1)
        elif indicator == 6:
            sampled_frames = sampled_frames.permute(0,1,3,2,4,5).contiguous()

        ## mix
        mixed_videos = lam * videos + (1 - lam) * sampled_frames
        ## choose samples according to prob
        if self.prob_aug < 1:
            rand_batch = torch.rand(batch_size)
            aug_ind = torch.where(rand_batch < self.prob_aug)
            ori_ind = torch.where(rand_batch >= self.prob_aug)
            all_videos = torch.cat([mixed_videos[aug_ind], videos[ori_ind]], dim=0).contiguous()
            all_label = torch.cat([label[aug_ind], label[ori_ind]], dim=0).contiguous()
        else:
            all_videos = mixed_videos
            all_label = label

        return all_videos, all_label


@BLENDINGS.register_module()
class StillMixFrameBankBlending(BaseMiniBatchBlending):
    """
    videos (torch.Tensor): Model input videos, float tensor with the
        shape of (B, N, C, H, W) or (B, N, C, T, H, W).
    label (torch.Tensor): Labels are converted from hard labels to soft labels.
    Hard labels are integer tensors with the shape of (B, 1) and all of the
    elements are in the range [0, num_classes - 1].
    Soft labels (probablity distribution over classes) are float tensors
    with the shape of (B, 1, num_classes) and all of the elements are in
    the range [0, 1].

    kwargs (dict, optional): Other keyword argument to be used to
        blending imgs and labels in a mini-batch.
    
    Args:
        num_classes (int): The number of classes.
        alpha (float): Parameters for Beta distribution.
    """

    def __init__(self, num_classes, video_path, frame_mean, frame_std, frame_size, frame_prob_file, bank_size, sample_stra='prob_gt', prob_thre=None, read_from='frame', update_num_epoch=None, alpha1=200.0, alpha2=200.0, prob_aug=0.5):
        super().__init__(num_classes=num_classes)
        self.beta = Beta(alpha1, alpha2)
        self.prob_aug = prob_aug
        self.video_path = video_path
        self.frame_mean = frame_mean
        self.frame_std = frame_std
        self.frame_size = frame_size
        self.frame_prob_data = pickle.load(open(frame_prob_file, 'rb'))
        self.bank_size = bank_size
        self.sample_stra = sample_stra
        self.prob_thre = prob_thre
        self.read_from = read_from
        self.update_num_epoch = update_num_epoch
        if self.update_num_epoch is not None:
            self.last_epoch = -1
            if not self.sample_stra.startswith('prob'):
                self.sample_frames_from_file()
        else:
            self.sample_frames_from_file()
            self.load_frames()
        
    def do_blending(self, videos, label, **kwargs):
        if self.update_num_epoch is not None:
            assert 'epoch_runner' in kwargs and 'epoch' in kwargs
            epoch = kwargs['epoch']
            if epoch != self.last_epoch and epoch % self.update_num_epoch == 0:
                if self.sample_stra.startswith('prob'):
                    self.sample_frames_from_file()
                self.load_frames()
            self.last_epoch = epoch

        lam = self.beta.sample()
        indicator = len(videos.shape)
        if indicator == 5: ## 2D recognizer
            batch_size, num_clip, channel, h, w = videos.shape
            tmp_video = videos
            sample_from = num_clip
            num_sample = batch_size
            num_elements = channel*h*w
        elif indicator == 6: ## 3D recognizer
            batch_size, num_clip, channel, clip_len, h, w = videos.shape
            tmp_video = videos.permute(0,1,3,2,4,5).contiguous()
            sample_from = clip_len
            num_sample = batch_size*num_clip
            num_elements = channel*h*w
        else:
            assert False, "unsupport video size"

        ## randomly choose frames from bank
        sample_queue_index = torch.randint(0, self.bank_size, (batch_size, num_sample//batch_size))
        sample_queue_index = sample_queue_index.view(-1)
        sampled_frames = self.bank[sample_queue_index].view(batch_size, num_sample//batch_size, 1, channel, h, w)

        if indicator == 5:
            sampled_frames = sampled_frames.squeeze(1)
        elif indicator == 6:
            sampled_frames = sampled_frames.permute(0,1,3,2,4,5).contiguous()
        
        ## mix
        mixed_videos = lam * videos + (1 - lam) * sampled_frames
        ## choose samples according to prob
        if self.prob_aug < 1:
            rand_batch = torch.rand(batch_size)
            aug_ind = torch.where(rand_batch < self.prob_aug)
            ori_ind = torch.where(rand_batch >= self.prob_aug)
            all_videos = torch.cat([mixed_videos[aug_ind], videos[ori_ind]], dim=0).contiguous()
            all_label = torch.cat([label[aug_ind], label[ori_ind]], dim=0).contiguous()
        else:
            all_videos = mixed_videos
            all_label = label

        return all_videos, all_label

    def sample_frames_from_file(self):
        self.frame_path_list_loaded = []
        if self.sample_stra.startswith('sorted'):
            all_prob = []
        for key in self.frame_prob_data.keys():
            if self.sample_stra.endswith('gt'):
                sample_prob = self.frame_prob_data[key][0]
            elif self.sample_stra.endswith('pred'):
                sample_prob = self.frame_prob_data[key][1]
            else:
                assert False, 'Invalid sampling strategy.'
            if self.sample_stra.startswith('prob'):
                if np.random.rand(1)[0] < sample_prob:
                    self.frame_path_list_loaded.append(key)
            elif self.sample_stra.startswith('thre'):
                if sample_prob >= self.prob_thre[0] and sample_prob <= self.prob_thre[1]:
                    self.frame_path_list_loaded.append(key)
            elif self.sample_stra.startswith('sorted'):
                self.frame_path_list_loaded.append(key)
                all_prob.append(sample_prob)
        if self.sample_stra.startswith('sorted'):
            if 'prop' in self.sample_stra:
                sorted_pair = sorted(zip(all_prob, self.frame_path_list_loaded))[int(self.prob_thre[0]*len(all_prob)):int(self.prob_thre[1]*len(all_prob))]
                self.frame_path_list_loaded = [x for _, x in sorted_pair]
            else:
                sorted_pair = sorted(zip(all_prob, self.frame_path_list_loaded))[::-1][:self.bank_size]
                self.frame_path_list_loaded = [x for _, x in sorted_pair]
        print('### Loaded frames: %d'%len(self.frame_path_list_loaded))
    
    def load_frames(self):
        frame_path_bank = np.random.permutation(self.frame_path_list_loaded)[:self.bank_size]
        print('### Sample frames: %d'%len(frame_path_bank))
        bank = []
        for frame_path in frame_path_bank:
            if self.read_from == 'frame':
                full_path = os.path.join(self.video_path, frame_path)
                ## follow the mmaction loading processes
                img_bytes = file_client.get(full_path)
                cur_frame = mmcv.imfrombytes(img_bytes, channel_order='rgb')
            elif self.read_from == 'video':
                full_path = os.path.join(self.video_path, frame_path.split('//')[0])
                which_frame = int(frame_path.split('//')[1])
                cur_video = decord.VideoReader(full_path)
                cur_frame = cur_video[which_frame].asnumpy()
            resize_frame = (cv2.resize(cur_frame.astype(float), dsize=self.frame_size) - self.frame_mean) / self.frame_std
            resize_frame = np.expand_dims(np.transpose(resize_frame, (2,0,1)), axis=0)
            bank.append(resize_frame)
        self.bank = torch.from_numpy(np.stack(bank, axis=0)).type(torch.FloatTensor).cuda()


@BLENDINGS.register_module()
class StillMixFrameBankBlending_MixLabel(BaseMiniBatchBlending):
    """
    videos (torch.Tensor): Model input videos, float tensor with the
        shape of (B, N, C, H, W) or (B, N, C, T, H, W).
    label (torch.Tensor): Labels are converted from hard labels to soft labels.
    Hard labels are integer tensors with the shape of (B, 1) and all of the
    elements are in the range [0, num_classes - 1].
    Soft labels (probablity distribution over classes) are float tensors
    with the shape of (B, 1, num_classes) and all of the elements are in
    the range [0, 1].

    kwargs (dict, optional): Other keyword argument to be used to
        blending imgs and labels in a mini-batch.
    
    Args:
        num_classes (int): The number of classes.
        alpha (float): Parameters for Beta distribution.
    """

    def __init__(self, num_classes, video_path, frame_mean, frame_std, frame_size, frame_prob_file, bank_size, sample_stra='prob_gt', prob_thre=None, read_from='frame', update_num_epoch=None, alpha1=200.0, alpha2=200.0, prob_aug=0.5, mix_label=0.5, label_file=None):
        super().__init__(num_classes=num_classes)
        self.num_classes = num_classes
        self.beta = Beta(alpha1, alpha2)
        self.prob_aug = prob_aug
        self.video_path = video_path
        self.frame_mean = frame_mean
        self.frame_std = frame_std
        self.frame_size = frame_size
        self.frame_prob_data = pickle.load(open(frame_prob_file, 'rb'))
        self.bank_size = bank_size
        self.sample_stra = sample_stra
        self.prob_thre = prob_thre
        self.read_from = read_from
        self.update_num_epoch = update_num_epoch
        self.mix_label = mix_label
        self.label_file = label_file
        if self.update_num_epoch is not None:
            self.last_epoch = -1
            if self.sample_stra.startswith('thre'):
                self.sample_frames_from_file()
        else:
            self.sample_frames_from_file()
            self.load_frames()
        
    def do_blending(self, videos, label, **kwargs):
        if self.update_num_epoch is not None:
            assert 'epoch_runner' in kwargs and 'epoch' in kwargs
            epoch = kwargs['epoch']
            if epoch != self.last_epoch and epoch % self.update_num_epoch == 0:
                if self.sample_stra.startswith('prob'):
                    self.sample_frames_from_file()
                self.load_frames()
            self.last_epoch = epoch

        lam = self.beta.sample()
        indicator = len(videos.shape)
        if indicator == 5: ## 2D recognizer
            batch_size, num_clip, channel, h, w = videos.shape
            tmp_video = videos
            sample_from = num_clip
            num_sample = batch_size
            num_elements = channel*h*w
        elif indicator == 6: ## 3D recognizer
            batch_size, num_clip, channel, clip_len, h, w = videos.shape
            tmp_video = videos.permute(0,1,3,2,4,5).contiguous()
            sample_from = clip_len
            num_sample = batch_size*num_clip
            num_elements = channel*h*w
        else:
            assert False, "unsupport video size"

        ## randomly choose frames from bank
        sample_queue_index = torch.randint(0, self.bank_size, (batch_size,))
        sampled_frames = torch.stack([self.bank[sample_queue_index]]*(num_sample//batch_size), dim=1)
        sampled_label = self.bank_label[sample_queue_index]
        sampled_label = F.one_hot(sampled_label, num_classes=self.num_classes).unsqueeze(1)

        if indicator == 5:
            sampled_frames = sampled_frames.squeeze(1)
        elif indicator == 6:
            sampled_frames = sampled_frames.permute(0,1,3,2,4,5).contiguous()
        
        ## mix
        mixed_videos = lam * videos + (1 - lam) * sampled_frames
        mixed_labels = self.mix_label * label + (1 - self.mix_label) * sampled_label
        ## choose samples according to prob
        if self.prob_aug < 1:
            rand_batch = torch.rand(batch_size)
            aug_ind = torch.where(rand_batch < self.prob_aug)
            ori_ind = torch.where(rand_batch >= self.prob_aug)
            all_videos = torch.cat([mixed_videos[aug_ind], videos[ori_ind]], dim=0).contiguous()
            all_label = torch.cat([mixed_labels[aug_ind], label[ori_ind]], dim=0).contiguous()
        else:
            all_videos = mixed_videos
            all_label = mixed_labels

        return all_videos, all_label

    def sample_frames_from_file(self):
        self.frame_path_list_loaded = []
        if self.sample_stra.startswith('sorted'):
            all_prob = []
        for key in self.frame_prob_data.keys():
            if self.sample_stra.endswith('gt'):
                sample_prob = self.frame_prob_data[key][0]
            elif self.sample_stra.endswith('pred'):
                sample_prob = self.frame_prob_data[key][1]
            else:
                assert False, 'Invalid sampling strategy.'
            if self.sample_stra.startswith('prob'):
                if np.random.rand(1)[0] < sample_prob:
                    self.frame_path_list_loaded.append(key)
            elif self.sample_stra.startswith('thre'):
                if sample_prob >= self.prob_thre[0] and sample_prob <= self.prob_thre[1]:
                    self.frame_path_list_loaded.append(key)
            elif self.sample_stra.startswith('sorted'):
                self.frame_path_list_loaded.append(key)
                all_prob.append(sample_prob)
        if self.sample_stra.startswith('sorted'):
            sorted_pair = sorted(zip(all_prob, self.frame_path_list_loaded))[::-1][:self.bank_size]
            self.frame_path_list_loaded = [x for _, x in sorted_pair]
        print('### Loaded frames: %d'%len(self.frame_path_list_loaded))
    
    def load_frames(self):
        frame_path_bank = np.random.permutation(self.frame_path_list_loaded)[:self.bank_size]
        print('### Sample frames: %d'%len(frame_path_bank))
        bank = []
        bank_label = []
        label_map = {}
        assert self.label_file is not None
        with open(self.label_file) as f:
            data = f.readlines()
        for item in data:
            if self.read_from == 'frame':
                video_name, _, label = item.strip().split(' ')
            elif self.read_from == 'video':
                video_name, label = item.strip().split(' ')
                video_name = video_name.split('.')[0]
            label_map[video_name] = int(label)
        for frame_path in frame_path_bank:
            if self.read_from == 'frame':
                full_path = os.path.join(self.video_path, frame_path)
                ## follow the mmaction loading processes
                img_bytes = file_client.get(full_path)
                cur_frame = mmcv.imfrombytes(img_bytes, channel_order='rgb')
                video_name = frame_path.split('/')[0]
            elif self.read_from == 'video':
                full_path = os.path.join(self.video_path, frame_path.split('//')[0])
                which_frame = int(frame_path.split('//')[1])
                cur_video = decord.VideoReader(full_path)
                cur_frame = cur_video[which_frame].asnumpy()
                video_name = frame_path.split('//')[0]
            resize_frame = (cv2.resize(cur_frame.astype(float), dsize=self.frame_size) - self.frame_mean) / self.frame_std
            resize_frame = np.expand_dims(np.transpose(resize_frame, (2,0,1)), axis=0)
            bank.append(resize_frame)
            bank_label.append(label_map[video_name])
        self.bank = torch.from_numpy(np.stack(bank, axis=0)).type(torch.FloatTensor).cuda()
        self.bank_label = torch.from_numpy(np.stack(bank_label, axis=0)).type(torch.LongTensor).cuda()


@BLENDINGS.register_module()
class TemporalSelect(BaseMiniBatchBlending):
    """
    select one frame for test
    """

    def __init__(self, num_classes):
        super().__init__(num_classes=num_classes)

    def do_blending(self, videos, label, **kwargs):
        assert len(kwargs) == 0, f'unexpected kwargs for mixup {kwargs}'

        indicator = len(videos.shape)
        if indicator == 5: ## 2D recognizer
            batch_size, num_clip, channel, h, w = videos.shape
            tmp_video = videos
            sample_from = num_clip
            num_sample = batch_size
            num_elements = channel*h*w
        elif indicator == 6: ## 3D recognizer
            batch_size, num_clip, channel, clip_len, h, w = videos.shape
            tmp_video = videos.permute(0,1,3,2,4,5).contiguous()
            sample_from = clip_len
            num_sample = batch_size*num_clip
            num_elements = channel*h*w
        else:
            assert False, "unsupport video size"

        idx = torch.randint(sample_from, (1,))[0]
        static_video = torch.stack([tmp_video.view(num_sample, sample_from, num_elements)[:,idx,:]]*sample_from, dim=1).view(batch_size, num_sample//batch_size, sample_from, channel, h, w)

        if indicator == 5:
            static_video = static_video.squeeze(1)
        elif indicator == 6:
            static_video = static_video.permute(0,1,3,2,4,5).contiguous()

        return static_video, label