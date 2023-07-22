# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import pickle

import mmcv
import numpy as np
import torch

from mmaction.datasets.pipelines import Compose
from mmaction.models import build_model
from mmcv import Config

#####
# python untrimmed_video_feature_extraction.py config_file ckpt_file --output-prefix output_dir
#####

def parse_args():
    parser = argparse.ArgumentParser(description='Extract Feature')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('ckpt', help='checkpoint file')
    parser.add_argument('--output-prefix', default='', help='output prefix')
    parser.add_argument('--batch_size', type=int, default=64)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    # args.is_rgb = args.modality == 'RGB'
    # args.clip_len = 8
    # args.input_format = 'NCHW' if args.is_rgb else 'NCHW_Flow'
    # rgb_norm_cfg = dict(
    #     mean=[123.675, 116.28, 103.53],
    #     std=[58.395, 57.12, 57.375],
    #     to_bgr=False)
    # flow_norm_cfg = dict(mean=[128, 128], std=[128, 128])
    # args.img_norm_cfg = rgb_norm_cfg if args.is_rgb else flow_norm_cfg
    # args.f_tmpl = 'img_{:05d}.jpg' if args.is_rgb else 'flow_{}_{:05d}.jpg'
    # args.in_channels = args.clip_len * (3 if args.is_rgb else 2)

    # define the data pipeline for Untrimmed Videos
    data_pipeline = cfg.test_pipeline
    data_pipeline = Compose(data_pipeline)


    model = build_model(cfg.model)
    # load pretrained weight into the feature extractor
    state_dict = torch.load(args.ckpt)['state_dict']
    model.load_state_dict(state_dict, strict=True)
    print('Load model: ', args.ckpt)
    model = model.cuda()
    model.eval()

    data = open(cfg.ann_file_test).readlines()
    data = [x.strip() for x in data]

    # enumerate Untrimmed videos, extract feature from each of them
    prog_bar = mmcv.ProgressBar(len(data))
    if not osp.exists(args.output_prefix):
        os.system(f'mkdir -p {args.output_prefix}')

    for item in data:
        frame_dir, length, _ = item.split()
        output_file = osp.basename(frame_dir) + '.npy'
        frame_dir = osp.join(cfg.data_root, frame_dir)
        output_file = osp.join(args.output_prefix, output_file)
        assert output_file.endswith('.npy')
        length = int(length)

        # prepare a pseudo sample
        tmpl = dict(
            frame_dir=frame_dir,
            total_frames=length,
            filename_tmpl=cfg.filename_tmpl,
            start_index=1,
            modality=cfg.modality)
        sample = data_pipeline(tmpl)
        imgs = sample['imgs']
        shape = imgs.shape
        # the original shape should be N_seg * C * H * W, resize it to N_seg *
        # 1 * C * H * W so that the network return feature of each frame (No
        # score average among segments)
        if cfg.input_format == 'NCHW':
            clip_len = cfg.clip_len
            num_clips = shape[0] // clip_len
            imgs = imgs.reshape((num_clips, clip_len) + shape[1:])

        def forward_data(model, data):
            # chop large data into pieces and extract feature from them
            results = []
            start_idx = 0
            num_clip = data.shape[0]
            while start_idx < num_clip:
                with torch.no_grad():
                    part = data[start_idx:start_idx + args.batch_size]
                    part = part.cuda()
                    feat = model.forward(part, return_loss=False)
                    results.append(feat)
                    start_idx += args.batch_size
            return np.concatenate(results)

        feat = forward_data(model, imgs)
        np.save(output_file, feat)
        prog_bar.update()


if __name__ == '__main__':
    main()
