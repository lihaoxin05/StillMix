### Support actorcutmix in video dataset
import copy
import numpy as np
import torch
import os.path as osp

from .builder import DATASETS
from .video_dataset import VideoDataset


@DATASETS.register_module()
class VideoDetDataset(VideoDataset):
    """Video with Det dataset for action recognition.
       Only valid in training time, not test-time behavior
    Args:
        det_file (str): Path to the human box detection result file.
    """

    def __init__(self, ann_file, det_file, pipeline, start_index=0, **kwargs):
        super().__init__(ann_file, pipeline, start_index=start_index, **kwargs)
        # Load human detection bbox
        if det_file is not None:
            self.load_detections(det_file)

    def load_detections(self, det_file):
        """Load human detection results and merge it with self.video_infos"""
        dets = np.load(det_file, allow_pickle=True).item()
        for idx in range(len(self.video_infos)):
            seq_name = osp.join(*self.video_infos[idx]['filename'].split('/')[-2:])
            self.video_infos[idx]['all_detections'] = dets[seq_name]

    def prepare_test_frames(self, idx):
        raise NotImplementedError
