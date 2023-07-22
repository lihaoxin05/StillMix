##### Thanks to https://github.com/vt-vl-lab/video-data-aug/blob/01667cdbd1b952f2510af3422beeeb76e0d9e15a/mmaction/datasets/rawframe_dataset_unlabeled.py #####
import copy
import numpy as np
import torch

from .pipelines import Compose
from .rawframe_dataset import RawframeDataset
from .builder import DATASETS


@DATASETS.register_module()
class RawframeDetDataset(RawframeDataset):
    """Rawframe with Det dataset for action recognition.
       Only valid in training time, not test-time behavior
    Args:
        det_file (str): Path to the human box detection result file.
    """

    def __init__(self,
                 ann_file,
                 det_file,
                 pipeline,
                 data_prefix=None,
                 test_mode=False,
                 filename_tmpl='img_{:05}.jpg',
                 with_offset=False,
                 multi_class=False,
                 num_classes=None,
                 start_index=1,
                 modality='RGB',
                 sample_by_class=False,
                 power=0.,
                 dynamic_length=False):
        assert modality == 'RGB'
        super().__init__(ann_file, pipeline, data_prefix, test_mode, filename_tmpl, with_offset, multi_class, num_classes, start_index, modality, sample_by_class, power, dynamic_length)

        # Load human detection bbox
        if det_file is not None:
            self.load_detections(det_file)

    def load_detections(self, det_file):
        """Load human detection results and merge it with self.video_infos"""
        dets = np.load(det_file, allow_pickle=True).item()
        for idx in range(len(self.video_infos)):
            seq_name = self.video_infos[idx]['frame_dir'].split('/')[-1].replace('v_HandstandPushups', 'v_HandStandPushups') ## video name inconsistent
            self.video_infos[idx]['all_detections'] = dets[seq_name]

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        results['start_index'] = self.start_index
        
        # prepare tensor in getitem
        if self.multi_class:
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot

        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        raise NotImplementedError
