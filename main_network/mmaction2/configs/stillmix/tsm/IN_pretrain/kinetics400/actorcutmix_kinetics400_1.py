_base_ = [
    '../../../../_base_/models/tsm_r50.py', '../../../../_base_/schedules/sgd_tsm_100e.py',
    '../../../../_base_/default_runtime.py'
]

# dataset settings
dataset_type_train = 'VideoDetDataset'
dataset_type_test = 'VideoDataset'
data_root = '/export/xxx/xxx/datasets/Kinetics_400'
det_file = '/export/xxx/xxx/datasets/Kinetics_400/human_detections/detection_maskrcnn_train.npy'
ann_file_train = '/export/xxx/xxx/work/dataset_config/Kinetics_400/lists/train.txt'
ann_file_val = '/export/xxx/xxx/work/dataset_config/Kinetics_400/lists/val.txt'
ann_file_test = '/export/xxx/xxx/work/dataset_config/Kinetics_400/lists/val.txt'
# ann_file_train = '/export/xxx/xxx/work/dataset_config/Kinetics_400/lists/train_train.txt'
# ann_file_val = '/export/xxx/xxx/work/dataset_config/Kinetics_400/lists/train_val.txt'
# ann_file_test = '/export/xxx/xxx/work/dataset_config/Kinetics_400/lists/train_val.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

model = dict(
    backbone=dict(num_segments=8),
    cls_head=dict(num_classes=400, num_segments=8, dropout_ratio=0.2),
    train_cfg=dict(_delete_=True, 
        blending=dict(type='ActorCutmixBlending', num_classes=400, prob_aug=0.25)),)

train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(type='DecordDecode'),
    dict(type='DetectionLoad', thres=0.4, offset=1),
    dict(type='ResizeWithBox', scale=(-1, 256)), 
    dict(type='RandomResizedCropWithBox'), 
    dict(type='FlipWithBox', flip_ratio=0.5), 
    dict(type='ResizeWithBox', scale=(224, 224), keep_ratio=False), 
    dict(type='BuildHumanMask'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatDetShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label', 'human_mask'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label', 'human_mask'])
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=4,
    train_dataloader=dict(drop_last=True),
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type_train,
        ann_file=ann_file_train,
        det_file=det_file,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type_test,
        ann_file=ann_file_val,
        data_prefix=data_root,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type_test,
        ann_file=ann_file_test,
        data_prefix=data_root,
        pipeline=test_pipeline))
evaluation = dict(
    interval=50, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4)

# learning policy
lr_config = dict(policy='step', step=[20, 40])
total_epochs = 50

# runtime settings
checkpoint_config = dict(interval=10)
work_dir = './work_dirs/stillmix_v2/tsm/IN_pretrain/kinetics400/actorcutmix/1'
