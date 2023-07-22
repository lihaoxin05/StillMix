_base_ = [
    '../../../../_base_/models/tsm_r50.py', '../../../../_base_/schedules/sgd_tsm_100e.py',
    '../../../../_base_/default_runtime.py'
]

# dataset settings
dataset_type_train = 'RawframeDetDataset'
dataset_type_test = 'RawframeDataset'
data_root = '/export/xxx/xxx/datasets/HMDB51/jpegs_256'
det_file = '/export/xxx/xxx/datasets/HMDB51/actor_detections.npy'
split = 1  # official train/test splits. valid numbers: 1, 2, 3
ann_file_train = f'/export/xxx/xxx/work/dataset_config/HMDB51/lists/trainlist0{split}.txt'
ann_file_val = f'/export/xxx/xxx/work/dataset_config/HMDB51/lists/testlist0{split}.txt'
ann_file_test = f'/export/xxx/xxx/work/dataset_config/HMDB51/lists/testlist0{split}.txt'
# ann_file_train = f'/export/xxx/xxx/work/dataset_config/HMDB51/lists/trainlist0{split}_train.txt'
# ann_file_val = f'/export/xxx/xxx/work/dataset_config/HMDB51/lists/trainlist0{split}_val.txt'
# ann_file_test = f'/export/xxx/xxx/work/dataset_config/HMDB51/lists/testlist0{split}.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

model = dict(
    backbone=dict(num_segments=8),
    cls_head=dict(num_classes=51, num_segments=8, dropout_ratio=0.2),
    train_cfg=dict(_delete_=True, 
        blending=dict(type='ActorCutmixBlending', num_classes=51, prob_aug=0.5)),)

train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(type='RawFrameDecode'),
    dict(type='DetectionLoad', thres=0.4),
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
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='RawFrameDecode'),
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
        filename_tmpl = 'frame{:06d}.jpg',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type_test,
        ann_file=ann_file_val,
        data_prefix=data_root,
        filename_tmpl = 'frame{:06d}.jpg',
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type_test,
        ann_file=ann_file_test,
        data_prefix=data_root,
        filename_tmpl = 'frame{:06d}.jpg',
        pipeline=test_pipeline))
evaluation = dict(
    interval=100, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-2)

# learning policy
lr_config = dict(policy='step', step=[40, 80])
total_epochs = 100

# runtime settings
checkpoint_config = dict(interval=100)
work_dir = './work_dirs/stillmix_v2/tsm/IN_pretrain/hmdb51/actorcutmix/1'
