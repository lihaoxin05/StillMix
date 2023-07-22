_base_ = ['../../../../_base_/default_runtime.py']

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
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowFast',
        pretrained=None,
        resample_rate=8,  # tau
        speed_ratio=8,  # alpha
        channel_ratio=8,  # beta_inv
        slow_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained='torchvision://resnet50',
            lateral=True,
            conv1_kernel=(1, 7, 7),
            dilations=(1, 1, 1, 1),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(0, 0, 1, 1),
            norm_eval=False),
        fast_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=False,
            base_channels=8,
            conv1_kernel=(5, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
            norm_eval=False)),
    cls_head=dict(
        type='SlowFastHead',
        in_channels=2304,  # 2048+256
        num_classes=400,
        spatial_type='avg',
        dropout_ratio=0.5),
    # model training and testing settings
    train_cfg=dict(blending=dict(type='ActorCutmixBlending', num_classes=400, prob_aug=0.25)),
    test_cfg=dict(average_clips='prob'))

train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='DetectionLoad', thres=0.4, offset=1),
    dict(type='ResizeWithBox', scale=(-1, 256)), 
    dict(type='RandomResizedCropWithBox'), 
    dict(type='FlipWithBox', flip_ratio=0.5), 
    dict(type='ResizeWithBox', scale=(224, 224), keep_ratio=False), 
    dict(type='BuildHumanMask'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatDetShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label', 'human_mask'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label', 'human_mask'])
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=10,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
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
    type='SGD', lr=0.01, momentum=0.9,
    weight_decay=1e-4) 
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[20, 40])
total_epochs = 50

# runtime settings
checkpoint_config = dict(interval=10)
work_dir = './work_dirs/stillmix_v2/slowfast/IN_pretrain/kinetics400/actorcutmix/1'
find_unused_parameters = False
