_base_ = [
    '../../../../_base_/models/swin/swin_tiny.py', '../../../../_base_/default_runtime.py'
]

# dataset settings
dataset_type_train = 'VideoDetDataset'
dataset_type_test = 'VideoDataset'
data_root = '/media/storage3/xxx/xxx/datasets/Kinetics_400'
det_file = '/media/storage3/xxx/xxx/datasets/Kinetics_400/human_detections/detection_maskrcnn_train.npy'
ann_file_train = '/home/xxx/xxx/work/dataset_config/Kinetics_400/lists/train.txt'
ann_file_val = '/home/xxx/xxx/work/dataset_config/Kinetics_400/lists/val.txt'
ann_file_test = '/home/xxx/xxx/work/dataset_config/Kinetics_400/lists/val.txt'
# ann_file_train = '/home/xxx/xxx/work/dataset_config/Kinetics_400/lists/train_train.txt'
# ann_file_val = '/home/xxx/xxx/work/dataset_config/Kinetics_400/lists/train_val.txt'
# ann_file_test = '/home/xxx/xxx/work/dataset_config/Kinetics_400/lists/val.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

model=dict(backbone=dict(patch_size=(2,4,4), drop_path_rate=0.1), 
        cls_head=dict(num_classes=400),
        test_cfg=dict(max_testing_views=4),
        train_cfg=dict(blending=dict(type='ActorCutmixBlending', num_classes=400, prob_aug=0.25)),)

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
    dict(type='Flip', flip_ratio=0),
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
        num_clips=4,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=24,
    workers_per_gpu=2,
    train_dataloader=dict(drop_last=True),
    val_dataloader=dict(
        videos_per_gpu=1,
        workers_per_gpu=1
    ),
    test_dataloader=dict(
        videos_per_gpu=1,
        workers_per_gpu=1
    ),
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
    interval=30, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer
optimizer = dict(type='AdamW', lr=3.75e-4, betas=(0.9, 0.999), weight_decay=0.02,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'backbone': dict(lr_mult=0.1)}))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=2.5
)
total_epochs = 30

# runtime settings
checkpoint_config = dict(interval=10)
work_dir = './work_dirs/stillmix_v2/Swin-T/IN_pretrain/kinetics400/actorcutmix/1'
find_unused_parameters = False


# # do not use mmdet version fp16
# fp16 = None
# optimizer_config = dict(
#     type="DistOptimizerHook",
#     update_interval=4,
#     grad_clip=None,
#     coalesce=True,
#     bucket_size_mb=-1,
#     use_fp16=False,
# )
