# model settings
total_seg = 10 
sampled_seg = 6 
model = dict(
    type='KDSampler2DRecognizer3D',
    use_sampler=True,
    resize_px=128,
    loss='hinge',
    alpha=5.0,
    gamma=0,
    num_layers=0,
    num_segments=total_seg,
    num_test_segments=sampled_seg,
    softmax=False,
    sampler=dict(
        #type='FlexibleMobileNetV2TSM',
        type='MobileNetV2',
        pretrained='mmcls://mobilenet_v2',
        #pretrained='modelzoo/anet_mobilenetv2_tsm_sampler_checkpoint.pth',
        is_sampler=False,
        total_segments=total_seg),
    backbone=dict(
        type='TimeSformer',
        num_frames=6,
        img_size=224,
        patch_size=16,
        embed_dims=768,
        in_channels=3,
        dropout_ratio=0.,
        transformer_layers=None,
        attention_type='divided_space_time',
        norm_cfg=dict(type='LN', eps=1e-6),
        freeze=True
        ),
    cls_head=dict(
        type='TimeSformerHead',
        num_classes=200, 
        in_channels=768,
        frozen=True,
        ),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))

# model training and testing settings
# dataset settings
dataset_type = 'RawframeDataset'
data_root = 'data/ActivityNet/anet_320p_frames'
data_root_val = 'data/ActivityNet/anet_320p_frames'
ann_file_train = 'data/ActivityNet/anet_train_video.txt'
ann_file_val = 'data/ActivityNet/anet_val_video_multi_label.txt'
ann_file_test = 'data/ActivityNet/anet_val_video_multi_label.txt'

img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_bgr=False)

train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=total_seg, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
"""
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=10,
        frame_interval=43,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=10,
        frame_interval=43,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
"""

val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=total_seg, num_clips=1, test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='UniformSampleFrames', clip_len=total_seg, num_clips=1, test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

data = dict(
    videos_per_gpu=150,
    workers_per_gpu=5,
    val_dataloader=dict(videos_per_gpu=150, workers_per_gpu=4),
    test_dataloader=dict(videos_per_gpu=150, workers_per_gpu=4),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        multi_class=True,
        num_classes=200,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        multi_class=True,
        num_classes=200,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
dev_check = dict(
    check = True,
    size = 224,
    input_format='NCTHW'
)
# optimizer
optimizer = dict(type='SGD', lr=(0.0001 / 8) * (150 / 16 * 2 / 8), momentum=0.9, weight_decay=0.0001)
# this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
#lr_config = dict(policy='CosineAnnealing', min_lr=0)
lr_config = dict(policy='CosineAnnealing', warmup='linear', warmup_iters=1500, warmup_ratio=0.001, min_lr=0)
total_epochs = 100
checkpoint_config = dict(interval=1, max_keep_ckpts=5)

log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
# runtime settings
dist_params = dict(backend='nccl', port=29702)
log_level = 'INFO'
work_dir = './work_dirs/activitynet_kd_hinge_mbnv2_timesformer'  # noqa: E501
adjust_parameters = dict(base_ratio=0.0, min_ratio=0., by_epoch=False, style='step')
evaluation = dict(interval=5, metrics=['mean_average_precision'])

# runtime settings
checkpoint_config = dict(interval=5)

#evaluation = dict(interval=1, metrics=['mean_average_precision'], gpu_collect=True)
load_from = 'modelzoo/timesformer_6x100x1_anet.pth'
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
