# model settings
model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='MobileNetV2TSM',
        #pretrained='mmcls://mobilenet_v2',
        pretrained='modelzoo/mini_kinetics_mobilenetv2_tsm_sampler_checkpoint.pth',
        is_sampler=False,
        shift_div=10,
        num_segments=10,
        total_segments=10,
        freeze_all=True,
        ),
    cls_head=dict(
        type='TSMHead',
        num_segments=10,
        num_classes=200,
        in_channels=1280,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.5,
        init_std=0.001,
        is_shift=True),
    train_cfg = None,
    test_cfg = dict(average_clips='prob'),
    )

# model training and testing settings
# dataset settings
dataset_type = 'RawframeDataset'
data_root = 'data/kinetics400/new_rawframes_train'
data_root_val = 'data/kinetics400/new_rawframes_val'
ann_file_train = 'data/mini_kinetics/mini_kinetics_train_list_rawframes.txt'
ann_file_val = 'data/mini_kinetics/mini_kinetics_val_list_rawframes.txt'
ann_file_test = 'data/mini_kinetics/mini_kinetics_val_list_rawframes.txt'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=10, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(128, 128), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=10, num_clips=1, test_mode=True),
    dict(type='RawFrameDecode'),
    # follow FrameExit
    # https://github.com/Qualcomm-AI-research/FrameExit/blob/main/config/activitynet_inference_2d.yml#L20-L21
    dict(type='Resize', scale=(-1, 128)),
    dict(type='CenterCrop', crop_size=128),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='UniformSampleFrames', clip_len=10, num_clips=1, test_mode=True),
    dict(type='RawFrameDecode'),
    # follow FrameExit
    # https://github.com/Qualcomm-AI-research/FrameExit/blob/main/config/activitynet_inference_2d.yml#L20-L21
    dict(type='Resize', scale=(-1, 128)),
    dict(type='CenterCrop', crop_size=128),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

data = dict(
    videos_per_gpu=400,
    workers_per_gpu=4,
    val_dataloader=dict(videos_per_gpu=400, workers_per_gpu=4),
    test_dataloader=dict(videos_per_gpu=400, workers_per_gpu=4),
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
# optimizer
optimizer = dict(type='SGD', lr=(0.001 / 8) * (200 / 8 * 2), momentum=0.9, weight_decay=0.0001)
# this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0)
total_epochs = 100
checkpoint_config = dict(interval=5, max_keep_ckpts=5)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/minik_mbnv2'  # noqa: E501
adjust_parameters = dict(base_ratio=0.0, min_ratio=0., by_epoch=False, style='step')
evaluation = dict(
    interval=1, metrics=['top_k_accuracy'], gpu_collect=True)
# directly port classification checkpoint from FrameExit
load_from = None 
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
