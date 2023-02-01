# model settings
model = dict(
    type='Sampler2DRecognizer2D',
    num_segments=9,
    use_sampler=True,
    bp_mode='tsn',
    explore_rate=0.1,
    resize_px=128,
    #reverse=True,
    sampler=dict(
        type='MobileNetV2TSM',
        pretrained='modelzoo/task_resize_tsm_mobilenetv2_1x1x10_100e_coin_rgb_remap.pth',
        is_sampler=True,
        shift_div=10,
        num_segments=10,
        total_segments=10,
        ),
    backbone=dict(
        type='ResNet',
        #pretrained='torchvision://resnet50',
        #torchvision_pretrain=False,
        #revise_keys=[('backbone.', '')],
        frozen_stages=5,
        depth=50,
        norm_eval=True),
    cls_head=dict(
        type='TSMHead',
        num_classes=180,
        in_channels=2048,
        spatial_type='avg',
        num_segments=9,
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.5,
        init_std=0.001,
        is_shift=False,
        #pretrained='modelzoo/step_r50_1x1x6.pth',
        #revise_keys=[('cls_head.', '')],
        frozen=True,
        final_loss=False
        ))

# model training and testing settings
train_cfg = None
test_cfg = dict(average_clips=None)
# dataset settings
dataset_type = 'RawframeDataset'
data_root = 'data/coin/raw_videos'
data_root_val = 'data/coin/raw_videos'
ann_file_train = 'data/coin/annotation/coin_task_train_list_rawframes.txt' 
ann_file_val = 'data/coin/annotation/coin_task_test_list_rawframes.txt'
ann_file_test = 'data/coin/annotation/coin_task_test_list_rawframes.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    #dict(type='UniformSampleFrames', clip_len=10, num_clips=1),
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=10),
    dict(type='RawFrameDecode'),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    #dict(type='UniformSampleFrames', clip_len=10, num_clips=1, test_mode=True),
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=10),
    dict(type='RawFrameDecode'),
    # follow FrameExit
    # https://github.com/Qualcomm-AI-research/FrameExit/blob/main/config/activitynet_inference_2d.yml#L20-L21
    dict(type='Resize', scale=(-1, 224)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    #dict(type='UniformSampleFrames', clip_len=10, num_clips=1, test_mode=True),
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=10),
    dict(type='RawFrameDecode'),
    # follow FrameExit
    # https://github.com/Qualcomm-AI-research/FrameExit/blob/main/config/activitynet_inference_2d.yml#L20-L21
    dict(type='Resize', scale=(-1, 224)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

data = dict(
    videos_per_gpu=120,
    workers_per_gpu=4,
    val_dataloader=dict(videos_per_gpu=80, workers_per_gpu=4),
    test_dataloader=dict(videos_per_gpu=80, workers_per_gpu=4),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        multi_class=True,
        num_classes=180,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        multi_class=True,
        num_classes=180,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=(0.001 / 8) * (120 / 16 * 1 / 8), momentum=0.9, weight_decay=0.0001)
# this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0)
total_epochs = 50
checkpoint_config = dict(interval=1, max_keep_ckpts=5)

log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/task_coin_10to6_resnet50'  # noqa: E501
adjust_parameters = dict(base_ratio=0.0, min_ratio=0., by_epoch=False, style='step')
evaluation = dict(
    interval=1, metrics=['mean_average_precision'], gpu_collect=True)
# directly port classification checkpoint from FrameExit
#load_from = 'modelzoo/step_resize_tsm_mobilenetv2_1x1x10_100e_coin_rgb.pth'
#load_from = 'modelzoo/task_r50_1x1x10.pth'
load_from = None
again_load = 'modelzoo/task_r50_1x1x10.pth'
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
