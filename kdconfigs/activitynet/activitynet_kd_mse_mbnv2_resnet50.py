# model settings
total_seg = 10 
sampled_seg = 6
model = dict(
    type='KDSampler2DRecognizer2D',
    use_sampler=True,
    resize_px=128,
    loss='mse',
    num_layers=0,
    num_segments=total_seg,
    num_test_segments=sampled_seg,
    softmax=False,
    sampler=dict(
        type='FlexibleMobileNetV2TSM',
        #pretrained='mmcls://mobilenet_v2',
        pretrained='modelzoo/anet_mobilenetv2_tsm_sampler_checkpoint.pth',
        is_sampler=False,
        shift_div=10,
        num_segments=10,
        total_segments=10),
    backbone=dict(
        type='ResNet50',
        ),
    cls_head=dict(
        type='R50Head',
        num_classes=200,
        in_channels=2048,
        frozen=True,
        final_loss=False,
        ))

# model training and testing settings
train_cfg = None
test_cfg = dict(average_clips='prob')
# dataset settings
dataset_type = 'RawframeDataset'
data_root = 'data/ActivityNet/anet_320p_frames'
data_root_val = 'data/ActivityNet/anet_320p_frames'
ann_file_train = 'data/ActivityNet/frameexit_anet_train_video.txt'
ann_file_val = 'data/ActivityNet/frameexit_anet_val_video_multi_label.txt'
ann_file_test = 'data/ActivityNet/frameexit_anet_val_video_multi_label.txt'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=total_seg, num_clips=1),
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
    dict(type='UniformSampleFrames', clip_len=total_seg, num_clips=1, test_mode=True),
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
    dict(type='UniformSampleFrames', clip_len=total_seg, num_clips=1, test_mode=True),
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
    videos_per_gpu=150,
    workers_per_gpu=6,
    val_dataloader=dict(videos_per_gpu=150, workers_per_gpu=6),
    test_dataloader=dict(videos_per_gpu=150, workers_per_gpu=6),
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
optimizer = dict(type='SGD', lr=(0.001 / 8) * (120 / 8 * 2), momentum=0.9, weight_decay=0.0001)
# this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0)
#lr_config = dict(policy='CosineAnnealing', warmup='linear', warmup_iters=1000, warmup_ratio=0.001, min_lr=0)
total_epochs = 100
checkpoint_config = dict(interval=1, max_keep_ckpts=5)

log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
# runtime settings
dist_params = dict(backend='nccl', port=29302)
log_level = 'INFO'
work_dir = './work_dirs/activitynet_kd_mse_mbnv2_tsm_resnet50'  # noqa: E501
adjust_parameters = dict(base_ratio=0.0, min_ratio=0., by_epoch=False, style='step')
evaluation = dict(interval=5, metrics=['mean_average_precision'])

# runtime settings
checkpoint_config = dict(interval=5)

#evaluation = dict(interval=1, metrics=['mean_average_precision'], gpu_collect=True)
# directly port classification checkpoint from FrameExit
#load_from = 'modelzoo/step_resize_tsm_mobilenetv2_1x1x10_100e_coin_rgb.pth'
load_from = 'modelzoo/anet_frameexit_classification_checkpoint.pth'
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
