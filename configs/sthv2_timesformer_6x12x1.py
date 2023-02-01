# model settings
model = dict(
    type='Recognizer3D',
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
        norm_cfg=dict(type='LN', eps=1e-6)),
    cls_head=dict(type='TimeSformerHead', num_classes=174, in_channels=768),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))

# dataset settings
dataset_type = 'RawframeDataset'
data_root = 'data/sthv2/rawframes'
data_root_val = 'data/sthv2/rawframes'
ann_file_train = 'data/sthv2/sthv2_train_list_rawframes.txt'
ann_file_val = 'data/sthv2/sthv2_test_list_rawframes.txt'
ann_file_test = 'data/sthv2/sthv2_test_list_rawframes.txt'

img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=6, frame_interval=12, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='RandomRescale', scale_range=(256, 320)),
    dict(type='RandomCrop', size=224),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=6,
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
        clip_len=6,
        frame_interval=12,
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
train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=10, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=10, num_clips=1, test_mode=True),
    dict(type='RawFrameDecode'),
    # follow FrameExit
    # https://github.com/Qualcomm-AI-research/FrameExit/blob/main/config/activitynet_inference_2d.yml#L20-L21
    dict(type='Resize', scale=(-1, 224)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='UniformSampleFrames', clip_len=10, num_clips=1, test_mode=True),
    dict(type='RawFrameDecode'),
    # follow FrameExit
    # https://github.com/Qualcomm-AI-research/FrameExit/blob/main/config/activitynet_inference_2d.yml#L20-L21
    dict(type='Resize', scale=(-1, 224)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

"""
data = dict(
    videos_per_gpu=80,
    workers_per_gpu=10,
    test_dataloader=dict(videos_per_gpu=80),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        num_classes=200,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        num_classes=200,
        data_prefix=data_root_val,
        pipeline=test_pipeline))

evaluation = dict(
    interval=5, metrics=['mean_average_precision'])

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.005 / 8 * 2 * 25 / 8,
    momentum=0.9,
    paramwise_cfg=dict(
        custom_keys={
            '.backbone.cls_token': dict(decay_mult=0.0),
            '.backbone.pos_embed': dict(decay_mult=0.0),
            '.backbone.time_embed': dict(decay_mult=0.0)
        }),
    weight_decay=1e-4,
    nesterov=True)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
dist_params = dict(backend='nccl', port=29722)
# learning policy
lr_config = dict(policy='step', step=[5, 10])
total_epochs = 15

# runtime settings
checkpoint_config = dict(interval=5)
work_dir = './work_dirs/timesformer_divST_6x12x1_15e_sthv2'
