model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNet50',
        #freeze_all=False,
        ),
    cls_head=dict(
        type='R50Head2',
        ),
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))

optimizer = dict(
    type='SGD',
    constructor='TSMOptimizerConstructor',
    paramwise_cfg=dict(fc_lr5=True),
    lr=0.01 / 8 * 60 / 8 * 1,  # this lr is used for 8 gpus
    momentum=0.9,
    weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[40, 80])
total_epochs = 100
checkpoint_config = dict(interval=5)

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl', port=29902)
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
dataset_type = 'RawframeDataset'
data_root = 'data/ActivityNet/anet_320p_frames'
data_root_val = 'data/ActivityNet/anet_320p_frames'
ann_file_train = 'data/ActivityNet/anet_train_video.txt' 
ann_file_val = 'data/ActivityNet/anet_val_video_multi_label.txt' 
ann_file_test = 'data/ActivityNet/anet_val_video_multi_label.txt' 
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=6),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1,
        num_fixed_crops=13),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=6,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=6,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
dev_check = dict(
    check = True,
    size = 224,
    input_format='NCHW'
)

data = dict(
    videos_per_gpu=60,
    workers_per_gpu=15,
    val_dataloader=dict(videos_per_gpu=60, workers_per_gpu=4),
    test_dataloader=dict(videos_per_gpu=60, workers_per_gpu=4),
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
evaluation = dict(
    interval=5, metrics=['mean_average_precision'])
work_dir = './work_dirs/anet_r50_1x1x6_100e/'
load_from = 'modelzoo/anet_frameexit_classification_checkpoint.pth'
omnisource = False
module_hooks = []
