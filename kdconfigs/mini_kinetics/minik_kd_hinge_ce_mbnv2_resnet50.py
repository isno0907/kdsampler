# model settings
num_gpu = 1
lr = 1e-1
batch_size = 120
total_seg = 10 
sampled_seg = 5
tcp = 29700

model = dict(
    type='KDSampler2DRecognizer2D',
    use_sampler=True,
    resize_px=128,
    loss='hinge',
    ce_loss=True,
    loss_lambda=0.9,
    gamma=0.03,
    num_segments=total_seg,
    num_test_segments=sampled_seg,
    num_classes=200,
    return_logit=False,
    softmax=True,
    temperature=0.3,
    dropout_ratio=0.2,
    sampler=dict(
        type='MobileNetV2',
        pretrained='modelzoo/mini_kinetics_mobilenetv2_tsm_sampler_checkpoint.pth',
        is_sampler=False,
        total_segments=total_seg),
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
train_cfg=None,
test_cfg=dict(average_clips='prob')

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


#img_norm_cfg = dict(
#    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_bgr=False)

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
    videos_per_gpu=batch_size,
    workers_per_gpu=8,
    val_dataloader=dict(videos_per_gpu=batch_size, workers_per_gpu=8),
    test_dataloader=dict(videos_per_gpu=batch_size, workers_per_gpu=8),
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
dev_check = dict(
    check = True,
    size = 224,
    input_format='NCHW'
)
# optimizer
optimizer = dict(type='SGD', lr=(lr / 8) * (batch_size / 40 * num_gpu / 8), momentum=0.9, weight_decay=0.0001)
# this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0)
#lr_config = dict(policy='CosineAnnealing', warmup='linear', warmup_iters=1500, warmup_ratio=0.001, min_lr=0)
total_epochs = 20
checkpoint_config = dict(interval=1, max_keep_ckpts=5)

log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
# runtime settings
dist_params = dict(backend='nccl', port=tcp)
log_level = 'INFO'
work_dir = './work_dirs/mini_kinetics_kd_hinge_ce_mbnv2_resnet50'  # noqa: E501
adjust_parameters = dict(base_ratio=0.0, min_ratio=0., by_epoch=False, style='step')
evaluation = dict(interval=1, metrics=['top_k_accuracy'],gpu_collect=True)

# runtime settings
checkpoint_config = dict(interval=5)

#evaluation = dict(interval=1, metrics=['mean_average_precision'], gpu_collect=True)
load_from = 'modelzoo/mini_kinetics_frameexit_classification_checkpoint.pth'
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
