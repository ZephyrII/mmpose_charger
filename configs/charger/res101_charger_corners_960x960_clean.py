_base_ = ['../_base_/datasets/charger_tape.py']
log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metric='acc', save_best='acc')
optimizer = dict(
    type='Adam',
    lr=1e-5,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[2, 3])
total_epochs = 10
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='NeptuneLoggerHook',
            init_kwargs=dict(
                # mode="debug",
                project='charger',
                api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4NGRmMmFkNi0wMWNjLTQxY2EtYjQ1OS01YjQ0YzRkYmFlNGIifQ==",
                name="AMCS_resnet101_corners_clean")
            )
    ])

channel_cfg = dict(
    num_output_channels=4,
    dataset_joints=4,
    dataset_channel=[
        [0, 1, 2, 3],
    ],
    inference_channel=[
        0, 1, 2, 3])

# model settings
model = dict(
    type='TopDownCharger',
    pretrained='torchvision://resnet101',
    backbone=dict(type='ResNet', depth=101),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=2048,
        out_channels=channel_cfg['num_output_channels'],
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11))

data_cfg = dict(
    image_size=[960, 960],
    heatmap_size=[240, 240],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=False,
    det_bbox_thr=0.0,
    bbox_file='',
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='TopDownRandomFlip', flip_prob=0.5),
    # dict(
    #     type='TopDownHalfBodyTransform',
    #     num_joints_half_body=8,
    #     prob_half_body=0.3),
    # dict(
    #     type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=3),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs'
        ]),
]


val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    # dict(
    #     type="TopDownGenerateTarget",
    #     sigma=2,
    #     unbiased_encoding=True,
    #     target_type="GaussianHeatMap",
    # ),
    dict(type='TopDownGenerateTarget', sigma=3),
    dict(
        type='Collect',
        keys=['img', "target", "target_weight"],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs'
        ]),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs'
        ]),
]

data_root = '/root/share/tf/dataset/final_localization/corners_1.0'
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='TopDownChargerDataset',
        ann_dir=f'{data_root}/annotations/',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='TopDownChargerDataset',
        ann_dir=f'{data_root}/val/annotations/',
        img_prefix=f'{data_root}/val/images/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='TopDownChargerDataset',
        ann_dir=f'{data_root}/val/annotations/',
        img_prefix=f'{data_root}/val/images/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
)