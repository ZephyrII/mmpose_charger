_base_ = ['../_base_/datasets/charger_tape.py']
log_level = 'INFO'
load_from = None
ex_name = "c_hrnet32_udp_512_hm128"
# resume_from = "/root/share/tf/mmpose_checkpoints/"+ex_name+"/epoch_7.pth"
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metric='acc', save_best='acc')
work_dir = "/root/share/tf/mmpose_checkpoints/"+ex_name+"/"

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
    step=[2, 3, 6])
total_epochs = 10
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='NeptuneLoggerHook',
            init_kwargs=dict(
                # run="CHAR-237",
                # project="tnowak/charger")
                # mode="debug",
                project='charger',
                api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4NGRmMmFkNi0wMWNjLTQxY2EtYjQ1OS01YjQ0YzRkYmFlNGIifQ==",
                name=ex_name,
                tags=["HRNet32", "512", "HM128", "aug"])
            )
    ])

target_type = 'GaussianHeatmap'

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
    pretrained='https://download.openmmlab.com/mmpose/'
    'pretrain_models/hrnet_w32-36af842e.pth',
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3, 
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256))),
    ),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=32,
        out_channels=channel_cfg['num_output_channels'],
        num_deconv_layers=0,
        extra=dict(final_conv_kernel=1, ),
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=False,
        target_type=target_type,
        modulate_kernel=17,
        use_udp=True))

data_cfg = dict(
    image_size=[512, 512],
    heatmap_size=[128, 128],
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
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=5, scale_factor=0.7),
    dict(
        type="Albumentation",
        transforms=[
            dict(type="HueSaturationValue", p=0.7),
            dict(type="GaussianBlur", p=0.3),
            dict(type="MotionBlur", p=0.3),
            dict(type="ColorJitter", p=0.4),
            dict(type="GaussNoise", p=0.1),
            dict(type="JpegCompression", p=0.6, quality_lower=80),
            dict(type="RandomFog", p=0.1),
            dict(type="RandomRain", p=0.1),
        ],
    ),
    dict(type='TopDownAffine', use_udp=True),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='TopDownGenerateTarget',
        sigma=3,
        encoding='UDP',
        target_type=target_type),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),# channel_order='bgr'),
    dict(type='TopDownAffine', use_udp=True),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='TopDownGenerateTarget',
        sigma=3,
        encoding='UDP',
        target_type=target_type),
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
    dict(type='TopDownAffine', use_udp=True),
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
    samples_per_gpu=2,
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