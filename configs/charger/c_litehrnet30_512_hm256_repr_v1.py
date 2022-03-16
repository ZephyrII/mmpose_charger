_base_ = ['../_base_/datasets/charger_tape.py']
log_level = 'INFO'
load_from = None
ex_name = "c_litehrnet_udp_512_hm256_repr_v1"
# resume_from = "/root/share/tf/mmpose_checkpoints/"+ex_name+"/epoch_8.pth"
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metric='acc', save_best='acc')
# work_dir = "/root/share/tf/mmpose_checkpoints/"+ex_name+"/"

hm_size=[256, 256]

optimizer = dict(
    type='Adam',
    lr=5e-5,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[2, 3, 6])
total_epochs = 20
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='NeptuneLoggerHook',
            init_kwargs=dict(
                # run="CHAR-236",
                # project="tnowak/charger")
                # mode="debug",
                project='charger',
                api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4NGRmMmFkNi0wMWNjLTQxY2EtYjQ1OS01YjQ0YzRkYmFlNGIifQ==",
                name=ex_name,
                tags=["LiteHRNet", "512", "HM256", "aug", "repr_cost", "trf"])
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
    pretrained=None,
    backbone=dict(
        type='LiteHRNet',
        in_channels=3,
        extra=dict(
            stem=dict(stem_channels=32, out_channels=32, expand_ratio=1),
            num_stages=3,
            stages_spec=dict(
                num_modules=(3, 8, 3),
                num_branches=(2, 3, 4),
                num_blocks=(2, 2, 2),
                module_type=('LITE', 'LITE', 'LITE'),
                with_fuse=(True, True, True),
                reduce_ratios=(8, 8, 8),
                num_channels=(
                    (40, 80),
                    (40, 80, 160),
                    (40, 80, 160, 320),
                )),
            with_head=True,
        )),
    keypoint_head=dict(
        type='TopdownHeatmapReprCostHead',
        in_channels=40,
        out_channels=channel_cfg['num_output_channels'],
        num_deconv_layers=1,
        num_deconv_filters=(256,),
        num_deconv_kernels=(4,),
        extra=dict(final_conv_kernel=1, hm_size=hm_size),
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
    heatmap_size=hm_size,
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
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale', 'bbox',
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
            'image_file', 'center', 'scale', 'rotation', 'bbox_score', 'bbox',
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