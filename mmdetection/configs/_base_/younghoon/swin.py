_base_ = '/opt/ml/detection/mmdetection/configs/swin/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.py'
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth'  # noqa

data_root = '/opt/ml/detection/dataset/'
CLASSES = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

data = dict(
    samples_per_gpu=4,
    train=dict(
        classes=CLASSES,
        ann_file=data_root + 'train.json',
        img_prefix=data_root 
    ),
    val=dict(
        classes=CLASSES,
        ann_file=data_root + 'test.json',
        img_prefix=data_root 
    ),
    test=dict(
        classes=CLASSES,
        ann_file=data_root + 'test.json',
        img_prefix=data_root 
    )
)

evaluation=dict(interval=0)

model = dict(
    backbone=dict(
        depths=[2, 2, 18, 2],
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    rpn_head=dict(
        anchor_generator=dict(
            scales=[4,6,8]
        )
    ))