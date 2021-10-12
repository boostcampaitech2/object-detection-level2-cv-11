# optimizer
load_from = "/opt/ml/object-detection-level2-cv-11/mmdetection/checkpoints/cascade_rcnn_r101_fpn_1x_coco_20200317-0b6a2fbf.pth"

# work_dir = '/opt/ml/detection/mmdetection/work_dirs/nagyeong/cascade'

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[12, 23])
runner = dict(type='EpochBasedRunner', max_epochs=48)
