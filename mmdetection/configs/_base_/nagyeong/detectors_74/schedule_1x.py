# optimizer
load_from="/opt/ml/object-detection-level2-cv-11/mmdetection/checkpoints/detectors_cascade_rcnn_r50_1x_coco-32a10ba0.pth"

optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16,19])
runner = dict(type='EpochBasedRunner', max_epochs=24)
work_dir = '/opt/ml/object-detection-level2-cv-11/mmdetection/work_dirs/nagyeong/detectors'
