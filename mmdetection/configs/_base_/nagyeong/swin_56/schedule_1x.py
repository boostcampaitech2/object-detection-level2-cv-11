# Scheduler
load_from="/opt/ml/object-detection-level2-cv-11/mmdetection/checkpoints/cascade_rcnn_r50_fpn_20e_coco_bbox_mAP-0.41_20200504_175131-e9872a90.pth"
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[27, 33])
runner = dict(type='EpochBasedRunner', max_epochs=40)

work_dir = '/opt/ml/object-detection-level2-cv-11/mmdetection/work_dirs/nagyeong/swin2'

