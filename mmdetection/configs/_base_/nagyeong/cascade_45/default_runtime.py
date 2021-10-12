work_dir = '/opt/ml/object-detection-level2-cv-11/mmdetection/work_dirs/nagyeong/cascade'

# load_from = "/opt/ml/object-detection-level2-cv-11/mmdetection/checkpoints/cascade_mask_rcnn_r101_fpn_1x_coco_20200203-befdf6ee.pth"


checkpoint_config = dict(max_keep_ckpts=12, interval=4)
# yapf:disable
log_config = dict(
    interval=274,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook',
            init_kwargs=dict(project='o_nag', entity='carry-van'))
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
# load_from = None
resume_from = None
workflow = [('train', 1)]
