checkpoint_config = dict(max_keep_ckpts=2, interval=12)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook',
            init_kwargs=dict(project='last', entity='carry-van', name='cascade'))
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = "/opt/ml/detection/mmdetection/configs/_base_/jinwoo_/cascade_mask_rcnn_r101_fpn_mstrain_3x_coco_20210628_165236-51a2d363.pth"
resume_from = None
workflow = [('train', 1)]