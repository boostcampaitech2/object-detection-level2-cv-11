checkpoint_config = dict(max_keep_ckpts=10, interval=2)
# yapf:disable
log_config = dict(
    interval=325,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook',
            init_kwargs=dict(project='last', entity='carry-van',name='detectors_last'))
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
# load_from = None
resume_from = None
workflow = [('train', 1)]