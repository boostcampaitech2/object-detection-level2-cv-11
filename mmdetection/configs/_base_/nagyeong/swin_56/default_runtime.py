checkpoint_config = dict(max_keep_ckpts=15, interval=4)
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