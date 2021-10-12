# # Scheduler
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[27, 33])

# optimizer = dict(type='AdamW', lr=0.00004, weight_decay=0.01)
# optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
# lr_config = dict(
#     policy='CosineAnnealing',
#     warmup='linear',
#     warmup_iters=1000,
#     warmup_ratio=1.0 / 10,
#     min_lr=4e-6)

# optimizer = dict(
#     type='AdamW',
#     lr=0.0001,
#     betas=(0.9, 0.999),
#     weight_decay=0.05,
#     paramwise_cfg=dict(
#         custom_keys={
#             'absolute_pos_embed': dict(decay_mult=0.),
#             'relative_position_bias_table': dict(decay_mult=0.),
#             'norm': dict(decay_mult=0.)
#         }))
# optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
# lr_config = dict(warmup_iters=1000, step=[27, 33])

runner = dict(type='EpochBasedRunner', max_epochs=96)
work_dir = '/opt/ml/detection/mmdetection/work_dirs/jinwoo/cascade_final'



