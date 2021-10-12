_base_ = [
    'cascade_mask_rcnn_r101_fpn_1x_coco.py',
    'dataset.py',
    'schedule_1x.py',
    'default_runtime.py'
]
# fp16 = dict(loss_scale=dict(init_scale=1024))
# pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth'  # noqa
# model = dict(
#     type='MaskRCNN',
#     backbone=dict(
#         _delete_=True,
#         type='SwinTransformer',
#         embed_dims=96,
#         depths=[2, 2, 18, 2],
#         num_heads=[3, 6, 12, 24],
#         window_size=7,
#         mlp_ratio=4,
#         qkv_bias=True,
#         qk_scale=None,
#         drop_rate=0.,
#         attn_drop_rate=0.,
#         drop_path_rate=0.2,
#         patch_norm=True,
#         out_indices=(0, 1, 2, 3),
#         with_cp=False,
#         convert_weights=True,
#         init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
#     neck=dict(in_channels=[96, 192, 384, 768]))