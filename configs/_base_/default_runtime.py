# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from =None # "/home/soicroot/Downloads/Khan/PiPa/work_dirs/local-basic/240207_1440_gtaHR2csHR_hrda_s0_42cae/iter_4000.pth"
workflow = [('train', 1)]
cudnn_benchmark = True
