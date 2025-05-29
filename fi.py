# 新配置继承了基本配置，并做了必要的修改
_base_ = '/cpfs01/projects-HDD/cfff-aad9fa3a0781_HDD/hz_24210980090/mmdetection/configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py'

# 我们还需要更改 head 中的 num_classes 以匹配数据集中的类别数
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=20), mask_head=dict(num_classes=20)))

# 修改数据集相关配置
data_root = '/cpfs01/projects-HDD/cfff-aad9fa3a0781_HDD/hz_24210980090/PASCAL_VOC2012/OpenDataLab___PASCAL_VOC2012/raw/train/VOCdevkit/VOC2012/JPEGImages/split_img'
metainfo = {
    'classes': ('aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor' ),
    'palette': [
    (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
    (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
    (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
    (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)
]
}
train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='/cpfs01/projects-HDD/cfff-aad9fa3a0781_HDD/hz_24210980090/PASCAL_VOC2012/OpenDataLab___PASCAL_VOC2012/raw/train/VOCdevkit/VOC2012/coco_train.json',
        data_prefix=dict(img='train/')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='/cpfs01/projects-HDD/cfff-aad9fa3a0781_HDD/hz_24210980090/PASCAL_VOC2012/OpenDataLab___PASCAL_VOC2012/raw/train/VOCdevkit/VOC2012/coco_val.json',
        data_prefix=dict(img='val/')))
test_dataloader = val_dataloader

# 修改评价指标相关配置
val_evaluator = dict(ann_file='/cpfs01/projects-HDD/cfff-aad9fa3a0781_HDD/hz_24210980090/PASCAL_VOC2012/OpenDataLab___PASCAL_VOC2012/raw/train/VOCdevkit/VOC2012/coco_val.json')
test_evaluator = val_evaluator
train_cfg = dict(
    type='EpochBasedTrainLoop',  # 训练循环的类型，请参考 https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py
    max_epochs=50,  # 最大训练轮次
    val_interval=1)
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))
# 使用预训练的 Mask R-CNN 模型权重来做初始化，可以提高模型性能
load_from = '/cpfs01/projects-HDD/cfff-aad9fa3a0781_HDD/hz_24210980090/mmdetection/model/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'