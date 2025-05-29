import os
import json
import torch
from mmengine.config import Config
from mmengine.registry import DATA_SAMPLERS
from mmdet.registry import DATASETS
from mmdet.utils import register_all_modules
from mmdet.apis import init_detector
from mmengine.dataset import pseudo_collate
from torch.utils.tensorboard import SummaryWriter

register_all_modules()

# 定义PASCAL VOC类别元信息
VOC_METAINFO = {
    'classes': (
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ),
    'palette': [
        (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
        (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
        (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
        (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)
    ]
}

def main():
    # 初始化TensorBoard
    writer = SummaryWriter(log_dir='/cpfs01/projects-HDD/cfff-aad9fa3a0781_HDD/hz_24210980090/val_loss_logdir')
    
    # 配置加载（使用FPN结构配置文件）
    config_path = '/cpfs01/projects-HDD/cfff-aad9fa3a0781_HDD/hz_24210980090/mmdetection/configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py'
    config = Config.fromfile(config_path)
    
    # ============= 关键修改1：模型配置 =============
    config.model.roi_head.bbox_head.num_classes = 20
    config.model.roi_head.mask_head.num_classes = 20
    
    # ============= 关键修改2：数据集配置 =============
    # 数据根目录
    data_root = '/cpfs01/projects-HDD/cfff-aad9fa3a0781_HDD/hz_24210980090/PASCAL_VOC2012/OpenDataLab___PASCAL_VOC2012/raw/train/VOCdevkit/VOC2012/'
    
    # 验证集配置
    config.val_dataloader.dataset.update(
        type='CocoDataset',  # 明确指定数据集类型
        metainfo=VOC_METAINFO.copy(),  # 必须使用深拷贝
        data_root=data_root,
        ann_file='coco_val.json',  # 相对于data_root的路径
        data_prefix=dict(img='JPEGImages/split_img/val/'),
        filter_cfg=dict(filter_empty_gt=False)  # 确保加载空样本
    )
    
    config.val_dataloader.batch_size = 2
    config.val_dataloader.num_workers = 4

    # 初始化空模型
    base_model = init_detector(config, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # ============= 关键修改3：正确构建数据集 =============
    val_dataset = DATASETS.build(config.val_dataloader.dataset)
    
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=config.val_dataloader.batch_size,
        sampler=DATA_SAMPLERS.build(
            dict(type='DefaultSampler', shuffle=False),
            default_args=dict(dataset=val_dataset)
        ),
        num_workers=config.val_dataloader.num_workers,
        persistent_workers=True,
        collate_fn=pseudo_collate
    )

    # 权重文件处理
    weight_dir = '/cpfs01/projects-HDD/cfff-aad9fa3a0781_HDD/hz_24210980090/model_res'
    weight_files = sorted(
        [f for f in os.listdir(weight_dir) if f.startswith('epoch_')],
        key=lambda x: int(x.split('_')[1].split('.')[0])
    )

    # 结果存储
    all_results = {}
    
    for weight_file in weight_files:
        epoch = int(weight_file.split('_')[1].split('.')[0])
        weight_path = os.path.join(weight_dir, weight_file)
        
        try:
            # ============= 关键修改4：正确加载模型 =============
            model = init_detector(config, weight_path, device='cuda:0')
            model.CLASSES = VOC_METAINFO['classes']  # 覆盖类别名称
            
            # 计算验证损失
            epoch_losses = {}
            with torch.no_grad():
                for data_batch in val_dataloader:
                    processed = model.data_preprocessor(data_batch, training=True)
                    losses = model.loss(
                        batch_inputs=processed['inputs'],
                        batch_data_samples=processed['data_samples']
                    )
                    for k, v in losses.items():
                        current = epoch_losses.get(k, 0.0)
                        if isinstance(v, (list, tuple)):
                            epoch_losses[k] = current + sum(item.item() for item in v)
                        else:
                            epoch_losses[k] = current + v.item()
            
            # 计算平均损失并记录
            avg_losses = {k: v/len(val_dataloader) for k, v in epoch_losses.items()}
            for name, value in avg_losses.items():
                writer.add_scalar(f'Val/{name}', value, epoch)
            
            all_results[epoch] = avg_losses
            print(f"Epoch {epoch} completed. Losses: {avg_losses}")

        except Exception as e:
            print(f"Error processing {weight_file}: {str(e)}")
            continue

    writer.close()
    
    with open('val_losses.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("验证完成，结果已保存")

if __name__ == '__main__':
    main()