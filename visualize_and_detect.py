
import argparse
import os
import os.path as osp
import random
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules

# 安全加载模型配置
def safe_model_loading():
    """临时修改torch.load以安全加载模型"""
    original_load = torch.load
    torch.load = lambda *args, **kwargs: original_load(*args, **kwargs, weights_only=False)
    return original_load

# 参数解析
def parse_arguments():
    parser = argparse.ArgumentParser(description='Mask R-CNN可视化对比工具')
    parser.add_argument('config', help='模型配置文件路径')
    parser.add_argument('checkpoint', help='训练好的权重文件路径')
    parser.add_argument('--test-images-dir', required=True, help='测试图像目录')
    parser.add_argument('--output-dir', default='visual_results', help='输出目录')
    parser.add_argument('--device', default='cuda:0', help='推理设备')
    parser.add_argument('--num-images', type=int, default=4, help='处理图像数量')
    parser.add_argument('--proposal-thr', type=float, default=0.1, help='候选框分数阈值')
    parser.add_argument('--det-thr', type=float, default=0.3, help='检测结果分数阈值')
    parser.add_argument('--max-proposals', type=int, default=50, help='最大候选框数量')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    return parser.parse_args()

# 可视化器类
class MaskRCNNVisualizer:
    """Mask R-CNN可视化工具"""
    
    VOC_CLASSES = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 
        'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.proposals = None
        self._register_rpn_hook()
        self.colors = plt.cm.Set3(np.linspace(0, 1, len(self.VOC_CLASSES)))
        
    def _register_rpn_hook(self):
        """注册RPN Hook用于获取候选框"""
        def hook_wrapper(module, inputs, outputs):
            if isinstance(outputs, tuple) and outputs[0]:
                self.proposals = outputs[0][0]  # 取第一个batch的proposals
        
        # 查找RPN head
        rpn_head = getattr(self.model, 'rpn_head', None)
        if rpn_head is None and hasattr(self.model, 'roi_head'):
            rpn_head = getattr(self.model.roi_head, 'rpn_head', None)
        
        if rpn_head:
            rpn_head.register_forward_hook(hook_wrapper)
    
    def process_image(self, img_path):
        """处理单张图像并返回结果"""
        # 重置proposals
        self.proposals = None
        
        # 进行推理
        result = inference_detector(self.model, img_path)
        
        # 获取最终预测结果
        pred = result.pred_instances
        valid = pred.scores >= self.args.det_thr
        detections = {
            'scores': pred.scores[valid].cpu().numpy(),
            'bboxes': pred.bboxes[valid].cpu().numpy(),
            'labels': pred.labels[valid].cpu().numpy(),
            'masks': getattr(pred, 'masks', None)
        }
        
        # 处理proposals
        proposals = self.proposals
        if proposals is not None:
            if torch.is_tensor(proposals):
                proposals = proposals.cpu().numpy()
            if len(proposals) > self.args.max_proposals:
                proposals = proposals[:self.args.max_proposals]
        
        return proposals, detections
    
    def visualize(self, img_path, output_dir):
        """主可视化流程"""
        # 读取图像
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img_name = osp.splitext(osp.basename(img_path))[0]
        
        # 处理图像
        proposals, detections = self.process_image(img_path)
        
        # 生成对比图
        self._create_proposal_comparison(img, proposals, detections, output_dir, img_name)
        self._create_segmentation_comparison(img, detections, output_dir, img_name)
        
        return img_name
    
    def _create_proposal_comparison(self, img, proposals, detections, output_dir, img_name):
        """生成候选框对比图"""
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        
        # 1. 原始图像
        axes[0].imshow(img)
        axes[0].set_title('Original Image', fontsize=16, fontweight='bold')
        axes[0].axis('off')
        
        # 2. RPN Proposals
        axes[1].imshow(img)
        self._draw_proposals(axes[1], proposals)
        axes[1].set_title(f'RPN Proposals\n({len(proposals) if proposals is not None else 0})', 
                         fontsize=16, fontweight='bold')
        axes[1].axis('off')
        
        # 3. Final Predictions
        axes[2].imshow(img)
        self._draw_detections(axes[2], detections)
        axes[2].set_title(f'Final Predictions\n({len(detections["bboxes"])})', 
                         fontsize=16, fontweight='bold')
        axes[2].axis('off')
        
        # 添加图例
        proposal_patch = Patch(color='blue', alpha=0.7, label='RPN Proposals')
        prediction_patch = Patch(color='red', alpha=0.7, label='Final Predictions')
        fig.legend(handles=[proposal_patch, prediction_patch], loc='upper center', 
                  bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=14)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        plt.savefig(osp.join(output_dir, f'{img_name}_proposals_vs_predictions.jpg'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Proposal comparison saved for: {img_name}")
    
    def _draw_proposals(self, ax, proposals):
        """绘制候选框"""
        if proposals is None or len(proposals) == 0:
            return
        
        for i, bbox in enumerate(proposals):
            if len(bbox) < 4:
                continue
                
            x1, y1, x2, y2 = bbox[:4]
            alpha = max(0.2, 1.0 - i / len(proposals))
            rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1.5, 
                           edgecolor='blue', facecolor='none', alpha=alpha)
            ax.add_patch(rect)
    
    def _draw_detections(self, ax, detections):
        """绘制检测结果"""
        for bbox, score, label in zip(detections['bboxes'], detections['scores'], detections['labels']):
            if len(bbox) < 4:
                continue
                
            x1, y1, x2, y2 = bbox[:4]
            color = self.colors[label % len(self.colors)]
            
            # 绘制边界框
            rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=3, 
                            edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            # 添加标签
            class_name = self.VOC_CLASSES[label] if label < len(self.VOC_CLASSES) else f'class_{label}'
            ax.text(x1, y1-5, f'{class_name}: {score:.2f}', 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8),
                   fontsize=12, color='white', weight='bold')
    
    def _create_segmentation_comparison(self, img, detections, output_dir, img_name):
        """创建检测与分割对比图"""
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        
        # 1. 原始图像
        axes[0].imshow(img)
        axes[0].set_title('Original Image', fontsize=16, fontweight='bold')
        axes[0].axis('off')
        
        # 2. 目标检测结果
        axes[1].imshow(img)
        self._draw_detections(axes[1], detections)
        axes[1].set_title(f'Object Detection\n({len(detections["bboxes"])})', 
                         fontsize=16, fontweight='bold')
        axes[1].axis('off')
        
        # 3. 实例分割结果
        axes[2].imshow(img)
        self._draw_segmentations(axes[2], detections)
        axes[2].set_title(f'Instance Segmentation\n({len(detections["masks"]) if detections["masks"] is not None else 0})', 
                         fontsize=16, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(osp.join(output_dir, f'{img_name}_detection_vs_segmentation.jpg'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Segmentation comparison saved for: {img_name}")
    
    def _draw_segmentations(self, ax, detections):
        """绘制分割结果"""
        if detections['masks'] is None:
            return
            
        masks = detections['masks']
        for i, (mask, score, label) in enumerate(zip(masks, detections['scores'], detections['labels'])):
            color = self.colors[label % len(self.colors)]
            
            # 创建彩色mask
            colored_mask = np.zeros((*mask.shape, 4))
            colored_mask[..., :3] = color[:3]
            colored_mask[..., 3] = mask * 0.6  # 设置透明度
            
            # 绘制mask
            ax.imshow(colored_mask)
            
            # 绘制边界框
            if i < len(detections['bboxes']):
                bbox = detections['bboxes'][i]
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = bbox[:4]
                    rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, 
                                    edgecolor=color, facecolor='none')
                    ax.add_patch(rect)
                    
                    # 添加标签
                    class_name = self.VOC_CLASSES[label] if label < len(self.VOC_CLASSES) else f'class_{label}'
                    ax.text(x1, y1-5, f'{class_name}: {score:.2f}', 
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8),
                           fontsize=12, color='white', weight='bold')

# 工具函数
def select_images(dir_path, num=4, seed=42):
    """随机选择指定数量的测试图像"""
    valid_ext = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = [osp.join(dir_path, f) for f in os.listdir(dir_path) 
             if osp.splitext(f)[1].lower() in valid_ext]
    
    if len(images) < num:
        print(f"Warning: Only found {len(images)} images, using all")
        return images
    
    random.seed(seed)
    return random.sample(images, num)

def create_summary(output_dir, args, processed):
    """生成总结报告"""
    report_path = osp.join(output_dir, 'summary.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Mask R-CNN Visualization Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Config: {args.config}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Test Images: {args.test_images_dir}\n")
        f.write(f"Output Dir: {args.output_dir}\n")
        f.write(f"Processed Images: {len(processed)}\n\n")
        
        f.write("Image Results:\n")
        for img in processed:
            f.write(f"- {img}\n")
            f.write(f"  Proposals vs Predictions: {img}_proposals_vs_predictions.jpg\n")
            f.write(f"  Detection vs Segmentation: {img}_detection_vs_segmentation.jpg\n")
    
    print(f"Summary saved to: {report_path}")

def main():
    args = parse_arguments()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 验证输入路径
    for path in [args.config, args.checkpoint, args.test_images_dir]:
        if not osp.exists(path):
            raise FileNotFoundError(f"Path does not exist: {path}")
    
    # 安全加载模型
    original_loader = safe_model_loading()
    try:
        register_all_modules()
        model = init_detector(args.config, args.checkpoint, device=args.device)
    finally:
        torch.load = original_loader
    
    # 初始化可视化器
    visualizer = MaskRCNNVisualizer(model, args)
    
    # 选择并处理图像
    selected = select_images(args.test_images_dir, args.num_images, args.seed)
    processed = []
    
    print(f"Processing {len(selected)} images:")
    for i, img_path in enumerate(selected, 1):
        print(f"  [{i}/{len(selected)}] {osp.basename(img_path)}")
        try:
            img_name = visualizer.visualize(img_path, args.output_dir)
            processed.append(img_name)
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    # 生成总结报告
    create_summary(args.output_dir, args, processed)
    print(f"\nVisualization completed! Results saved to: {args.output_dir}")

if __name__ == '__main__':
    main()