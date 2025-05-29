# mmde_Mask-R_CNN
## fi.py的使用：  
此文件为配置文件，在对VOC数据集修改为COCO格式化，根据自己数据集路径进行修改，输入如下代码进行训练：
```python
python tools/train.py path/to/fi.py
```
## train.py的使用：
用上传的文件对mmdetection/tools/train.py进行替换，此文件实现了训练过程中的可视化，并对epoch的权重模型进行保存。
## new_val.py的使用：
由于原来train过程中并没有输出验证集的loss，我们可以调用此文件对验证集上的loss进行可视化。
## visual_and_detect.py使用：
此文件可以根据已有模型权重对图片进行实例分割和目标检测。
## 权重模型地址： 
https://pan.baidu.com/s/1bL4uirr5ty2dwoN_28bnxw  
提取码：1Q7R
