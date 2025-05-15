# project3

项目结构
```
/project/
├── data/                            # 数据集文件夹
│   ├── VOCdevkit/                   # 原始 VOC 数据集
│   │   └── VOC2007/
│   │       ├── Annotations/         # 原始 VOC 标注 XML 文件
│   │       ├── ImageSets/           # 数据集划分（train, val, test）
│   │       ├── JPEGImages/          # 原始图片文件
│   │       └── SegmentationClass/   # 语义分割标注
│   └── COCO/                        # 转换后的 COCO 格式数据集
│   │    ├── annotations/             # COCO 格式的标注 JSON 文件
│   │    │   ├── train2007.json
│   │    │   └── val2007.json
│   │    └── 
│   │        ├── train2007/           # 训练集图片
│   │        └── val2007/             # 验证集图片
│   └── voctococo.py                 # voc数据转coco数据脚本
│ 
├── mmdetection/                     # mmdetection 代码库
│   ├── configs/                     # 配置文件目录  主要修改配置
│   ├── tools/                       # 工具脚本
│
├── outputs/                         # 输出文件夹
│   ├── mask_rcnn/                   # Mask R-CNN 输出目录
│   │   ├── latest.pth               # 最新的模型权重
│   │   └── logs/                    # 训练日志和 TensorBoard 可视化
│   └── sparse_rcnn/                 # Sparse R-CNN 输出目录
│       ├── latest.pth               # 最新的模型权重
│       └── logs/                    # 训练日志和 TensorBoard 可视化
│
└── inference.py                     # 主要推理脚本
```


# 一、数据集
### 下载数据集
```
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
```
### 解压数据集
```
tar -xvf VOCtrainval_06-Nov-2007.tar
tar -xvf VOCtest_06-Nov-2007.tar
```
### voc数据集转化为coco数据集
进入data文件夹
```
python voctococo.py
```

### 数据集结构
```
├── data/
│   ├── VOCdevkit/
│   │   └── VOC2007/
│   │       ├── Annotations/        # 原始 VOC 格式的标注 XML 文件
│   │       ├── ImageSets/          # 数据集划分（train, val, test）
│   │       ├── JPEGImages/         # 原始图片文件
│   │       └── SegmentationClass/  # 语义分割标注
│   │
│   └── COCO/                       # 转换后的 COCO 格式数据集
│       ├── annotations/            # COCO 格式的标注 JSON 文件
│       │   ├── train2007.json
│       │   └── val2007.json
│       │
│       └── ├── train2007/          # 训练图片（从 VOC2007/JPEGImages 复制）
│           └── val2007/            # 验证图片（从 VOC2007/JPEGImages 复制）

```


# 二、mmdetection 安装及配置
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -U openmim
mim install mmcv-full
git clone https://github.com/open-mmlab/mmdetection.git .
pip install -r requirements.txt
pip install -e .
```

修改相关配置 如data classes
```
configs/_base_/datasets/voc0712.py
mmdet/datasets/coco.py
mmdet/core/evaluation/class_names.py
```

模型相关配置重新设定，放在指定文件夹
```
mask_rcnn.py
sparse_rcnn.py 
```

# 三、训练 
```
cd mmdetection
python tools/train.py configs/mask_rcnn/mask_rcnn.py
python tools/train.py configs/queryinst/sparse_rcnn.py
```

### tensorboard可视化 
```
tensorboard --logdir='.output/mask_rcnn_coco2007/tensorboard --port=6006'
tensorboard --logdir='.output/sparse_rcnn_coco2007/tensorboard --port=6007'
```

### inferience
```
python inference.py
```

![image](https://github.com/user-attachments/assets/43f33181-b16d-4548-ac32-986d15c80eec)
![image](https://github.com/user-attachments/assets/7af1d8d4-676f-49b5-a95a-79bfb2a0b714)


