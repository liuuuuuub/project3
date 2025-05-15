from mmdet.apis import DetInferencer
import os

# 创建保存结果的目录
save_dir_1 = './outputs/mask_rcnn_coco2007/inference-seg/mask-1'
os.makedirs(save_dir_1, exist_ok=True)

# 初始化推理器
inferencer_1 = DetInferencer(
    weights='./outputs/mask_rcnn_coco2007/epoch_36.pth',
    device='cuda:0'
)
# 测试图像路径
test_images = [
    './data/VOCdevkit/VOC2007/JPEGImages/000004.jpg',
    './data/VOCdevkit/VOC2007/JPEGImages/000009.jpg',
    './data/VOCdevkit/VOC2007/JPEGImages/000014.jpg',
    './data/VOCdevkit/VOC2007/JPEGImages/000021.jpg',
    './data/VOCdevkit/VOC2007/JPEGImages/000133.jpg',
    './data/VOCdevkit/VOC2007/JPEGImages/000010.jpg',
    './data/VOCdevkit/VOC2007/JPEGImages/000012.jpg',
    './data/test/01.jpg',
    './data/test/02.jpg',
    './data/test/03.jpg'
]

# 执行推理
inferencer_1(test_images, out_dir=save_dir_1) 
 
