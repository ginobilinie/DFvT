# DFvT for Object Detection

This repo contains the supported code and configuration files to reproduce object detection results of [DFvT](https://github.com/ginobilinie/DFvT). It is based on [mmdetection](https://github.com/open-mmlab/mmdetection).

## Results and Models

### Mask R-CNN

| Backbone | Pretrained model | Lr Schd | box mAP | mask mAP | #params | FLOPs | config | log | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |:---: |
| DFvT-T | [model](https://github.com/ginobilinie/DFvT/releases/download/Detection/DFvT_tiny_coco_pretrained.pth) | 1x | 34.8 | 32.6 | 25M | 178G | [config](configs/DFvT/maskrcnn_DFvT_tiny_1x_coco.py) | [log](https://github.com/ginobilinie/DFvT/releases/download/Detection/log_DFvT_tiny_coco1x.txt) | [github](https://github.com/ginobilinie/DFvT/releases/download/Detection/DFvT_tiny_for_coco.pth) |
| DFvT-S| [model](https://github.com/ginobilinie/DFvT/releases/download/Detection/DFvT_small_coco_pretrained.pth) | 1x | 39.2 | 36.3 | 32M | 198G | [config](configs/DFvT/maskrcnn_DFvT_small_1x_coco.py) | [log](https://github.com/ginobilinie/DFvT/releases/download/Detection/log_DFvT_small_coco1x.txt)| [github](https://github.com/ginobilinie/DFvT/releases/download/Detection/DFvT_small_for_coco.pth ) |
| DFvT-B | [model](https://github.com/ginobilinie/DFvT/releases/download/Detection/DFvT_base_coco_pretrained.pth) | 1x | 43.4 | 39.0 | 58M | 242G | [config](configs/DFvT/maskrcnn_DFvT_base_1x_coco.py) | [log](https://github.com/ginobilinie/DFvT/releases/download/Detection/log_DFvT_base_coco1x.txt)| [github](https://github.com/ginobilinie/DFvT/releases/download/Detection/DFvT_base_for_coco.pth)|




### Installation

Please refer to [get_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md) for installation and dataset preparation.
For your convenience, I provide my own environment (anaconda) for reference:
```
Python: 3.7.10
PyTorch: 1.7.1
TorchVision: 0.8.2
OpenCV: 4.5.2
MMCV: 1.3.17
MMDetection: 2.11.0
```

### Training

To train a detector with pre-trained models, run:
```
# single-gpu training
python tools/train.py <CONFIG_FILE> --cfg-options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments]

# multi-gpu training
tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --cfg-options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments]
```
For example, to train a Mask R-CNN model with a `DFvT-B` backbone and 4 gpus, run:
```
tools/dist_train.sh configs/DFvT/maskrcnn_DFvT_base_1x_coco.py 4 --cfg-options model.pretrained=./DFvT_base_coco_pretrained.pth
```

### Inference
```
# single-gpu testing
python tools/test.py <CONFIG_FILE> <DET_CHECKPOINT_FILE> --eval bbox segm

# multi-gpu testing
tools/dist_test.sh <CONFIG_FILE> <DET_CHECKPOINT_FILE> <GPU_NUM> --eval bbox segm
```
For example, to test a Mask R-CNN model with a `DFvT-B` backbone and 4 gpus, run:
```
tools/dist_test.sh configs/DFvT/maskrcnn_DFvT_base_1x_coco.py ./DFvT_base_for_coco.pth 4 --eval bbox segm
```





### Apex (optional):
We use apex for mixed precision training by default. To install apex, run:
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
If you would like to disable apex, modify the type of runner as `EpochBasedRunner` and comment out the following code block in the [configuration files](configs/swin):
```
# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)
```

## Citing DFvT
```
@inproceedings{gao2022dfvt,
  title={Doubly-Fused ViT: Fuse Information from Vision Transformer Doubly with Local Representation},
  author={Gao, Li and Nie, Dong and Li, Bo and Ren, Xiaofeng},
  booktitle={Proceedings of the European Conference on Computer Vision},
  year={2022}
}
```

## Other Links

> **Image Classification**: See [DFvT for Image Classification](https://github.com/ginobilinie/DFvT).


