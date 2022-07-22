# DFvT for Image Classification

Official implementation of "Doubly-Fused ViT: Fuse Information from Vision Transformer Doubly with Local Representation".

Accepted by ECCV 2022.

> Authors: Li Gao, Dong Nie, Bo Li, Xiaofeng Ren.

## Model Zoo

|  model       | Top-1 Acc.(%) | #Params(M) | FLOPs(G)  | Link   |
| :-----------:  | :---------: | :----------: | :---------: | :----------: |
| DFvT-Tiny  | 72.95     | 4.0   | 0.3     | [model](https://github.com/ginobilinie/DFvT/releases/download/models/DFvT_T_7295.pth)/[log](https://github.com/ginobilinie/DFvT/releases/download/logs/DFvT_T_log.txt) |
| DFvT-Small| 78.29    | 11.2    | 0.8    | [model](https://github.com/ginobilinie/DFvT/releases/download/models/DFvT_S_7829.pth)/[log](https://github.com/ginobilinie/DFvT/releases/download/logs/DFvT_S_log.txt)   |
| DFvT-Base  | 81.98   | 37.3      | 2.5    | [model](https://github.com/ginobilinie/DFvT/releases/download/models/DFvT_B_8198.pth)/[log](https://github.com/ginobilinie/DFvT/releases/download/logs/DFvT_B_log.txt)|
## Prerequisite
>Creat a new conda environment

```
conda create -n DFvT python=3.7 -y
conda activate DFvT
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch
pip install timm==0.3.2 opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8

```
>install Apex

```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

>Prepare dataset

Download ImageNet-1K dataset from http://image-net.org/, then organize the folder as follows:

```
imagenet
├── train
│   ├── class1
│   │   ├── img1.jpeg
│   │   ├── img2.jpeg
│   │   └── ...
│   ├── class2
│   │   ├── img3.jpeg
│   │   └── ...
│   └── ...
└── val
    ├── class1
    │   ├── img4.jpeg
    │   ├── img5.jpeg
    │   └── ...
    ├── class2
    │   ├── img6.jpeg
    │   └── ...
    └── ...


```

## Training and Evaluation example

### Training from scratch
```
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> --master_port 12345  main.py \ 
--cfg <config-file> --data-path <imagenet-path> [--batch-size <batch-size-per-gpu> --output <output-directory> --tag <job-tag>]
```
For example, to evaluate the DFvT-S with 4 GPU:
```
python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345  main.py \
--cfg configs/small.yaml --data-path <imagenet-path> --batch-size 256
```

### Evaluation

To evaluate a pre-trained DFvT on ImageNet val, run:
```
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> --master_port 12345 main.py --eval \
--cfg <config-file> --resume <checkpoint> --data-path <imagenet-path> 
```
For example, to evaluate the DFvT-S with a single GPU:
```
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --eval \
--cfg configs/small.yaml --resume DFvT_S_7829.pth --data-path <imagenet-path>
```
### Throughput

```
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py \
--cfg <config-file> --data-path <imagenet-path> --batch-size 64 --throughput --amp-opt-level O0
```

## License
The code is heavily borrowed from [Swin-Transformer](https://github.com/microsoft/Swin-Transformer).

If you use this code in your research please consider citing
```
@inproceedings{gao2022dfvt,
  title={Doubly-Fused ViT: Fuse Information from Vision Transformer Doubly with Local Representation},
  author={Gao, Li and Nie, Dong and Li, Bo and Ren, Xiaofeng},
  booktitle={Proceedings of the European Conference on Computer Vision},
  year={2022}
}
```
