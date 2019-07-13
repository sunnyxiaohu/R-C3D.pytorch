# A *Faster* Pytorch Implementation of R-C3D
## News:

We reorganized the code and make it faster and more convenient. 
The [old branch](https://github.com/sunnyxiaohu/R-C3D.pytorch/tree/201810) is with tag: 201810

## Introduction

This project is a *faster* pytorch implementation of R-C3D, aimed to accelerating the training of R-C3D temporal action detection models. 
During our implementing, we referred the below implementations: 
* [jwyang/faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch.git), developed based on Pytorch + TensorFlow + Numpy

* [huijuan/R-C3D](https://github.com/VisionLearningGroup/R-C3D), developed based on caffe, official implementation

Our implementation has several unique and new features compared with the above implementations:

* **It is pure Pytorch code**. We convert all the numpy implementations to pytorch!

* **It supports parallel training**. We use a multiple GPU wrapper (nn.DataParallel here) to make it flexible to use one or more GPUs, with batch size equal or bigger than 1.
* **It supports parallel test**. We use a multiple GPU wrapper (nn.DataParallel here) to make it flexible to use one or more GPUs, with batch size equal or bigger than 1. 
* **It is memory efficient**. We can train 3dresnet-18 and with batchsize = 3 (3 video buffer, buffer length=768) on a sigle 1080Ti (11 GB). When training with 8 GPU, the maximum batchsize for each GPU is 3 (Res18), totally 24.
* **It supports roibatchLoader for training and test**. The training and test process can use the the roibatchLoader.
* **It is faster**. Based on the above modifications, the training is much faster. We report the training speed on 1080Ti in the tables below.

## Benchmarking

We benchmark our code thoroughly on two action detection datasets: [Activitynet-v1.3](http://activity-net.org/) val subset, [THUMOS14](http://crcv.ucf.edu/THUMOS14/), using two different network architecture: C3D and [3d-resnet](https://github.com/kenshohara/3D-ResNets-PyTorch.git). Below are the results:
### Note!!!
The prev URLs of benchmark are not working anymore, we provide they here:
the benckmark models: [trained models](https://stduestceducn-my.sharepoint.com/:f:/g/personal/wangshiguang_std_uestc_edu_cn/Eom87B7Kv5BJi2RxIjdWTcYBE-W86FoJKHFZ06Fx1w2AJw?e=PSMYSy)

1). THUMOS14 (buffer length=768; pooled_lhw=4,2,2; anchor scales=2,4,5,6,8,9,10,12,14,16; fixed the 2 bottom layers)

model    | #GPUs | batch size | lr        | lr_decay | max_epoch     |  time/epoch | mem/GPU | mAP(IoU thresh=0.5)
---------|--------|-----|--------|-----|-----|-------|--------|-----
R-C3D published paper   | 1 | 1 | 1e-4 | | | 4.95 hr | ~700MB | 28.9
[C3D](~~https://stduestceducn-my.sharepoint.com/:u:/g/personal/wangshiguang_std_uestc_edu_cn/Efp9ps9rdvlCj6Ozk_878NUBl2MlbUU2wp-9SB2mieyOOw?e=bHYJhv~~)     | 2 | 2 | 1e-4 | 3   | 5   |  2.97 hr | 600MB   | 38.5


2). Activitynet-v1.3 (buffer length=768; pooled_lhw=4,2,2 for c3d, pooled_lhw=16,4,4 for 3d-resnet; anchor scales=1,1.25, 1.5,1.75, 2,2.5, 3,3.5, 4,4.5, 5,5.5, 6,7, 8,9,10,11,12,14,16,18,20,22,24,28,32,36,40,44,52,60,68,76,84,92,100;)

model    | #GPUs | batch size | lr        | lr_decay | max_epoch     |  time/epoch | mem/GPU | mAP(IoU thresh=0.5) | Average mAP(IoU thresh=0.5:0.05:0.95) 
---------|--------|-----|--------|-----|-----|-------|--------|-----|-----
R-C3D published paper   | 1 | 1 | 1e-4 | | | 4.95 hr | ~700MB | 26.45 | 13.33
[C3D](~~https://stduestceducn-my.sharepoint.com/:u:/g/personal/wangshiguang_std_uestc_edu_cn/EefOfbXO0M5EsDZjQsmA9LsBAoOKpDPeuFmXj-exlS0HKQ?e=PZla3F~~)     | 4 | 4 | 1e-4 | 6   | 8   |  2.97 hr | 600MB   | 28.8 | 15.1
[3d-resnet18](~~https://stduestceducn-my.sharepoint.com/:u:/g/personal/wangshiguang_std_uestc_edu_cn/EZQhlbLdTRZJtSIQacDtMgcBgQ8Uk3zDM4Uf2mY8unMxUQ?e=8DMeOw~~)     | 4 | 4 | 1e-4 | 6   | 8   |  1.29 hr | 255MB   | 26.2 | 14.2
[3d-resnet34](~~https://stduestceducn-my.sharepoint.com/:u:/g/personal/wangshiguang_std_uestc_edu_cn/ERjmmLCBYuBDmF0c3wDuhOgBtwfxQSiVY2zzpNRhulqMUA?e=UZhSt4~~)     | 4 | 4 | 1e-4 | 8   | 10   |  1.75 hr | 285MB   | 28.8 | 15.5
[3d-resnet50](~~https://stduestceducn-my.sharepoint.com/:u:/g/personal/wangshiguang_std_uestc_edu_cn/EX3MPCQQFNZNkKiI9_NU44oBu16JyerS7bYxRFTB2131Ww?e=Tenwv9~~)     | 4 | 4 | 1e-4 | 8   | 10   |  2.06 hr | ~800MB   | 30.4 | 16.6

### What we are going to do
- [x] Support pytorch-0.4.0.
- [x] Run systematical experiments on ActivityNet and THUMOS14
- [ ] Add other main network support (eco, i3d, resnet-3d)
- [ ] ~~Write a detailed report about the new stuffs in our implementations, and the quantitative results in our experiments.~~

## Preparation 


First of all, clone the code
```
git clone https://github.com/sunnyxiaohu/R-C3D.pytorch.git
```

Then, create a folder:
```
cd R-C3D.pytorch && mkdir data
```

### prerequisites

* Python 3.6
* Pytorch 0.4.0
* CUDA 8.0 or higher
* MATLAB (optionally)

### Data Preparation

* **THUOMS14**: 
1. Download the data.
2. Extract the video frames.
```
cd ./preprocess/thumos14
python generate_frames.py
```
3. Generate training and validation roidb
```
python generate_roidb_training.py
python generate_roidb_validation.py
```

* **ActivityNet**: 
1. Download the data. Please follow the instructions in
2. Extract the video frames.
```
cd ./preprocess/activitynet
python generate_frames.py
```
3. Generate training and validation roidb
```
python generate_roidb_training.py
python generate_roidb_validation.py
```
### Pretrained Model
####Note!!!
The prev URLs of pretrained models are not working anymore, we provide they here:
the pretrained models: [pretrained models](https://stduestceducn-my.sharepoint.com/:f:/g/personal/wangshiguang_std_uestc_edu_cn/EnGrfGdKoe1Ak0MX_qW5kPMBLuR3v1wnNr3DSGMdd5_MLg?e=aapzH4)

We used two pretrained models in our experiments, VGG and ResNet101. You can download these two models from:
* C3D-pretrained on Sports1M: ~~[OneDrive](https://stduestceducn-my.sharepoint.com/:u:/g/personal/wangshiguang_std_uestc_edu_cn/EY6MiTbnIjFKmAZBv8phHsEBW0XEd6PT5F2GMKVF9ttuXA?e=KEqkWR) and [BaiduYun](https://pan.baidu.com/s/1saNqGBkzZHwZpG-A5RDLVw)~~
* C3D-pretrained on ActivityNet: ~~[OneDrive](https://stduestceducn-my.sharepoint.com/:u:/g/personal/wangshiguang_std_uestc_edu_cn/EYkuRfVANqtHiTu5Y92vN2MBo-HEgkW3X5DwBCR99h4sYA?e=fgSxqy)~~
* 3d-resnet18 pretrained on Kinetics: ~~[OneDrive](https://stduestceducn-my.sharepoint.com/:u:/g/personal/wangshiguang_std_uestc_edu_cn/EU8I47N1UVRLpYa15gfoYnkBvcnZq-Q44P9Sl2qhDKT3NA?e=pG1MWp)~~
* 3d-resnet34 pretrained on Kinetics: ~~[OneDrive](https://stduestceducn-my.sharepoint.com/:u:/g/personal/wangshiguang_std_uestc_edu_cn/EbOI3QS9il5Olw0eJgZEhk0BYHHd623AL2GCHc6Rk9RpOQ?e=KsrVv4)~~
* 3d-resnet50 pretrained on Kinetics: ~~[OneDrive](https://stduestceducn-my.sharepoint.com/:u:/g/personal/wangshiguang_std_uestc_edu_cn/EXxvgN6ncjpOgcGnKkhc71sBssvSlY7OSxow-vHnN4svHg?e=V41C4D)~~
Download them and put them into the data/pretrained_model/.

**NOTE**. We would suggest to use Caffe pretrained models from the above link to reproduce our results.

**If you want to use pytorch pre-trained models, please remember to transpose images from BGR to RGB, and also use the same data transformer (minus mean and normalize) as used in pretrained model.**

### Compilation

As pointed out by [ruotianluo/pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn), choose the right `-arch` in `make.sh` file, to compile the cuda code:

  | GPU model  | Architecture |
  | ------------- | ------------- |
  | TitanX (Maxwell/Pascal) | sm_52 |
  | GTX 960M | sm_50 |
  | GTX 1080 (Ti) | sm_61 |
  | Grid K520 (AWS g2.2xlarge) | sm_30 |
  | Tesla K80 (AWS p2.xlarge) | sm_37 |
  
More details about setting the architecture can be found [here](https://developer.nvidia.com/cuda-gpus) or [here](http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)

Install all the python dependencies using pip:
```
pip install -r requirements.txt
```

Compile the cuda dependencies using following simple commands:

```
cd lib
sh make.sh
```

It will compile all the modules you need, including NMS, ROI_Temporal_Pooing. The default version is compiled with Python 3.6, please compile by yourself if you are using a different python version.

**As pointed out in this [issue](https://github.com/jwyang/faster-rcnn.pytorch/issues/16), if you encounter some error during the compilation, you might miss to export the CUDA paths to your environment.**

## Train 

Before training, set the right directory to save and load the trained models. Change the arguments "save_dir" and "load_dir" in trainval_net.py and test_net.py to adapt to your environment.

To train a R-C3D model with C3D on THUMOS14, simply run:
```
CUDA_VISIBLE_DEVICES=$GPU_ID ./script_train c3d thumos14 
                   --gpus 0 1 2 3 --bs 4 \
                   --nw $WORKER_NUMBER \
                   --lr $LEARNING_RATE --lr_decay_step $DECAY_STEP
```
where 'bs' is the batch size with default 1.
Above, BATCH_SIZE and WORKER_NUMBER can be set adaptively according to your GPU memory size.


Change dataset to "activivity" or 'charades' if you want to train on ActivityNet or Charades.
Change network to "res34" or "res18" if you want to use 3d-resnet34 network or 3d-resnet18 network

## Test

If you want to evaluate the detection performance of a c3d model on thumos14 test set, simply run
```
CUDA_VISIBLE_DEVICES=$GPU_ID ./script_test c3d thumos14 
                   --gpus 0 1 2 3 --bs 4 \
                   --nw $WORKER_NUMBER \
                   --checksession $SESSION1 --checkepoch $EPOCH --checkpoint $CHECKPOINT
```
Specify the specific model session, chechepoch and checkpoint, e.g., SESSION=1, EPOCH=3, CHECKPOINT=13711.

## Authorship

This project is contributed by [Shiguang Wang](https://github.com/sunnxiaohu.git), and many others (thanks to them!).

## Citation

    @article{swfaster2rc3d,
        Author = {Shiguang Wang and Jian Cheng},
        Title = {A Faster Pytorch Implementation of R-C3D},
        Journal = {https://github.com/sunnyxiaohu/R-C3D.pytorch.git},
        Year = {2018}
    } 
    
    @inproceedings{xu2017r,
        title={R-C3D: region convolutional 3d network for temporal activity detection},
        author={Xu, Huijuan and Das, Abir and Saenko, Kate},
        booktitle={IEEE Int. Conf. on Computer Vision (ICCV)},
        pages={5794--5803},
        year={2017}
    }
