# A *Faster* Pytorch Implementation of R-C3D

## Introduction

This project is a *faster* pytorch implementation of R-C3D, aimed to accelerating the training of R-C3D object detection models. Recently, there are a number of good implementations:

* [rbgirshick/py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn), developed based on Pycaffe + Numpy

* [longcw/faster_rcnn_pytorch](https://github.com/longcw/faster_rcnn_pytorch), developed based on Pytorch + Numpy

* [endernewton/tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn), developed based on TensorFlow + Numpy

* [ruotianluo/pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn), developed based on Pytorch + TensorFlow + Numpy

* [huijuan/R-C3D](https://github.com/VisionLearningGroup/R-C3D)

During our implementing, we referred the above implementations, especailly [longcw/faster_rcnn_pytorch](https://github.com/longcw/faster_rcnn_pytorch). However, our implementation has several unique and new features compared with the above implementations:

* **It is pure Pytorch code**. We convert all the numpy implementations to pytorch!

* **It supports multiple GPUs training**. We use a multiple GPU wrapper (nn.DataParallel here) to make it flexible to use one or more GPUs, as a merit of the above two features.

* **It is memory efficient**. We limit the image aspect ratio, and group images with similar aspect ratios into a minibatch. As such, we can train resnet101 and VGG16 with batchsize = 4 (4 images) on a sigle Titan X (12 GB). When training with 8 GPU, the maximum batchsize for each GPU is 3 (Res101), totally 24.

* **It is faster**. Based on the above modifications, the training is much faster. We report the training speed on NVIDIA TITAN Xp in the tables below.

## Benchmarking

We benchmark our code thoroughly on three datasets: pascal voc, coco and imagenet-200, using two different network architecture: vgg16 and resnet101. Below are the results:

### What we are going to do

- [x] Support both python2 and python3 (great thanks to [cclauss](https://github.com/cclauss)).
- [ ] Add other main network support (i3d, resnet-3d)
- [ ] ~~Run systematical experiments on ActivityNet.~~
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

### Data Preparation

* **THUOMS14**: 
1. Download the data. Please follow the instructions in
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
### Pretrained Model

We used two pretrained models in our experiments, VGG and ResNet101. You can download these two models from:
* C3D: ucf101-caffe.pth
Download them and put them into the data/pretrained_model/.

**NOTE**. We compare the pretrained models from Pytorch and Caffe, and surprisingly find Caffe pretrained models have slightly better performance than Pytorch pretrained. We would suggest to use Caffe pretrained models from the above link to reproduce our results. 

**If you want to use pytorch pre-trained models, please remember to transpose images from BGR to RGB, and also use the same data transformer (minus mean and normalize) as used in pretrained model.**

### IMPORTANT NOTES: 
the introduction below are not well constructed yet.

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

## Pretrained-models
You have to convert the pretrained caffemodels to pytorch first.

## Train 

Before training, set the right directory to save and load the trained models. Change the arguments "save_dir" and "load_dir" in trainval_net.py and test_net.py to adapt to your environment.

To train a R-C3D model with vgg16 on pascal_voc, simply run:
```
CUDA_VISIBLE_DEVICES=$GPU_ID python trainval_net.py \
                   --dataset thumos14 --net c3d \
                   --bs $BATCH_SIZE --nw $WORKER_NUMBER \
                   --lr $LEARNING_RATE --lr_decay_step $DECAY_STEP \
                   --cuda
```
where 'bs' is the batch size with default 1.
Above, BATCH_SIZE and WORKER_NUMBER can be set adaptively according to your GPU memory size.

~~ For multiple GPUs (not test yet.)  Try: ~~
```
python trainval_net.py --dataset thumos14 --net c3d \
                       --bs 24 --nw 8 \
                       --lr $LEARNING_RATE --lr_decay_step $DECAY_STEP \
                       --cuda --mGPUs
```

Change dataset to "activivity" or 'charades' if you want to train on ActivityNet or Charades.

## Test

If you want to evlauate the detection performance of a pre-trained c3d model on thumos14 test set, simply run
```
python test_net.py --dataset thumos14 --net c3d \
                   --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
                   --cuda
```
Specify the specific model session, chechepoch and checkpoint, e.g., SESSION=1, EPOCH=3, CHECKPOINT=13711.

## Authorship

This project is equally contributed by [Shiguang Wang](https://github.com/sunnxiaohu.git), and many others (thanks to them!).

## Citation

    @article{swfaster2rc3d,
        Author = {Shiguang Wang and Jiang Cheng},
        Title = {A Faster Pytorch Implementation of R-C3D},
        Journal = {https://github.com/jwyang/faster-rcnn.pytorch},
        Year = {2018}
    } 
    
    @inproceedings{xu2017r,
        title={R-C3D: region convolutional 3d network for temporal activity detection},
        author={Xu, Huijuan and Das, Abir and Saenko, Kate},
        booktitle={IEEE Int. Conf. on Computer Vision (ICCV)},
        pages={5794--5803},
        year={2017}
    }
