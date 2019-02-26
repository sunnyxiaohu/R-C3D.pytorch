# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import time
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient

#from model.rpn.c3d import c3d
from model.tdcnn.c3d import C3D, c3d_tdcnn
from model.tdcnn.i3d import I3D, i3d_tdcnn
from model.tdcnn.resnet import resnet_tdcnn
from model.tdcnn.eco import eco_tdcnn

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a R-C3D network')
    parser.add_argument('--dataset', dest='dataset',default='thumos14', type=str, choices=['thumos14', 'activitynet'],
                      help='training dataset')
    parser.add_argument('--net', dest='net',default='c3d', type=str, choices=['c3d', 'res18', 'res34', 'res50', 'eco'],
                      help='main network c3d, i3d, res34, res50')
    parser.add_argument('--start_epoch', dest='start_epoch', default=1, type=int,
                      help='starting epoch')
    parser.add_argument('--epochs', dest='max_epochs', default=8, type=int,
                      help='number of epochs to train')
    parser.add_argument('--disp_interval', default=100, type=int,
                      help='number of iterations to display')    
    parser.add_argument('--save_dir', default="./models",nargs=argparse.REMAINDER,
                      help='directory to save models')
    parser.add_argument('--output_dir',default="./output",nargs=argparse.REMAINDER,
                      help='directory to save log file')                      
    parser.add_argument('--nw', dest='num_workers', default=12, type=int,
                      help='number of worker to load data')
    parser.add_argument('--gpus', dest='gpus', nargs='+', type=int, default=0,
                      help='gpu ids.')                     
    parser.add_argument('--bs', dest='batch_size', default=1, type=int,
                      help='batch_size')
    parser.add_argument('--roidb_dir', dest='roidb_dir',default="./preprocess",
                      help='roidb_dir')

    # config optimization
    parser.add_argument('--o', dest='optimizer',default="sgd", type=str,
                      help='training optimizer')
    parser.add_argument('--lr', dest='lr', default=0.0001, type=float,
                      help='starting learning rate')
    parser.add_argument('--lr_decay_step', dest='lr_decay_step', default=6, type=int,
                      help='step to do learning rate decay, unit is epoch')
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma', default=0.1, type=float,
                      help='learning rate decay ratio')

    # set training session
    parser.add_argument('--s', dest='session', default=1, type=int,
                      help='training session')

    # resume trained model
    parser.add_argument('--resume',default=False, action='store_true',
                      help='resume checkpoint or not')
    parser.add_argument('--checksession', default=1, type=int,
                      help='checksession to load model')
    parser.add_argument('--checkepoch', default=8, type=int,
                      help='checkepoch to load model')
    parser.add_argument('--checkpoint', default=9388, type=int,
                      help='checkpoint to load model')
    # log and display
    parser.add_argument('--use_tfboard',default=False, action='store_true',
                      help='whether use tensorflow tensorboard')

    args = parser.parse_args()
    return args


class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0,batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data

def get_roidb(path):
    data = pickle.load(open(path, 'rb'))
    return data

def train_net(tdcnn_demo, dataloader, optimizer, args):
    # setting to train mode
    tdcnn_demo.train()
    loss_temp = 0
    start = time.time()
        
    data_start = time.time()
    for step, (video_data, gt_twins, num_gt) in enumerate(dataloader):
        video_data = video_data.cuda()
        gt_twins = gt_twins.cuda()
        data_time = time.time()-data_start
        
        tdcnn_demo.zero_grad()
        rois, cls_prob, twin_pred, rpn_loss_cls, rpn_loss_twin, \
        RCNN_loss_cls, RCNN_loss_twin, rois_label = tdcnn_demo(video_data, gt_twins)
        loss = rpn_loss_cls.mean() + rpn_loss_twin.mean() \
           + RCNN_loss_cls.mean() + RCNN_loss_twin.mean()          
        loss_temp += loss.item()

        # backward
        optimizer.zero_grad()
        loss.backward()
        # if args.net == "vgg16": clip_gradient(tdcnn_demo, 100.)
        optimizer.step()

        if step % args.disp_interval == 0:
            end = time.time()
            if step > 0:
                loss_temp /= args.disp_interval

            loss_rpn_cls = rpn_loss_cls.mean().item()
            loss_rpn_twin = rpn_loss_twin.mean().item()
            loss_rcnn_cls = RCNN_loss_cls.mean().item()
            loss_rcnn_twin = RCNN_loss_twin.mean().item()
            fg_cnt = torch.sum(rois_label.data.ne(0))
            bg_cnt = rois_label.data.numel() - fg_cnt
            gt_cnt = num_gt.sum().item()

            print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                                    % (args.session, args.epoch, step+1, len(dataloader), loss_temp, args.lr))
            print("\t\t\tfg/bg=(%d/%d), gt_twins: %d, time cost: %f" % (fg_cnt, bg_cnt, gt_cnt, end-start))
            print("\t\t\trpn_cls: %.4f, rpn_twin: %.4f, rcnn_cls: %.4f, rcnn_twin %.4f" \
                          % (loss_rpn_cls, loss_rpn_twin, loss_rcnn_cls, loss_rcnn_twin))
            print("one step data time: %.4f" % (data_time))                          
            if args.use_tfboard:
                info = {
                'loss': loss_temp,
                'loss_rpn_cls': loss_rpn_cls,
                'loss_rpn_twin': loss_rpn_twin,
                'loss_rcnn_cls': loss_rcnn_cls,
                'loss_rcnn_twin': loss_rcnn_twin
                }
                for tag, value in info.items(): logger.scalar_summary(tag, value, step)

            loss_temp = 0
            start = time.time()
        data_start = time.time()
                    
    end = time.time()
    print(end - start)    
    
if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.use_tfboard:
        from model.utils.logger import Logger
        # Set the logger
        logger = Logger('./logs')

    if args.dataset == "thumos14":
        args.imdb_name = "train_data_25fps_flipped.pkl"
        args.imdbval_name = "val_data_25fps.pkl"
        args.num_classes = 21
        args.set_cfgs = ['ANCHOR_SCALES', '[2,4,5,6,8,9,10,12,14,16]', 'NUM_CLASSES', args.num_classes]
        #args.set_cfgs = ['ANCHOR_SCALES', '[2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56]', 'NUM_CLASSES', args.num_classes]
    elif args.dataset == "activitynet":
        args.imdb_name = "train_data_25fps_flipped.pkl" #_192.pkl"
        args.imdbval_name = "val_data_25fps.pkl"
        args.num_classes = 201
        #args.set_cfgs = ['ANCHOR_SCALES', '[1,2,3,4,5,6,7,8,10,12,14,16,20,24,28,32,40,48,56,64]', 'NUM_CLASSES', args.num_classes] / stride
        args.set_cfgs = ['ANCHOR_SCALES', '[1,1.25, 1.5,1.75, 2,2.5, 3,3.5, 4,4.5, 5,5.5, 6,7, 8,9,10,11,12,14,16,18,20,22,24,28,32,36,40,44,52,60,68,76,84,92,100]', 'NUM_CLASSES', args.num_classes]  

    args.cfg_file = "cfgs/{}_{}.yml".format(args.net, args.dataset)

    cfg.CUDA = True 
    cfg.USE_GPU_NMS = True

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    # for reproduce
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(cfg.RNG_SEED)

    cudnn.benchmark = True

    # train set
    roidb_path = args.roidb_dir + "/" + args.dataset + "/" + args.imdb_name
    roidb = get_roidb(roidb_path)

    print('{:d} roidb entries'.format(len(roidb)))

    model_dir = args.save_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    output_dir = args.output_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    #sampler_batch = sampler(train_size, args.batch_size)
    dataset = roibatchLoader(roidb)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                             num_workers=args.num_workers, shuffle=True)
      
    # initilize the network here.
    if args.net == 'c3d':
        tdcnn_demo = c3d_tdcnn(pretrained=True)
    elif args.net == 'res18':
        tdcnn_demo = resnet_tdcnn(depth=18, pretrained=True)
    elif args.net == 'res34':
        tdcnn_demo = resnet_tdcnn(depth=34, pretrained=True)
    elif args.net == 'res50':
        tdcnn_demo = resnet_tdcnn(depth=50, pretrained=True)
    elif args.net == 'eco':
        tdcnn_demo = eco_tdcnn(pretrained=True)        
    else:
        print("network is not defined")

    tdcnn_demo.create_architecture()
    print(tdcnn_demo)

    params = []
    for key, value in dict(tdcnn_demo.named_parameters()).items():
        if value.requires_grad:
            print(key)
            if 'bias' in key:
                params += [{'params':[value],'lr': args.lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                    'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params':[value],'lr': args.lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if args.optimizer == "adam":
        args.lr = args.lr * 0.1
        optimizer = torch.optim.Adam(params)

    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    if args.resume:
        load_name = os.path.join(model_dir,
          'tdcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
        checkpoint = torch.load(load_name)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch'] + 1
        tdcnn_demo.load_state_dict(checkpoint['model'])
        optimizer_tmp = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)
        optimizer_tmp.load_state_dict(checkpoint['optimizer'])
        args.lr = optimizer_tmp.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
            print("loaded checkpoint %s" % (load_name))

    if torch.cuda.is_available():
        tdcnn_demo = tdcnn_demo.cuda()
        if isinstance(args.gpus, int):
            args.gpus = [args.gpus]
        tdcnn_demo = nn.parallel.DataParallel(tdcnn_demo, device_ids = args.gpus)

    for epoch in range(args.start_epoch, args.max_epochs + 1):
        if epoch % (args.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            args.lr *= args.lr_decay_gamma
            
        args.epoch = epoch
        train_net(tdcnn_demo, dataloader, optimizer, args)

        if len(args.gpus) > 1:
            save_name = os.path.join(model_dir, 'tdcnn_{}_{}_{}.pth'.format(args.session, epoch, len(dataloader)))
            save_checkpoint({
                'session': args.session,
                'epoch': epoch,
                'model': tdcnn_demo.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'pooling_mode': cfg.POOLING_MODE
            }, save_name)
        else:
            save_name = os.path.join(model_dir, 'tdcnn_{}_{}_{}.pth'.format(args.session, epoch, len(dataloader)))
            save_checkpoint({
                'session': args.session,
                'epoch': epoch,
                'model': tdcnn_demo.state_dict(),
                'optimizer': optimizer.state_dict(),
                'pooling_mode': cfg.POOLING_MODE
            }, save_name)
        print('save model: {}'.format(save_name))
