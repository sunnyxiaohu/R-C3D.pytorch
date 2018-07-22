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
import pdb
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
from model.tdcnn.resnet import resnet34, resnet50, resnet_tdcnn

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a R-C3D network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='thumos14', type=str)
  parser.add_argument('--net', dest='net',
                    help='main network c3d, i3d, res34, res50',
                    default='c3d', type=str)
  parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=1, type=int)
  parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=5, type=int)
  parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=100, type=int)
  parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                      help='number of iterations to check',
                      default=10000, type=int)

  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save models', default="./models",
                      nargs=argparse.REMAINDER)
  parser.add_argument('--nw', dest='num_workers',
                      help='number of worker to load data',
                      default=0, type=int)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')                     
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)

  parser.add_argument('--roidb_dir', dest='roidb_dir',
                      help='roidb_dir',
                      default="./preprocess")

# config optimization
  parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="sgd", type=str)
  parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=0.0001, type=float)
  parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=3, type=int)
  parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)

# set training session
  parser.add_argument('--s', dest='session',
                      help='training session',
                      default=1, type=int)

# resume trained model
  parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      default=False, type=bool)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load model',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load model',
                      default=13711, type=int)
# log and diaplay
  parser.add_argument('--use_tfboard', dest='use_tfboard',
                      help='whether use tensorflow tensorboard',
                      default=False, action='store_true')

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
      args.num_classes = 2 if cfg.AGNOSTIC else 21
      args.set_cfgs = ['ANCHOR_SCALES', '[2,4,5,6,8,9,10,12,14,16]', 'MAX_NUM_GT_TWINS', '20']

  args.cfg_file = "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)
  np.random.seed(cfg.RNG_SEED)

  cudnn.benchmark = True
  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.
  cfg.TRAIN.USE_FLIPPED = True
  cfg.USE_GPU_NMS = args.cuda
  roidb_path = args.roidb_dir + "/" + args.dataset + "/" + args.imdb_name
  roidb = get_roidb(roidb_path)
  train_size = len(roidb)

  print('{:d} roidb entries'.format(len(roidb)))

  output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  sampler_batch = sampler(train_size, args.batch_size)

  dataset = roibatchLoader(roidb, args.num_classes)

  dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                            sampler=sampler_batch, num_workers=args.num_workers)

  device = torch.device("cuda:0" if args.cuda else "cpu")

  if args.cuda:
    cfg.CUDA = True

  # initilize the network here.
  if args.net == 'c3d':
    tdcnn_demo = c3d_tdcnn(class_agnostic=cfg.AGNOSTIC, pretrained=True)
  elif args.net == 'i3d':
    tdcnn_demo = i3d_tdcnn(class_agnostic=cfg.AGNOSTIC, pretrained=True)
  elif args.net == 'res34':
    tdcnn_demo = resnet_tdcnn(depth=34, class_agnostic=cfg.AGNOSTIC, pretrained=True)
  elif args.net == 'res50':
    tdcnn_demo = resnet_tdcnn(depth=50, class_agnostic=cfg.AGNOSTIC, pretrained=True)
  else:
    print("network is not defined")
    pdb.set_trace()

  tdcnn_demo.create_architecture()

  lr = cfg.TRAIN.LEARNING_RATE
  #tr_momentum = cfg.TRAIN.MOMENTUM
  #tr_momentum = args.momentum

  params = []
  for key, value in dict(tdcnn_demo.named_parameters()).items():
    if value.requires_grad:
      print(key)
      if 'bias' in key:
        params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
      else:
        params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

  if args.optimizer == "adam":
    lr = lr * 0.1
    optimizer = torch.optim.Adam(params)

  elif args.optimizer == "sgd":
    optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

  if args.resume:
    load_name = os.path.join(output_dir,
      'tdcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
    checkpoint = torch.load(load_name)
    args.session = checkpoint['session']
    args.start_epoch = checkpoint['epoch']
    tdcnn_demo.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr = optimizer.param_groups[0]['lr']
    if 'pooling_mode' in checkpoint.keys():
      cfg.POOLING_MODE = checkpoint['pooling_mode']
    print("loaded checkpoint %s" % (load_name))

  if args.mGPUs:
    tdcnn_demo = nn.DataParallel(tdcnn_demo)

  
  tdcnn_demo = tdcnn_demo.to(device)

  iters_per_epoch = int(train_size / args.batch_size)

  for epoch in range(args.start_epoch, args.max_epochs + 1):
    # setting to train mode
    tdcnn_demo.train()
    loss_temp = 0
    start = time.time()

    if epoch % (args.lr_decay_step + 1) == 0:
        adjust_learning_rate(optimizer, args.lr_decay_gamma)
        lr *= args.lr_decay_gamma

    #data_iter = iter(dataloader)
    for step, (video_data, gt_twins) in enumerate(dataloader):
      #data = next(data_iter)
      #video_data = data[0]
      #video_data.data.resize_(data[0].size()).copy_(data[0])
      #gt_twins.data.resize_(data[1].size()).copy_(data[1])
      video_data = video_data.to(device)
      gt_twins = gt_twins.to(device)
      
      tdcnn_demo.zero_grad()
      rois, cls_prob, twin_pred, \
      rpn_loss_cls, rpn_loss_twin, \
      RCNN_loss_cls, RCNN_loss_twin, \
      rois_label = tdcnn_demo(video_data, gt_twins)
      loss = rpn_loss_cls.mean() + rpn_loss_twin.mean() \
           + RCNN_loss_cls.mean() + RCNN_loss_twin.mean()          
      loss_temp += loss.item()

      # backward
      optimizer.zero_grad()
      loss.backward()
      # if args.net == "vgg16":
      #clip_gradient(tdcnn_demo, 100.)
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

        print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                                % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
        print("\t\t\tfg/bg=(%d/%d), gt_twins: %d, time cost: %f" % (fg_cnt, bg_cnt, gt_twins.size(1), end-start))
        print("\t\t\trpn_cls: %.4f, rpn_twin: %.4f, rcnn_cls: %.4f, rcnn_twin %.4f" \
                      % (loss_rpn_cls, loss_rpn_twin, loss_rcnn_cls, loss_rcnn_twin))
        if args.use_tfboard:
          info = {
            'loss': loss_temp,
            'loss_rpn_cls': loss_rpn_cls,
            'loss_rpn_twin': loss_rpn_twin,
            'loss_rcnn_cls': loss_rcnn_cls,
            'loss_rcnn_twin': loss_rcnn_twin
          }
          for tag, value in info.items():
            logger.scalar_summary(tag, value, step)

        loss_temp = 0
        start = time.time()

    if args.mGPUs:
      save_name = os.path.join(output_dir, 'tdcnn_{}_{}_{}.pth'.format(args.session, epoch, step))
      save_checkpoint({
        'session': args.session,
        'epoch': epoch + 1,
        'model': tdcnn_demo.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'pooling_mode': cfg.POOLING_MODE,
        'class_agnostic': cfg.AGNOSTIC,
      }, save_name)
    else:
      save_name = os.path.join(output_dir, 'tdcnn_{}_{}_{}.pth'.format(args.session, epoch, step))
      save_checkpoint({
        'session': args.session,
        'epoch': epoch + 1,
        'model': tdcnn_demo.state_dict(),
        'optimizer': optimizer.state_dict(),
        'pooling_mode': cfg.POOLING_MODE,
        'class_agnostic': cfg.AGNOSTIC,
      }, save_name)
    print('save model: {}'.format(save_name))

    end = time.time()
    print(end - start)
