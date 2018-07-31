# --------------------------------------------------------
# Pytorch R-C3D
# Licensed under The MIT License [see LICENSE for details]
# Written by Shiguang Wang, based on code from Huijuan Xu
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
import cv2
import pickle
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import pickle
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.twin_transform import clip_twins
from model.nms.nms_wrapper import nms
from model.rpn.twin_transform import twin_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.tdcnn.c3d import C3D, c3d_tdcnn
from model.tdcnn.i3d import I3D, i3d_tdcnn
from model.utils.blob import prep_im_for_blob, video_list_to_blob
from model.tdcnn.resnet import resnet34, resnet50, resnet_tdcnn

import pdb

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Test a R-C3D network')
  parser.add_argument('--dataset', dest='dataset',
                      help='test dataset',
                      default='thumos14', type=str)
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='cfgs/c3d.yml', type=str)
  parser.add_argument('--net', dest='net',
                      help='main network c3d, i3d, res34, res50',
                      default='c3d', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models', default="./models",
                      type=str)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=13711, type=int)
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_true')
  parser.add_argument('--roidb_dir', dest='roidb_dir',
                      help='roidb_dir',
                      default="./preprocess")
  args = parser.parse_args()
  return args
  
def get_roidb(path):
    data = pickle.load(open(path, 'rb'))
    return data

def _get_video_blob(roidb):
    """Builds an input blob from the videos in the roidb at the specified
    scales.
    """
    processed_videos = []
    item = roidb

    for key in item:
      print (key, ": ", item[key])
      
    video_length = cfg.TRAIN.LENGTH[0]  
    video = np.zeros((video_length, cfg.TRAIN.CROP_SIZE,
                        cfg.TRAIN.CROP_SIZE, 3))

    j = 0
    random_idx = [int((cfg.TRAIN.FRAME_SIZE[1]-cfg.TRAIN.CROP_SIZE) / 2), 
                      int((cfg.TRAIN.FRAME_SIZE[0]-cfg.TRAIN.CROP_SIZE) / 2)]
    if cfg.INPUT == 'video':
      for video_info in item['frames']:
          prefix = item['fg_name'] if video_info[0] else item['bg_name']
          for idx in xrange(video_info[1], video_info[2], video_info[3]):
            frame = cv2.imread('%s/image_%s.jpg'%(prefix,str(idx+1).zfill(5)))
            frame = prep_im_for_blob(frame, cfg.PIXEL_MEANS, tuple(cfg.TRAIN.FRAME_SIZE[::-1]),
                                     cfg.TRAIN.CROP_SIZE, random_idx)

            if item['flipped']:
                frame = frame[:, ::-1, :]

            video[j] = frame
            j = j + 1

    else:
        for video_info in item['frames']:
          prefix = item['fg_name'] if video_info[0] else item['bg_name']
          for idx in xrange(video_info[1], video_info[2]):
            frame = cv2.imread('%s/image_%s.jpg'%(prefix,str(idx+1).zfill(5)))
            frame = prep_im_for_blob(frame, cfg.PIXEL_MEANS, tuple(cfg.TRAIN.FRAME_SIZE[::-1]),
                                     cfg.TRAIN.CROP_SIZE, random_idx)

            if item['flipped']:
                frame = frame[:, ::-1, :]

            if DEBUG:
              cv2.imshow('frame', frame/255.0)
              cv2.waitKey(0)
              cv2.destroyAllWindows()

            video[j] = frame
            j = j + 1
    
    # padding for the same length
    while ( j < video_length):
      video[j] = frame
      j = j + 1
    processed_videos.append(video)

    # Create a blob to hold the input images
    blob = video_list_to_blob(processed_videos)

    return torch.from_numpy(blob)

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)
  
  cudnn.benchmark = True
  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  np.random.seed(cfg.RNG_SEED)
  if args.dataset == "thumos14":
      args.imdb_name = "train_data_25fps_flipped.pkl"
      args.imdbval_name = "val_data_25fps.pkl"
      args.num_classes = 2 if cfg.AGNOSTIC else 21
      args.set_cfgs = ['ANCHOR_SCALES', '[2,4,5,6,8,9,10,12,14,16]', 'MAX_NUM_GT_TWINS', '20', 'NUM_CLASSES', args.num_classes]
  elif args.dataset == "activitynet":
      args.imdb_name = "train_data_3fps_flipped.pkl"
      args.imdbval_name = "val_data_3fps.pkl"
      args.num_classes = 2 if cfg.AGNOSTIC else 201
      args.set_cfgs = ['ANCHOR_SCALES', '[1,2,3,4,5,6,7,8,10,12,14,16,20,24,28,32,40,48,56,64]', 'MAX_NUM_GT_TWINS', '20', 'NUM_CLASSES', args.num_classes]  

  args.cfg_file = "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)
    
  cfg.USE_GPU_NMS = args.cuda
  cfg.CUDA = args.cuda
  
  print('Using config:')
  pprint.pprint(cfg)  

  roidb_path = args.roidb_dir + "/" + args.dataset + "/" + args.imdbval_name
  roidb = get_roidb(roidb_path)
  cfg.TRAIN.USE_FLIPPED = False

  num_videos = len(roidb)

  print('{:d} roidb entries'.format(num_videos))

  input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(input_dir):
    raise Exception('There is no input directory for loading network from ' + input_dir)
  load_name = os.path.join(input_dir,
    'tdcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

  # initilize the network here.
  if args.net == 'c3d':
    tdcnn_demo = c3d_tdcnn(class_agnostic=cfg.AGNOSTIC, pretrained=False)
  elif args.net =='res34':
    tdcnn_demo = resnet_tdcnn(depth=34, class_agnostic=cfg.AGNOSTIC, pretrained=False)
  elif args.net =='res50':
    tdcnn_demo = resnet_tdcnn(depth=50, class_agnostic=cfg.AGNOSTIC, pretrained=False)
  else:
    print("network is not defined")
    pdb.set_trace()

  tdcnn_demo.create_architecture()

  print("load checkpoint %s" % (load_name))
  checkpoint = torch.load(load_name)
  tdcnn_demo.load_state_dict(checkpoint['model'])
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']


  print('load model successfully!')
  
  device = torch.device("cuda:0" if args.cuda else "cpu")
  
  if args.cuda:
    tdcnn_demo = tdcnn_demo.to(device)

  start = time.time()
  # TODO: Add restriction for max_per_video
  max_per_video = 0

  vis = args.vis

  if vis:
    thresh = 0.05
  else:
    thresh = 0.005

  #save_name = 'c3d_tdcnn_3'

  all_twins = [[[] for _ in xrange(num_videos)]
               for _ in xrange(args.num_classes)]

  #output_dir = get_output_dir(imdb, save_name)
  #dataset = roibatchLoader(roidb, args.num_classes)

  #dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
  #                          shuffle=False, num_workers=0,
  #                          pin_memory=True)
  #dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
  #                          shuffle=False, num_workers=0,
  #                          pin_memory=True)

  #data_iter = iter(dataloader)

  _t = {'im_detect': time.time(), 'misc': time.time()}

  tdcnn_demo.eval()
  empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))
  for i in xrange(num_videos):
      #data = next(data_iter)
      video_data = _get_video_blob(roidb[i])
      gt_twins = torch.Tensor(video_data.size(0), 0, 3)
      video_data = video_data.to(device)
      gt_twins = gt_twins.to(device)

      det_tic = time.time()
      rois, cls_prob, twin_pred, \
      rpn_loss_cls, rpn_loss_twin, \
      RCNN_loss_cls, RCNN_loss_twin, \
      rois_label = tdcnn_demo(video_data, gt_twins)

      scores = cls_prob.data
      twins = rois.data[:, :, 1:3]
      if cfg.TEST.TWIN_REG:
          # Apply bounding-twin regression deltas
          twin_deltas = twin_pred.data
          if cfg.TRAIN.TWIN_NORMALIZE_TARGETS_PRECOMPUTED:
          # Optionally normalize targets by a precomputed mean and stdev
            if cfg.AGNOSTIC:
                twin_deltas = twin_deltas.view(-1, 2) * torch.FloatTensor(cfg.TRAIN.TWIN_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.TWIN_NORMALIZE_MEANS).cuda()
                twin_deltas = twin_deltas.view(1, -1, 2)
            else:
                twin_deltas = twin_deltas.view(-1, 2) * torch.FloatTensor(cfg.TRAIN.TWIN_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.TWIN_NORMALIZE_MEANS).cuda()
                twin_deltas = twin_deltas.view(1, -1, 2 * args.num_classes)

          pred_twins = twin_transform_inv(twins, twin_deltas, 1)
          pred_twins = clip_twins(pred_twins, cfg.TRAIN.LENGTH[0], 1)
      else:
          # Simply repeat the twins, once for each class
          pred_twins = np.tile(twins, (1, scores.shape[1]))

      # pred_twins /= data[1][0][2]

      scores = scores.squeeze()
      pred_twins = pred_twins.squeeze()
      det_toc = time.time()
      detect_time = det_toc - det_tic
      misc_tic = time.time()
      if vis:
          #im = cv2.imread(imdb.video_path_at(i))
          #im2show = np.copy(im)
          pass
      for j in xrange(1, args.num_classes):
          inds = torch.nonzero(scores[:,j]>thresh).view(-1)
          # if there is det
          if inds.numel() > 0:
            cls_scores = scores[:,j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            if cfg.AGNOSTIC:
              cls_twins = pred_twins[inds, :]
            else:
              cls_twins = pred_twins[inds][:, j * 2:(j + 1) * 2]
            
            cls_dets = torch.cat((cls_twins, cls_scores.unsqueeze(1)), 1)
            # cls_dets = torch.cat((cls_twins, cls_scores), 1)
            cls_dets = cls_dets[order]
            keep = nms(cls_dets, cfg.TEST.NMS)
            if ( len(keep)>0 ):
              cls_dets = cls_dets[keep.view(-1).long()]
              print ("activity: ", j)
              print (cls_dets.cpu().numpy())
            if vis:
              #im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3)
              pass
            all_twins[j][i] = cls_dets.cpu().numpy()
          else:
            all_twins[j][i] = empty_array

      # Limit to max_per_video detections *over all classes*
      if max_per_video > 0:
          video_scores = np.hstack([all_twins[j][i][:, -1]
                                    for j in xrange(1, args.num_classes)])
          if len(video_scores) > max_per_video:
              video_thresh = np.sort(video_scores)[-max_per_video]
              for j in xrange(1, imdb.num_classes):
                  keep = np.where(all_twins[j][i][:, -1] >= video_thresh)[0]
                  all_twins[j][i] = all_twins[j][i][keep, :]

      misc_toc = time.time()
      nms_time = misc_toc - misc_tic
  
      print ('im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_videos, detect_time, nms_time))

      #sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
      #    .format(i + 1, num_videos, detect_time, nms_time))
      #sys.stdout.flush()

      if vis:
          #cv2.imwrite('result.png', im2show)
          #pdb.set_trace()
          #cv2.imshow('test', im2show)
          #cv2.waitKey(0)
          pass

  end = time.time()
  print("test time: %0.4fs" % (end - start))
