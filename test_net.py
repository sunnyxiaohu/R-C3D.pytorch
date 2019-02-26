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

#np.set_printoptions(threshold='nan')
DEBUG=False

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a R-C3D network')
    parser.add_argument('--dataset', dest='dataset',default='thumos14', type=str,
                      help='test dataset')
    parser.add_argument('--net', dest='net',default='c3d', type=str, choices=['c3d', 'res18', 'res34', 'res50', 'eco'],
                      help='main network c3d, i3d, res34, res50')
    parser.add_argument('--set', dest='set_cfgs', nargs=argparse.REMAINDER,
                      help='set config keys', default=None)
    parser.add_argument('--load_dir', dest='load_dir',type=str,
                      help='directory to load models', default="./models")                      
    parser.add_argument('--output_dir', dest='output_dir',type=str,
                      help='directory for the log files', default="./output")
    parser.add_argument('--cuda', dest='cuda', action='store_true',
                      help='whether use CUDA')
    parser.add_argument('--checksession', default=1, type=int,
                      help='checksession to load model')                      
    parser.add_argument('--checkepoch', default=1, type=int,
                      help='checkepoch to load network')                     
    parser.add_argument('--checkpoint', default=9388, type=int,
                      help='checkpoint to load network')
    parser.add_argument('--nw', dest='num_workers', default=8, type=int,
                      help='number of worker to load data')
    parser.add_argument('--bs', dest='batch_size', default=1, type=int,
                        help='batch_size, only support batch_size=1')
    parser.add_argument('--vis', dest='vis', action='store_true',
                      help='visualization mode')
    parser.add_argument('--roidb_dir', dest='roidb_dir', default="./preprocess",
                      help='roidb_dir')
    parser.add_argument('--gpus', dest='gpus', nargs='+', type=int, default=0,
                      help='gpu ids.')                        
    args = parser.parse_args()
    return args
  
def get_roidb(path):
    data = pickle.load(open(path, 'rb'))
    return data
    
def test_net(tdcnn_demo, dataloader, args):

    start = time.time()
    # TODO: Add restriction for max_per_video
    max_per_video = 0

    if args.vis:
        thresh = 0.05
    else:
        thresh = 0.005
    
    all_twins = [[[] for _ in xrange(args.num_videos)]
               for _ in xrange(args.num_classes)]

    _t = {'im_detect': time.time(), 'misc': time.time()}

    tdcnn_demo.eval()
    empty_array = np.transpose(np.array([[],[],[]]), (1,0))
  
    data_tic = time.time()
    for i, (video_data, gt_twins, num_gt, video_info) in enumerate(dataloader):
        video_data = video_data.cuda()
        gt_twins = gt_twins.cuda()
        batch_size = video_data.shape[0]
        data_toc = time.time()
        data_time = data_toc - data_tic

        det_tic = time.time()
        rois, cls_prob, twin_pred = tdcnn_demo(video_data, gt_twins)
#        rpn_loss_cls, rpn_loss_twin, \
#        RCNN_loss_cls, RCNN_loss_twin, rois_label = tdcnn_demo(video_data, gt_twins)

        scores_all = cls_prob.data
        twins = rois.data[:, :, 1:3]

        if cfg.TEST.TWIN_REG:
            # Apply bounding-twin regression deltas
            twin_deltas = twin_pred.data
            if cfg.TRAIN.TWIN_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                twin_deltas = twin_deltas.view(-1, 2) * torch.FloatTensor(cfg.TRAIN.TWIN_NORMALIZE_STDS).type_as(twin_deltas) \
                       + torch.FloatTensor(cfg.TRAIN.TWIN_NORMALIZE_MEANS).type_as(twin_deltas)
                twin_deltas = twin_deltas.view(batch_size, -1, 2 * args.num_classes)

            pred_twins_all = twin_transform_inv(twins, twin_deltas, batch_size)
            pred_twins_all = clip_twins(pred_twins_all, cfg.TRAIN.LENGTH[0], batch_size)
        else:
            # Simply repeat the twins, once for each class
            pred_twins_all = np.tile(twins, (1, scores_all.shape[1]))
            
        det_toc = time.time()
        detect_time = det_toc - det_tic
        
        for b in range(batch_size):
            misc_tic = time.time()        
            print(video_info[b])        
            scores = scores_all[b] #scores.squeeze()
            pred_twins = pred_twins_all[b] #.squeeze()

            # skip j = 0, because it's the background class          
            for j in xrange(1, args.num_classes):
                inds = torch.nonzero(scores[:,j]>thresh).view(-1)
                # if there is det
                if inds.numel() > 0:
                    cls_scores = scores[:,j][inds]
                    _, order = torch.sort(cls_scores, 0, True)
                    cls_twins = pred_twins[inds][:, j * 2:(j + 1) * 2]
                    
                    cls_dets = torch.cat((cls_twins, cls_scores.unsqueeze(1)), 1)
                    # cls_dets = torch.cat((cls_twins, cls_scores), 1)
                    cls_dets = cls_dets[order]
                    keep = nms(cls_dets, cfg.TEST.NMS)
                    if ( len(keep)>0 ):
                          cls_dets = cls_dets[keep.view(-1).long()]
                          print ("activity: ", j)
                          print (cls_dets.cpu().numpy())
                      
                    all_twins[j][i*batch_size+b] = cls_dets.cpu().numpy()
                else:
                    all_twins[j][i*batch_size+b] = empty_array

            # Limit to max_per_video detections *over all classes*
            if max_per_video > 0:
                  video_scores = np.hstack([all_twins[j][i*batch_size+b][:, -1]
                                            for j in xrange(1, args.num_classes)])
                  if len(video_scores) > max_per_video:
                      video_thresh = np.sort(video_scores)[-max_per_video]
                      for j in xrange(1, args.num_classes):
                          keep = np.where(all_twins[j][i*batch_size+b][:, -1] >= video_thresh)[0]
                          all_twins[j][i*batch_size+b] = all_twins[j][i*batch_size+b][keep, :]
                          
            misc_toc = time.time()
            nms_time = misc_toc - misc_tic                          
            print ('im_detect: {:d}/{:d} {:.3f}s {:.3f}s {:.3f}s' \
              .format(i*batch_size+b+1, args.num_videos, data_time/batch_size, detect_time/batch_size, nms_time))              

        if args.vis:
          pass
          
        data_tic = time.time()
    end = time.time()
    print("test time: %0.4fs" % (end - start))
  
  
if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    np.random.seed(cfg.RNG_SEED)
    if args.dataset == "thumos14":
        #args.imdb_name = "train_data_25fps_flipped.pkl"
        args.imdbval_name = "val_data_25fps.pkl"
        args.num_classes = 21
        args.set_cfgs = ['ANCHOR_SCALES', '[2,4,5,6,8,9,10,12,14,16]', 'NUM_CLASSES', args.num_classes]
        #args.set_cfgs = ['ANCHOR_SCALES', '[2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56]', 'NUM_CLASSES', args.num_classes]
    elif args.dataset == "activitynet":
        #args.imdb_name = "train_data_5fps_flipped.pkl"
        args.imdbval_name = "val_data_25fps.pkl"
        args.num_classes = 201
        #args.set_cfgs = ['ANCHOR_SCALES', '[1,2,3,4,5,6,7,8,10,12,14,16,20,24,28,32,40,48,56,64]', 'NUM_CLASSES', args.num_classes]  
        args.set_cfgs = ['ANCHOR_SCALES', '[1,1.25, 1.5,1.75, 2,2.5, 3,3.5, 4,4.5, 5,5.5, 6,7, 8,9,10,11,12,14,16,18,20,22,24,28,32,36,40,44,52,60,68,76,84,92,100]', 'NUM_CLASSES', args.num_classes]
                  
    args.cfg_file = "cfgs/{}_{}.yml".format(args.net, args.dataset)

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
    dataset = roibatchLoader(roidb, phase='test')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                             num_workers=args.num_workers, shuffle=False)
                             
    num_videos = len(dataset)
    args.num_videos = num_videos
    print('{:d} roidb entries'.format(num_videos))

    model_dir = args.load_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(model_dir):
        raise Exception('There is no input directory for loading network from ' + model_dir)
    output_dir = args.output_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    load_name = os.path.join(model_dir,
    'tdcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

    # initilize the network here.
    if args.net == 'c3d':
        tdcnn_demo = c3d_tdcnn(pretrained=False)
    elif args.net =='res18':
        tdcnn_demo = resnet_tdcnn(depth=18, pretrained=False)        
    elif args.net =='res34':
        tdcnn_demo = resnet_tdcnn(depth=34, pretrained=False)
    elif args.net =='res50':
        tdcnn_demo = resnet_tdcnn(depth=50, pretrained=False)
    else:
        print("network is not defined")

    tdcnn_demo.create_architecture()
    # save memory
    for key, value in tdcnn_demo.named_parameters(): value.requires_grad=False
    print(tdcnn_demo)

#    if args.cuda and torch.cuda.is_available():
#        tdcnn_demo = tdcnn_demo.cuda()
#        if isinstance(args.gpus, int):
#            args.gpus = [args.gpus]
        #assert len(args.gpus) == args.batch_size, "only support one batch_size for one gpu"
#        tdcnn_demo = nn.parallel.DataParallel(tdcnn_demo, device_ids = args.gpus)

    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    tdcnn_demo.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']
        print('load model successfully!')

    if args.cuda and torch.cuda.is_available():
        tdcnn_demo = tdcnn_demo.cuda()
        if isinstance(args.gpus, int):
            args.gpus = [args.gpus]
        #assert len(args.gpus) == args.batch_size, "only support one batch_size for one gpu"
        tdcnn_demo = nn.parallel.DataParallel(tdcnn_demo, device_ids = args.gpus)    

    test_net(tdcnn_demo, dataloader, args)

