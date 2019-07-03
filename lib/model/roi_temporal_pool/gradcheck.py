import torch
from torch.autograd import gradcheck

import os.path as osp
import sys
sys.path.append(osp.abspath(osp.join(__file__, '../../')))
from roi_temporal_pooling import RoITemporalPool

feat = torch.randn(4, 16, 15, 15, 15, requires_grad=True).cuda()
rois = torch.Tensor([[0, 0, 50], [0, 10, 43],
                     [1, 67, 110]]).cuda()
inputs = (feat, rois)
print('Gradcheck for roi pooling...')
test = gradcheck(RoITemporalPool(4, 1.0 / 8), inputs, eps=1e-5, atol=1e-3)
print(test)
