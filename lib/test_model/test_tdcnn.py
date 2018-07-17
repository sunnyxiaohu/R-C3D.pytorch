import torch
from torch.nn.modules.module import Module
from torch.autograd import Variable
from model.psroi_pooling.modules.psroi_pool import _PSRoIPooling
import numpy as np

class test_PSRoIPooling(Module):
    def __init__(self, output_dim, group_size, spatial_scale):
        super(test_PSRoIPooling, self).__init__()
        self.psroi_pool = _PSRoIPooling(output_dim, group_size, spatial_scale)

    def forward(self, feature, rois):
        return self.psroi_pool(feature, rois)


if __name__ == '__main__':
  feature = Variable(torch.FloatTensor(1, 18, 12, 10), requires_grad=True)
  rois = Variable(torch.from_numpy(np.array([[0, 20, 50, 120, 150], [0, 80, 100, 150, 200]])).float())
  psroi_pool = test_PSRoIPooling(2, 3, 1./16)
  if torch.cuda.is_available():
      feature = feature.cuda()
      rois = rois.cuda()
      psroi_pool = psroi_pool.cuda()
  print(feature)
  print(rois)
  print(psroi_pool)
  pooled_feature = psroi_pool(feature, rois)
  

