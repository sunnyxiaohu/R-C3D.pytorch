import torch
import torch.nn as nn
from model.tdcnn.c3d import C3D
from .rpn import _RPN


class c3d(_RPN):
    def __init__(self, din, pretrained=False, out_scores=False):
        self.model_path = 'data/pretrained_model/activitynet_iter_30000_3fps-caffe.pth'
        self.dout_base_model = din
        self.pretrained = pretrained
        _RPN.__init__(self, din, out_scores)

    def _init_modules(self):
        c3d = C3D()
        if self.pretrained:
            print("Loading pretrained weights from %s" %(self.model_path))
            state_dict = torch.load(self.model_path)
            c3d.load_state_dict({k:v for k,v in state_dict.items() if k in c3d.state_dict()})

        # using conv1a -> conv5b, not using the last maxpool layer
        self.RPN_base = nn.Sequential(*list(c3d.features._modules.values())[:-1])

        # Fix the layers before pool2:
        for layer in range(6):
            for p in self.RPN_base[layer].parameters(): p.requires_grad = False

