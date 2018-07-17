import torch
import torch.nn as nn
from model.tdcnn.i3d import I3D
from .rpn import _RPN


class i3d(_RPN):
    def __init__(self, din, pretrained=False, out_scores=False):
        self.model_path = 'data/pretrained_model/rgb_imagenet.pkl'
        self.dout_base_model = din
        self.pretrained = pretrained
        _RPN.__init__(self, din, out_scores)

    def _init_modules(self):
        i3d = I3D()
        if self.pretrained:
            print("Loading pretrained weights from %s" %(self.model_path))
            state_dict = torch.load(self.model_path)
            i3d.load_state_dict({k:v for k,v in state_dict.items() if k in i3d.state_dict()})

        # Using 
        self.RPN_base = nn.Sequential(*list(i3d.features._modules.values())[:-5])

        # Fix blocks:
        # TODO: fix blocks optionally
        for layer in range(1):
            for p in self.RPN_base[layer].parameters(): p.requires_grad = False

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad=False

        self.RPN_base.apply(set_bn_fix)

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        if mode:
            # Set fixed blocks to be in eval mode
            self.RPN_base.eval()
            self.RPN_base[2].train() #
            self.RPN_base[3].train() #
            self.RPN_base[5].train()
            self.RPN_base[6].train()
            self.RPN_base[8].train()
            self.RPN_base[9].train()
            self.RPN_base[10].train()
            self.RPN_base[11].train()
            self.RPN_base[12].train()

        def set_bn_eval(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()

        self.RPN_base.apply(set_bn_eval)
