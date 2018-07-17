import torch
import torch.nn as nn
from model.tdcnn.resnet import resnet34, resnet50, resnet101
from .rpn import _RPN
from model.utils.config import cfg

class res(_RPN):
    def __init__(self, din, depth=34, pretrained=False, out_scores=False):
        self.dout_base_model = din
        self.pretrained = pretrained
        self.depth = depth
        self.shortcut_type = 'A' if depth in [18, 34] else 'B'
        self.model_path = '/home/agwang/Deeplearning/pytorch_dir/pretrainedmodels/resnet-{}-kinetics.pth'.format(depth)
        _RPN.__init__(self, din, out_scores)

    def _init_modules(self):
        #resnet = resnet34(sample_size=112, sample_duration=768, shortcut_type='A')
        net_str = "resnet{}(sample_size=112, sample_duration=768, shortcut_type=\'{}\')".format(self.depth, self.shortcut_type)
        resnet = eval(net_str)
        if self.pretrained:
            print("Loading pretrained weights from %s" %(self.model_path))
            state_dict = torch.load(self.model_path)['state_dict']
            # parallel unpack (model = nn.DataParallel(model, device_ids=None))
            resnet.load_state_dict({k[7:] : v for k,v in state_dict.items() if k[7:] in resnet.state_dict()})

        # Using , shape(1,256,96,7,7)
        #self.RPN_base = nn.Sequential(*list(resnet._modules.values())[:-3])
        self.RPN_base = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu,
                        resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3)

        # Fix blocks:
        # TODO: fix blocks optionally
        for p in self.RPN_base[0].parameters(): p.requires_grad=False
        for p in self.RPN_base[1].parameters(): p.requires_grad=False

        assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
        if cfg.RESNET.FIXED_BLOCKS >= 3:
            for p in self.RPN_base[6].parameters(): p.requires_grad=False
        if cfg.RESNET.FIXED_BLOCKS >= 2:
            for p in self.RPN_base[5].parameters(): p.requires_grad=False
        if cfg.RESNET.FIXED_BLOCKS >= 1:
            for p in self.RPN_base[4].parameters(): p.requires_grad=False

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad=False

        self.RPN_base.apply(set_bn_fix)

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        if mode:
            # Set fixed blocks to be in eval mode, FIXED_BLOCKS=0
            self.RPN_base.eval()
            #self.RPN_base[4].train()
            self.RPN_base[5].train()
            self.RPN_base[6].train()

        def set_bn_eval(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()

        self.RPN_base.apply(set_bn_eval)
