import torch.nn as nn
import torch
from model.tdcnn.tdcnn import _TDCNN
from model.tdcnn.resnet import resnet18
import math
import tf_model_zoo


class eco_tdcnn(_TDCNN):
    def __init__(self, pretrained=False):
        self.model_path = 'data/pretrained_model/eco_lite_rgb_16F_kinetics_v2.pth'
        self.dout_base_model = 256
        self.pretrained = pretrained
        _TDCNN.__init__(self)

    def _init_modules(self):
        eco_bottom = getattr(tf_model_zoo, 'ECO')(model_path='lib/tf_model_zoo/ECO/ECO_bottom.yaml', num_segments=192)
        eco_top = getattr(tf_model_zoo, 'ECO')(model_path='lib/tf_model_zoo/ECO/ECO_top.yaml', num_segments=192)
        
        if self.pretrained:
            print("Loading pretrained weights from %s" %(self.model_path))
            pretrained_dict = torch.load(self.model_path)['state_dict']
            # 'module.base_model.xx'
            pretrained_dict = {'.'.join(k.split('.')[2:]) : v.cpu() for k, v in pretrained_dict.items()}
            # eco_bottom
            model_dict = eco_bottom.state_dict()
            new_state_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict) and (v.size() == model_dict[k].size())}
            eco_bottom.load_state_dict(new_state_dict)
            # eco_top
            model_dict = eco_top.state_dict()
            new_state_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict) and (v.size() == model_dict[k].size())}
            eco_top.load_state_dict(new_state_dict)
                        
        # Using conv1_7x7_s2 -> res4b_bn
        self.RCNN_base = eco_bottom
        # Using res4b_bn -> res5b_bn
        self.RCNN_top = eco_top
        
        # TODO: Fix the layers before ...
        #for layer in range(6):
        #    for p in self.RCNN_base[layer].parameters(): p.requires_grad = False

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad=False
            if classname.find('BatchNorm3d') != -1:
                for p in m.parameters(): p.requires_grad=False
                
        self.RCNN_base.apply(set_bn_fix)
        self.RCNN_top.apply(set_bn_fix) 
        
        # not using the last maxpool layer
        self.RCNN_cls_score = nn.Linear(512, self.n_classes)
        self.RCNN_twin_pred = nn.Linear(512, 2 * self.n_classes)      
        
    def prepare_data(self, video_data):
        ''' video_data will be (batch_size, C, L, H, W)
            prepared data will be (batch_size*L, C, H, W)
        '''
        prepared_data = video_data.permute(0, 2, 1, 3, 4).contiguous()
        prepared_data = prepared_data.view((-1,)+prepared_data.shape[2:])
        return prepared_data
        
    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        if mode:
            pass

        def set_bn_eval(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm3d') != -1:
                m.eval()

        self.RCNN_base.apply(set_bn_eval)
        self.RCNN_top.apply(set_bn_eval)
        
    def _head_to_tail(self, pool5):      
        fc6 = self.RCNN_top(pool5).mean(4).mean(3).mean(2)
        return fc6
