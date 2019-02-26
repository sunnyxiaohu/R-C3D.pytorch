import torch.nn as nn
import torch
from model.tdcnn.tdcnn import _TDCNN
import math

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    maxpool_count = 0
    for v in cfg:
        if v == 'M':
            maxpool_count += 1
            if maxpool_count==1:
                layers += [nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))]
            elif maxpool_count==5:
                layers += [nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2), padding=(0,1,1))]
            else:
                layers += [nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))]
        else:
            conv3d = nn.Conv3d(in_channels, v, kernel_size=(3,3,3), padding=(1,1,1))
            if batch_norm:
                layers += [conv3d, nn.BatchNorm3d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv3d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
}

class C3D(nn.Module):
    """
    The C3D network as described in [1].
        References
        ----------
       [1] Tran, Du, et al. "Learning spatiotemporal features with 3d convolutional networks."
       Proceedings of the IEEE international conference on computer vision. 2015.
    """

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def __init__(self):
        super(C3D, self).__init__()
        self.features = make_layers(cfg['A'], batch_norm=False)
        self.classifier = nn.Sequential(
            nn.Linear(512*1*4*4, 4096),
            nn.ReLU(True),
            nn.Dropout(inplace=False),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(inplace=False),
            nn.Linear(4096, 487),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class c3d_tdcnn(_TDCNN):
    def __init__(self, pretrained=False):
        self.model_path = 'data/pretrained_model/activitynet_iter_30000_3fps-caffe.pth' #ucf101-caffe.pth' #c3d_sports1M.pth' #activitynet_iter_30000_3fps-caffe.pth
        self.dout_base_model = 512
        self.pretrained = pretrained
        _TDCNN.__init__(self)

    def _init_modules(self):
        c3d = C3D()
        if self.pretrained:
            print("Loading pretrained weights from %s" %(self.model_path))
            state_dict = torch.load(self.model_path)
            c3d.load_state_dict({k:v for k,v in state_dict.items() if k in c3d.state_dict()})

        # Using conv1 -> conv5b, not using the last maxpool
        self.RCNN_base = nn.Sequential(*list(c3d.features._modules.values())[:-1])
        # Using fc6
        self.RCNN_top = nn.Sequential(*list(c3d.classifier._modules.values())[:-4])
        # Fix the layers before pool2:
        for layer in range(6):
            for p in self.RCNN_base[layer].parameters(): p.requires_grad = False

        # not using the last maxpool layer
        self.RCNN_cls_score = nn.Linear(4096, self.n_classes)
        self.RCNN_twin_pred = nn.Linear(4096, 2 * self.n_classes)      

    def _head_to_tail(self, pool5):
        pool5_flat = pool5.view(pool5.size(0), -1)
        fc6 = self.RCNN_top(pool5_flat)

        return fc6
