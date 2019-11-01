from __future__ import absolute_import

import numpy as np
import torch

def nms_cpu(dets, thresh):
    dets = dets.numpy()
    x1 = dets[:, 0]
    x2 = dets[:, 1]
    scores = dets[:, 2]

    length = (x2 - x1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order.item(0)
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        #yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        #yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1 + 1)
        ovr = inter / (length[i] + length[order[1:]] - inter)

        inds = np.where(ovr < thresh)[0]
        order = order[inds+1]

    return torch.IntTensor(keep)
