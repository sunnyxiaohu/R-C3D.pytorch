#coding=utf-8

"""
    @lichenyang 2019.10.29
"""    

import numpy as np

def textread1(detfilename):
    """
        Load the txt file. Return each column as a np.array.
    """
    videonames, t1, t2, clsid, conf = [], [], [], [], []

    with open(detfilename) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            # print(line)
            line_cont = line.split()
            videonames.append(line_cont[0])
            t1.append(float(line_cont[1]))
            t2.append(float(line_cont[2]))
            clsid.append(int(line_cont[3]))
            conf.append(float(line_cont[4]))

    return np.array(videonames), np.array(t1), np.array(t2), np.array(clsid), np.array(conf)


def textread2(detfilename):
    """
        Load the txt file. Return each column as a np.array.
    """
    ids, class_names = [], []

    with open(detfilename) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            # print(line)
            line_cont = line.split()
            ids.append(int(line_cont[0]))
            class_names.append(line_cont[1])

    return np.array(ids), np.array(class_names)


def textread3(detfilename):
    """
        Load the txt file. Return each column as a np.array.
    """
    videonames, t1, t2 = [], [], []

    with open(detfilename) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            # print(line)
            line_cont = line.split()
            videonames.append(line_cont[0])
            t1.append(float(line_cont[1]))
            t2.append(float(line_cont[2]))

    return np.array(videonames), np.array(t1), np.array(t2)


def nms_temporal(dets, thresh):
    """
        NMS func refer to 
            1. lib/model/nms/nms_cpu.py
            2. evaluation/thumos14/Evaluation/nms_temporal.m
        - inputs:
            dets: (N, 3), 3 means (x1, x2, score)
            thresh: float
        - outputs:
            keep: (N, )
    """

    x1 = dets[:, 0]
    x2 = dets[:, 1]
    scores = dets[:, 2]

    length = x2 - x1
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order.item(0)
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        #yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        #yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1)
        ovr = inter / (length[i] + length[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        # print("inds", inds)
        order = order[inds+1]

    return np.array(keep, dtype=np.int)


def intervaloverlapvalseconds(i1, i2, *args):
    """
        Calculate the inter overlap matrix of i1 and i2.
        - inputs:
            i1: (M, 2)
            i2: (N, 2)
            normtype: int
            gt: list[dict]
            det: list[dict]

        -outputs:
            ov: (M, N)
    """

    nargin = 2 + len(args)
    if nargin == 2:
        normtype = 0
    elif nargin == 3:
        normtype = args[0]
    elif nargin == 5:
        normtype = args[0]
        gt = args[1]
        det = args[2]
    else:
        print("Please check the arguments!")
        raise

    M = i1.shape[0]
    N = i2.shape[0]

    ov = np.zeros((M, N))

    for i in range(M):
        for j in range(N):
            # print("hahahahahaha", M, N, i1[i, :], i2[j, :])
            ov[i, j] = intervalsingleoverlapvalseconds(i1[i, :], i2[j, :], normtype)
            if nargin == 5:
                # Ensure the class is same.
                ov[i, j] *= (gt[i]['class_name'] == det[j]['class_name'])

    return ov


def intervalsingleoverlapvalseconds(i1, i2, normtype):
    """
        Calculate the inter overlap of two proposal.
        - inputs:
            i1: (2, )
            i2: (2, )
            normtype: int
                <0: w/o norm
                ==1: norm by the length of i1
                ==2: norm by the length of i2
                else: norm by the union of i1 and i2
    """

    i1 = np.sort(i1)
    i2 = np.sort(i2)

    ov = 0

    if normtype < 0:
        ua = 1
    elif normtype == 1:
        ua = i1[1] - i1[0]
    elif normtype == 2:
        ua = i2[1] - i2[0]
    else:
        bu = [np.min([i1[0], i2[0]]), np.max([i1[1], i2[1]])]
        ua = bu[1] - bu[0]

    bi = [np.max([i1[0], i2[0]]), np.min([i1[1], i2[1]])]
    iw = bi[1] - bi[0]

    iw = np.max([0.0, iw])

    ov = iw / ua


    return ov






