#coding=utf-8

"""
    @lichenyang 2019.10.29

    The python version of eval_thumos14.m

"""    

import os
import numpy as np
import os.path as osp
import scipy.io as scio

from utils import textread1, nms_temporal
from TH14evalDet_Updated import TH14evalDet_Updated

def eval_thumos14():
    """
        Calculate the mAP for all threshold in test_thresholds.
    """

    AP_all=[]
    PR_all=[]
    mAP=[]
    REC_all=[]

    detfilename = '../../tmp.txt.bak'
    gtpath = '../annotation/annotation_test/'
    subset = 'test'

    new_length = 16
    test_thresholds =   [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    nms_thresholds =    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    class_num = 20
    video_num = 213
    proposals_per_video = 200

    for index in range(len(test_thresholds)):
        threshold = test_thresholds[index]
        print('\n threshold={} \n'.format(threshold))
        videonames, t1, t2, clsid, conf = textread1(detfilename)
        # print(videonames)

        # step1 is ignored.

        # step2: Fillter confidence lower than the threshold
        confid = conf > 0.005

        videonames = videonames[confid]

        t1 = t1[confid]
        t2 = t2[confid]
        clsid = clsid[confid]
        conf = conf[confid]

        # step3: NMS after detection -per video
        overlap_nms = nms_thresholds[index]
        videoid = np.unique(videonames)
        # print(videoid)
        # print(len(videoid))

        pick_nms = np.array([], dtype=np.int)
        for idx in range(len(videoid)):
            vid = videoid[idx]
            for cls in range(1, class_num + 1):
                inputpick = np.nonzero((videonames == vid)&(clsid == cls))[0]
                # print(cls, vid, inputpick)
                # print(inputpick)
                res_nms = nms_temporal(np.hstack(\
                    [
                    t1[inputpick, np.newaxis], 
                    t2[inputpick, np.newaxis],
                    conf[inputpick, np.newaxis]
                    ]), overlap_nms)
                # print(res_nms)
                pick_nms = np.append(pick_nms, inputpick[res_nms])
                # print(pick_nms)

        # print(len(pick_nms), proposals_per_video*video_num)
        pick_nms = pick_nms[:min(len(pick_nms), proposals_per_video*video_num)]
        videonames = videonames[pick_nms]
        t1 = t1[pick_nms]
        t2 = t2[pick_nms]
        clsid = clsid[pick_nms]
        conf = conf[pick_nms]
        # print(t1[-20:])
        # print(t1.shape)

        with open('tmp_run.txt', 'w') as fout:
            for i in range(len(videonames)):
                line = ' '.join([videonames[i], str(t1[i]), str(t2[i]), str(clsid[i]), str(conf[i])]) + '\n'
                fout.write(line)

        pr_all, ap_all, map = TH14evalDet_Updated('tmp_run.txt', gtpath, subset, threshold)

        mAP.append(map)
        AP_all.append(ap_all)
        PR_all.append(pr_all)
        ave_rec = 0
        for ii in range(class_num):
            ave_rec += pr_all[ii]['rec'][-1]
        ave_rec /= class_num
        REC_all.append(ave_rec)

        # print("REC_all", REC_all)


    mat_dict = dict(
        PR_all=PR_all,
        AP_all=AP_all,
        REC_all=REC_all,
        mAP=mAP
        )
    scio.savemat('res_thumos14.mat', mat_dict)



if __name__ == '__main__':
    eval_thumos14()





