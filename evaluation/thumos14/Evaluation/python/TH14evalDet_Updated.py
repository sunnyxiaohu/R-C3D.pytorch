#coding=utf-8

"""
    @lichenyang 2019.10.29

    The python version of TH14evalDet_Updated.m

"""    

import os
import numpy as np
import os.path as osp


from utils import textread1, textread2, textread3, intervaloverlapvalseconds


def TH14evalDet_Updated(detfilename, gtpath, subset, threshold=0.5):
    """
        - outputs:
            pr_all: list[dict]
            ap_all: list
            map: float
    """
    pr_all, ap_all, map = [], [], 0.0

    th14classids, th14classnames = textread2(osp.join(gtpath, 'detclasslist.txt'))

    gtevents = []
    th14classnamesamb = np.append(th14classnames, 'Ambiguous')
    for i in range(len(th14classnamesamb)):
        class_name = th14classnamesamb[i]
        gtfilename = osp.join(gtpath, class_name + '_' + subset + '.txt')
        if not osp.exists(gtfilename):
            print('TH14evaldet: Could not find GT file {}'.format(gtfilename))
            raise
        videonames, t1, t2 = textread3(gtfilename)
        for j in range(len(videonames)):
            gtevent = dict(
                videoname=videonames[j],
                timeinterval=np.array([t1[j], t2[j]]),
                class_name=class_name,
                conf=1,
                )
            gtevents.append(gtevent)


    if not osp.exists(detfilename):
        print('TH14evaldet: Could not find GT file {}'.format(detfilename))
        raise

    videonames, t1, t2, clsid, conf = textread1(detfilename)

    detevents = []
    for i in range(len(videonames)):
        ind = np.nonzero(clsid[i] == th14classids)[0]

        # print(ind)

        if len(ind):
            detevent = dict(
                videoname=videonames[i],
                timeinterval=np.array([t1[i], t2[i]]),
                class_name=th14classnames[ind],
                conf=conf[i],
                )
            detevents.append(detevent)
        else:
            print('WARNING: Reported class ID {} is not \
                among THUMOS14 detection classes.\n'.format(clsid[i]))


    for i in range(len(th14classnames)):
        class_name = th14classnames[i]
        classid = np.where(th14classnames == class_name)[0] + 1 # plus 1: id start from 1
        assert len(classid) == 1

        rec, prec, ap, bgcls_err, othercls_err, loc_err = \
            TH14eventdetpr(detevents, gtevents, class_name, threshold)

        pr_dict = dict(
            class_name=class_name,
            classid=classid,
            overlapthresh=threshold,
            prec=prec,
            rec=rec,
            ap=ap,
            bgcls_err=bgcls_err,
            othercls_err=othercls_err,
            loc_err=loc_err,
            )
        pr_all.append(pr_dict)
        ap_all.append(ap)

        print('AP:{} at overlap {} for {}\n'.format(ap, threshold, class_name))

    map = np.mean(np.array(ap_all))
    print('\n\nMAP: {} \n\n'.format(map))

    return pr_all, ap_all, map



def TH14eventdetpr(detevents, gtevents, class_name, overlapthresh):
    """
        - outputs:
            rec: (N, ), N is the number of detected proposal.
            prec: (N, ), N is the number of detected proposal.
            ap: float
            bgcls_err: (N, ), N is the number of detected proposal.
            othercls_err: (N, ), N is the number of detected proposal.
            loc_err: (N, ), N is the number of detected proposal.
            * Note: N <= len(inddetclass), because some proposals overlap with ambiguous gt are ignored.
    """

    gtvideonames = np.array([item['videoname'] for item in gtevents])
    detvideonames = np.array([item['videoname'] for item in detevents])
    videonames = np.unique(np.concatenate([gtvideonames, detvideonames]))

    # print(len(np.unique(gtvideonames)), len(np.unique(detvideonames)))
    # print(len(videonames))

    unsortConf = np.array([], dtype=np.int)
    unsortFlag = np.array([], dtype=np.int)
    npos = len(np.where(np.array([item['class_name'] for item in gtevents])\
                 == class_name)[0])
    assert npos > 0

    indgtclass = np.where(np.array([item['class_name'] for item in gtevents])\
                 == class_name)[0]
    indambclass = np.where(np.array([item['class_name'] for item in gtevents])\
                 == 'Ambiguous')[0]
    indothergtclass = np.setxor1d(np.arange(len(gtevents)), \
        np.concatenate([indgtclass, indambclass]))
    inddetclass = np.where(np.array([item['class_name'] for item in detevents])\
                 == class_name)[0]

    # print("inddetclass", len(inddetclass))

    if len(inddetclass) == 0:
        print('Class {} no instance, skip\n'.format(class_name))
        return 0,0,0,0,0,0

    correctPortion = np.zeros(len(videonames))
    groundNum = np.zeros(len(videonames))

    for i in range(len(videonames)):

        gt = [gtevents[x] for x in 
            np.intersect1d(np.where(videonames[i] == gtvideonames)[0], indgtclass)]
        othergt = [gtevents[x] for x in 
            np.intersect1d(np.where(videonames[i] == gtvideonames)[0], indothergtclass)]
        amb = [gtevents[x] for x in 
            np.intersect1d(np.where(videonames[i] == gtvideonames)[0], indambclass)]
        det = [detevents[x] for x in 
            np.intersect1d(np.where(videonames[i] == detvideonames)[0], inddetclass)]

        groundNum[i] = len(gt)

        if len(det):

            ids = np.argsort(-np.array([item['conf'] for item in det]))
            det = [det[x] for x in ids]  # descending order
            conf = np.array([item['conf'] for item in det])
            indfree = np.ones(len(det), dtype=np.int)
            indamb = np.zeros(len(det), dtype=np.int)

            # interesct event detection intervals with GT
            if len(gt):
                ov = intervaloverlapvalseconds(
                    np.stack([item['timeinterval'] for item in gt]), 
                    np.stack([item['timeinterval'] for item in det]))
                # print(class_name, videonames[i], ov.shape, ov[0,0])
                indfree[np.sum(ov, axis=0)>0] = 3
                for k in range(ov.shape[0]):
                    ind = np.where(indfree>0)[0]
                    # print(indfree)
                    ov_tmp = ov[k, ind]
                    if len(ov_tmp) == 0:    # len(ov_tmp)==0 means no appropriate proposal.
                        break
                    im = np.argmax(ov_tmp)
                    if ov_tmp[im] > overlapthresh:
                        indfree[ind[im]] = 0

            # interesct event detection intervals with otherGT
            if len(othergt):
                otherov = intervaloverlapvalseconds(
                    np.stack([item['timeinterval'] for item in othergt]), 
                    np.stack([item['timeinterval'] for item in det]))
                for k in range(otherov.shape[0]):
                    ind = np.where(indfree==1)[0]
                    # print(otherov.shape[0],indfree)
                    ov_tmp = otherov[k, ind]
                    if len(ov_tmp) == 0:    # len(ov_tmp)==0 means no appropriate proposal.
                        break
                    im = np.argmax(ov_tmp)
                    if ov_tmp[im] > overlapthresh:
                        indfree[ind[im]] = 4

            # respect ambiguous events (overlapping detections will be removed from the FP list)
            if len(amb):
                ovamb = intervaloverlapvalseconds(
                    np.stack([item['timeinterval'] for item in amb]), 
                    np.stack([item['timeinterval'] for item in det]))
                indamb = np.sum(ovamb, axis=0)


            idx1 = np.where(indfree==0)[0]  # matched
            idx2 = np.where(np.logical_and(indfree==1, indamb==0))[0]  # unmatched BGCls_ERR
            idx3 = np.where(np.logical_and(indfree==3, indamb==0))[0]  # unmatched Loc_ERR
            idx4 = np.where(np.logical_and(indfree==4, indamb==0))[0]  # unmatched OtherCls_ERR

            flag = np.concatenate([
                np.ones(len(idx1)),
                2*np.ones(len(idx2)),
                3*np.ones(len(idx3)),
                4*np.ones(len(idx4)),
                ]).astype(np.int)

            Idx = np.concatenate([idx1, idx2, idx3, idx4])
            ttIdx = np.argsort(Idx) # increasing order
            idxall = Idx[ttIdx]
            # print("idxall", idxall)
            flagall = flag[ttIdx]

            unsortConf = np.append(unsortConf, conf[idxall])    # Actually, conf[idxall] is equal to conf.
            unsortFlag = np.append(unsortFlag, flagall)

            if len(gt) != 0:
                correctPortion[i] = len(np.where(indfree==0)[0]) / len(gt)


    conf = np.stack([unsortConf, unsortFlag])

    # print("conf", conf.shape[1])

    ids = np.argsort(-conf[0, :])   # descending order
    tp = np.cumsum(conf[1, ids]==1)
    fp = np.cumsum(np.logical_or.reduce([conf[1, ids]==2, 
        conf[1, ids]==3, conf[1, ids]==4]))
    tmp = conf[1, ids]==1
    rec = tp / npos
    prec = tp / (fp + tp)
    ap = prap(rec, prec, tmp, npos)

    # Cls_Err and Loc_Err
    fp_bgcls = np.cumsum(conf[1, ids]==2)
    fp_loc = np.cumsum(conf[1, ids]==3)
    fp_othercls = np.cumsum(conf[1, ids]==4)

    bgcls_err = fp_bgcls / (fp+tp)
    loc_err = fp_loc / (fp+tp)
    othercls_err = fp_othercls / (fp+tp)

    return rec, prec, ap, bgcls_err, othercls_err, loc_err



def prap(rec, prec, tmp, npos):
    """
        Calculate the area under the PR curve.
    """
    ap = 0
    for i in range(len(prec)):
        if tmp[i] == 1:
            ap = ap + prec[i]
    ap = ap / npos

    return ap





