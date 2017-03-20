import _init_paths
import os, sys, pdb
import argparse
import numpy as np
import action_util as action
from datasets.UCF101 import UCF101
from datasets.JHMDB import JHMDB
from datasets.ucfsports import ucfsports
from fast_rcnn.config import cfg
import cPickle as pickle

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN for action detection')
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--imdb', dest='imdb', help='which imdb file to test')
    parser.add_argument('--res', dest='res', help='the linked results file')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    if not os.path.isfile(args.res):
        raise IOError(('{:s} not found.').format(args.res))
    else:
        with open(args.res, 'r') as fid:
            pred_all_dets = pickle.load(fid)

    bTemporal = False
    if 'UCF101' in args.imdb: 
        data_test = UCF101(args.imdb, 'TEST')
        bTemporal = True
    elif 'JHMDB' in args.imdb:
        data_test = JHMDB(args.imdb, 'TEST')
    elif 'UCF-Sports' in args.imdb:
        data_test = ucfsports(args.imdb, 'TEST')

    # ==== video AP evaluation ====
    gt_video = data_test.get_test_video_annotations()
    iou_thrs = [t/10.0 for t in range(1, 8)]
    iou_thrs = [0.05] + iou_thrs
    v_mAPs = {}
    for iou_thr in iou_thrs:
        ap_all = action.evaluate_linked_res_videoAP(gt_video, pred_all_dets, data_test._classes, iou_thr, bTemporal)
        # print data_test._classes[1:]
        # print ap_all
        print 'mean video AP: {}'.format(np.mean(np.array(ap_all)))
        v_mAPs[iou_thr] = np.mean(np.array(ap_all))
    print v_mAPs