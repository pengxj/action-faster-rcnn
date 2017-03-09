#!/usr/bin/env python

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
import caffe



def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN for actino detection')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--proto', dest='proto', help='the caffe prototxt file')
    parser.add_argument('--net', dest='net', help='the caffe net model file')
    parser.add_argument('--imdb', dest='imdb', help='which imdb file to test')
    parser.add_argument('--out', dest='savepath', help='where to save the detection results (pickle)')
    parser.add_argument('--thr', dest='thr', help='the threshold to keep for image level detection',default=0.0, type=float)
    parser.add_argument('--len', dest='LEN', help='flow lenght', default=5, type=int)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    if not os.path.isfile(args.net):
        raise IOError(('{:s} not found.').format(args.net))
    
    cfg.TEST.HAS_RPN = True
    cfg.TEST.SCALES = [600]

    bTemporal = False
    prior_length = None
    if 'RGB' in args.imdb and 'FLOW' in args.imdb:
        cfg.PIXEL_MEANSa = cfg.PIXEL_MEANS
        cfg.PIXEL_MEANSb = np.array([[[128., 128., 128.]*args.LEN]])

    if 'UCF101' in args.imdb: 
        data_test = UCF101(args.imdb, 'TEST')
        data_train = UCF101(args.imdb, 'TRAIN')
        frame_rootpath = '/home/lear/xpeng/data/UCF101/frames_240'
        flow_rootpath = '/home/lear/xpeng/data/UCF101/flows_color'
        bTemporal = True
        # information for temporal localization
        train_gt_v = data_train.get_train_video_annotations()
        keys = train_gt_v.keys()
        keys.sort()
        prior_length = {}
        global_cls = train_gt_v[keys[0]]['gt_classes']
        global_len = 0.0
        global_cnt = 0.0
        for i in range(len(keys)):             
            if not global_cls==train_gt_v[keys[i]]['gt_classes']:
                print global_cls, global_len/global_cnt
                prior_length[global_cls] = global_len/global_cnt
                global_cls = train_gt_v[keys[i]]['gt_classes']
                global_len = 0.0
                global_cnt = 0.0
            else:
                global_cnt += len(train_gt_v[keys[i]]['tubes'])
                for annot in train_gt_v[keys[i]]['tubes']:
                    global_len += annot.shape[0]
        prior_length[global_cls] = global_len/global_cnt

    elif 'JHMDB' in args.imdb:
        data_test = JHMDB(args.imdb, 'TEST')
        frame_rootpath = '/home/lear/xpeng/data/JHMDB/frames'
        flow_rootpath = '/home/lear/xpeng/data/JHMDB/flows_color'
    elif 'UCF-Sports' in args.imdb:
        data_test = ucfsports(args.imdb, 'TEST')
        frame_rootpath = '/home/lear/xpeng/data/ucf_sports_actions/UCFsports/data'
        flow_rootpath = '/home/lear/xpeng/data/ucf_sports_actions/broxflow'

    roidb = data_test.roidb
    keep_det_thr = args.thr
    if not os.path.exists(args.savepath):
        if args.cpu_mode:
            caffe.set_mode_cpu()
        else:
            caffe.set_mode_gpu()
            caffe.set_device(args.gpu_id)
        caffe_net = caffe.Net(args.proto, args.net, caffe.TEST)

        pred_all_dets = {}
        n_fr = len(data_test.image_index)
        for i in range(n_fr):
            image_name, flow_name = data_test.image_index[i].split(',')
            image_path = os.path.join(frame_rootpath, image_name) # change to your own special root path
            flow_path = os.path.join(flow_rootpath, flow_name)
            pred_all_dets[flow_name] = action.detect_action_2strem(caffe_net, image_path,\
             flow_path, keep_det_thr, args.LEN) # the flow_name used as a key since we also use the last one in UCF101 class
        with open(args.savepath,'w') as fid:
            pickle.dump(pred_all_dets, fid)
    else:
        with open(args.savepath,'r') as fid:
            print 'loading predictions...'
            import gc
            gc.disable()
            pred_all_dets = pickle.load(fid)
            gc.enable()            
    # ap_all = action.evaluate_frameAP(roidb, pred_all_dets, data_test._classes)
    # print data_test._classes[1:]
    # print ap_all
    # print 'mean frame AP: {}'.format(np.mean(np.array(ap_all)))

    # ==== video AP evaluation ====
    gt_video = data_test.get_test_video_annotations()
    iou_thrs = [t/10.0 for t in range(1, 8)]
    iou_thrs = [0.05] + iou_thrs
    v_mAPs = {}
    for iou_thr in iou_thrs:
        ap_all = action.evaluate_videoAP(gt_video, pred_all_dets, data_test._classes, iou_thr, bTemporal, prior_length)
        # print data_test._classes[1:]
        # print ap_all
        print 'mean video AP: {}'.format(np.mean(np.array(ap_all)))
        v_mAPs[iou_thr] = np.mean(np.array(ap_all))
    print v_mAPs