
import _init_paths
from _init_paths import *
from fast_rcnn.nms_wrapper import nms
import numpy as np 
import os, pdb
import cv2
from fast_rcnn.test import im_detect, im_detect_2stream


def roidb_to_array(roidb):
    '''
    transform the roidb to array of [label, img_index, box]
    '''
    n_fr = len(roidb)
    res = []
    for i in range(n_fr):
        n_box = roidb[i]['boxes'].shape[0]
        for n in range( n_box ):
            box_label = roidb[i]['gt_classes'][n]
            box = roidb[i]['boxes'][n,:]
            ires = np.hstack((box_label, i+1, box)) 
            res.append(ires)    
    return np.array(res)

def pred_to_array(pred):
    '''
    transform the pred results to array of [label, img_index, box, all_cls_score]
    '''
    n_fr = len(pred)
    keys = pred.keys()
    keys.sort()
    res = []
    for i in range(n_fr): # image
        image = keys[i]
        for cls_ind in pred[image]:
            if np.array(pred[image][cls_ind]).size>0:
                n_box = pred[image][cls_ind].shape[0]
                for n in range(n_box):
                    box_label = int(cls_ind)
                    box = pred[image][cls_ind][n, :4]
                    score = pred[image][cls_ind][n, 4] 
                    ires = np.hstack((box_label, i, box, score))
                    res.append(ires)
    return np.array(res)

def detect_action_img(net, im_file, CONF_THRESH = 0, LEN = 1, NMS_THRESH = 0.3):
    """Detect object classes in an image using pre-computed object proposals. 
    return finaldets: dict of class index
    """ 
    def get_image_format(imfile):
        name, ext = imfile.split('.')
        return '{:0%d}.%s' % (len(name), ext)
    if LEN==1:
        im = cv2.imread(im_file)
    else:
        offset = LEN//2
        impath = os.path.dirname(im_file)
        im_name = os.path.basename(im_file)
        img_format = get_image_format(im_name)
        for j in xrange(LEN):
            imnum = int(im_name.split('.')[0]) + j - offset
            assert imnum>0        
            imfile = img_format.format( imnum )
            im_1 = cv2.imread(os.path.join(impath, imfile)) 
            assert im_1 != None
            if j==0:
                im = np.zeros((im_1.shape[0], im_1.shape[1], im_1.shape[2]*LEN), dtype=np.float32)
            im[:,:,j*3:(j+1)*3] = im_1
    scores, boxes = im_detect(net, im)
    print ('{}, Detection for {:d} object proposals').format(im_file,boxes.shape[0])
    finaldets = {}
    num_cls = scores.shape[1] # including bkg
    # dict_dets = {} # just for vis and save
    for cls_ind in range(num_cls-1):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH) ##
        dets = dets[keep, :]
        keep2 = list(np.where(dets[:, -1] >= CONF_THRESH)[0])
        finaldets[cls_ind] = dets[keep2, :]
    return finaldets

def detect_action_2strem(net, im_file, flow_file, CONF_THRESH = 0, LEN = 1, NMS_THRESH = 0.3):
    """Detect object classes in an image using pre-computed object proposals. 
    return finaldets: dict of class index
    """ 
    def get_image_format(imfile):
        name, ext = imfile.split('.')
        return '{:0%d}.%s' % (len(name), ext)

    im = cv2.imread(im_file)
    assert im != None
    if LEN==1:
        flows = cv2.imread(flow_file)
    else:
        offset = LEN//2
        impath = os.path.dirname(flow_file)
        im_name = os.path.basename(flow_file)
        img_format = get_image_format(im_name)
        for j in xrange(LEN):
            imnum = int(im_name.split('.')[0]) + j - offset
            assert imnum>0        
            imfile = img_format.format( imnum )
            im_1 = cv2.imread(os.path.join(impath, imfile)) 
            assert im_1 != None
            if j==0:
                flows = np.zeros((im_1.shape[0], im_1.shape[1], im_1.shape[2]*LEN), dtype=np.float32)
            flows[:,:,j*3:(j+1)*3] = im_1
    scores, boxes = im_detect_2stream(net, im, flows)

    print ('{},{}, Detection for {:d} object proposals').format(im_file,flow_file,boxes.shape[0])
    finaldets = {}
    num_cls = scores.shape[1] # including bkg
    # dict_dets = {} # just for vis and save
    for cls_ind in range(num_cls-1):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH) ##
        dets = dets[keep, :]
        keep2 = list(np.where(dets[:, -1] >= CONF_THRESH)[0])
        finaldets[cls_ind] = dets[keep2, :]
    return finaldets

def evaluate_frameAP(roidb, all_boxes, CLASSES, iou_thresh = 0.5):
    gt_allboxes = roidb_to_array(roidb)
    pred_allboxes = pred_to_array(all_boxes)
    ap_all = []
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1
        gt = gt_allboxes[np.where(gt_allboxes[:,0]==cls_ind)[0],1:]
        pred_cls = pred_allboxes[np.where(pred_allboxes[:,0]==cls_ind)[0],1:]
        
        ap = frame_ap_one(gt, pred_cls, iou_thresh)
        ap_all.append(ap)

    return ap_all

def frame_ap_one(gt, pred, iou_thresh = 0.5):
    '''
    both gt and pred are pre-processed to ensure that they belong to the same category
    gt: array[img_index, box]
    pred: array[img_index, box, cls_score]
    '''
    img_index = pred[:,0]
    pr = np.empty((pred.shape[0]+1,2), dtype=np.float32) # precision,recall
    pr[0,0] = 1.0
    pr[0,1] = 0.0
    fn = gt.shape[0]
    fp = 0
    tp = 0
    sorted_ind = np.argsort(-pred[:, -1])
    for i, k in enumerate(sorted_ind):
        box = pred[k,:]
        ispositive = False
        index_this = np.where(gt[:,0]==box[0])[0]
        if index_this.size>0:
            BBGT = gt[index_this, 1:]
            iou = iou2d(BBGT, box[1:])
            argmax = np.argmax(iou) # get the max overlap window
            if iou[argmax]>=iou_thresh:
                ispositive = True
                gt = np.delete(gt, index_this[argmax], 0)
        if ispositive:
            tp += 1
            fn -= 1
        else:
            fp += 1
        pr[i+1,0] = float(tp)/float(tp+fp)
        pr[i+1,1] = float(tp)/float(tp+fn)
        ap = voc_ap(pr)
    return ap

def detect_action_video(caffe_net, v_path, LEN = 1, CONF_THRESH = 0.2, NMS_THRESH = 0.3):
    '''
    This function is a demo for video action detection, only support single frame for now
    v_path: path/to/video.avi, it can be rgb or optical flow videos
    '''
    assert LEN==1
    cap = cv2.VideoCapture(v_path)

    while(cap.isOpened()):
        ret, frame = cap.read()
        scores, boxes = im_detect(caffe_net, frame)
        finaldets = {}
        num_cls = scores.shape[1] # including bkg
        for cls_ind in range(num_cls-1):
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH) ##
            dets = dets[keep, :]
            # pdb.set_trace()
            finaldets[cls_ind] = dets[dets[:, -1] >= CONF_THRESH, :]

            # display
            disArr = np.copy(finaldets[cls_ind])
            disArr[:,2] -= disArr[:,0]
            disArr[:,3] -= disArr[:,1]
            for i in range(disArr.shape[0]):
                cv2.rectangle(frame, (disArr[i,0], disArr[i, 1]), (disArr[i,2], disArr[i, 3]), 3, (0,255,0))
                cv2.putText(frame, 'CLS:'+str(cls_ind), (disArr[i,0], disArr[i, 1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (0,0,255),2)
        
        cv2.imshow('video', frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def evaluate_linked_res_videoAP(gt_videos, all_dets, CLASSES, iou_thresh = 0.2, bTemporal = False):
    '''Evaluate saved already-linked video detections

    Args:
        gt_videos: {vname:{tubes: [[frame_index, x1,y1,x2,y2]]}, {gt_classes: vlabel}} 
        all_dets:  {vname: [class_index, cls_score , array([frame_index, x1,y1,x2,y2, cls_score_for_frame])]}
        CLASSES:

    Returns:
        average precision of all classes
    '''
    def format_vdets(vdets):
        '''format video etections as [label, video_index, tube_score, array[frame_index, x1,y1,x2,y2, cls_score_for_frame] ]
        '''
        keys = vdets.keys()
        keys.sort() 
        res = []
        for i in range(len(keys)):
            v_det = vdets[keys[i]]
            for det in v_det:
                res.append([det[0]+1,i+1,det[1], det[2]])
        return res

    gt_videos_format = gt_to_videts(gt_videos)
    pred_videos_format = format_vdets(all_dets)
    ap_all = []    
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1
        print cls_ind
        gt = [g[1:] for g in gt_videos_format if g[0]==cls_ind]
        pred = [p[1:] for p in pred_videos_format if p[0]==cls_ind]
        
        argsort_scores = np.argsort(-np.array([score for _,score, _ in pred])) 
        pr = np.empty((len(pred)+1,2), dtype=np.float32)# precision,recall
        pr[0,0] = 1.0
        pr[0,1] = 0.0
        fn = len(gt) #sum([len(a[1]) for a in gt])
        fp = 0
        tp = 0

        gt_v_index = [g[0] for g in gt]
        for i,k in enumerate(argsort_scores):
            if i%100==0: print "%6.2f%% boxes processed, %d positives found, %d remain"%(100*float(i)/argsort_scores.size, tp, fn)
            video_index, tube_score , boxes = pred[k]
            ispositive = False
            if video_index in gt_v_index:
                gt_this_index, gt_this = [], []
                for j, g in enumerate(gt):
                    if g[0]==video_index:
                        gt_this.append(g[1])
                        gt_this_index.append(j)
                if len(gt_this)>0:
                    if bTemporal:
                        iou = np.array([iou3dt(g,boxes[:,:5]) for g in gt_this])
                    else:            
                        if boxes.shape[0]>gt_this[0].shape[0]:
                            # in case some frame don't have gt 
                            iou = np.array([iou3d(g, boxes[int(g[0,0]-1):int(g[-1,0]),:5]) for g in gt_this]) 
                        elif boxes.shape[0]<gt_this[0].shape[0]:
                            # in flow case 
                            iou = np.array([iou3d(g[int(boxes[0,0]-1):int(boxes[-1,0]),:], boxes[:,:5]) for g in gt_this]) 
                        else:
                            iou = np.array([iou3d(g, boxes[:,:5]) for g in gt_this]) 

                    if iou.size>0: # on ucf101 if invalid annotation ....
                        argmax = np.argmax(iou)
                        if iou[argmax]>=iou_thresh:
                            ispositive = True
                            del gt[gt_this_index[argmax]]
            if ispositive:
                tp += 1
                fn -= 1
            else:
                fp += 1
            pr[i+1,0] = float(tp)/float(tp+fp)
            pr[i+1,1] = float(tp)/float(tp+fn)
        ap = voc_ap(pr)
        ap_all.append(ap)

    return ap_all    

def gt_to_videts(gt_v):
    # return  [label, video_index, [[frame_index, x1,y1,x2,y2]] ]
    keys = gt_v.keys()
    keys.sort()
    res = []
    for i in range(len(keys)):
        v_annot = gt_v[keys[i]]
        for j in range(len(v_annot['tubes'])):
            res.append([v_annot['gt_classes'], i+1, v_annot['tubes'][j]])
    return res

def evaluate_videoAP(gt_videos, all_boxes, CLASSES, iou_thresh = 0.2, bTemporal = False, prior_length = None):
    '''
    gt_videos: {vname:{tubes: [[frame_index, x1,y1,x2,y2]]}, {gt_classes: vlabel}} 
    all_boxes: {imgname:{cls_ind:array[x1,y1,x2,y2, cls_score]}}
    '''
    def imagebox_to_videts(img_boxes, CLASSES, savefile='temp_pred.pkl'):
        # return [label, video_index, [frame_index, [[x1,y1,x2,y2, score]] ] ]
        keys = all_boxes.keys()
        keys.sort()        
        res = []
        for cls_ind, cls in enumerate(CLASSES[1:]):
            v_cnt = 1
            frame_index = 1
            v_dets = []
            cls_ind += 1
            preVideo = os.path.dirname(keys[0])
            for i in range(len(keys)):
                curVideo = os.path.dirname(keys[i])
                img_cls_dets = img_boxes[keys[i]][cls_ind]
                v_dets.append([frame_index, img_cls_dets])
                frame_index += 1
                if preVideo!=curVideo:
                    preVideo = curVideo
                    frame_index = 1
                    tmp_dets = v_dets[-1]
                    del v_dets[-1]
                    res.append([cls_ind, v_cnt, v_dets])
                    v_cnt += 1
                    v_dets = []
                    v_dets.append(tmp_dets)
            # the last video
            print 'num of videos:{}'.format(v_cnt)
            res.append([cls_ind, v_cnt, v_dets])
        return res

    gt_videos_format = gt_to_videts(gt_videos)
    pred_videos_format = imagebox_to_videts(all_boxes, CLASSES)
    ap_all = []    
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1
        print cls_ind
        gt = [g[1:] for g in gt_videos_format if g[0]==cls_ind]
        pred_cls = [p[1:] for p in pred_videos_format if p[0]==cls_ind]
        if bTemporal: 
            cls_len = prior_length[cls_ind]
        else:
            cls_len = None
        ap = video_ap_one(gt, pred_cls, iou_thresh, bTemporal, cls_len)
        ap_all.append(ap)

    return ap_all

    #
def video_ap_one(gt, pred_videos, iou_thresh = 0.2, bTemporal = False, gtlen = None):
    '''
    gt: [ video_index, array[frame_index, x1,y1,x2,y2] ]

    pred_videos: [ video_index, [ [frame_index, [[x1,y1,x2,y2, score]] ] ] ]
    '''
    # link for prediction
    pred = []
    for pred_v in pred_videos:
        video_index = pred_v[0]
        pred_link_v = link_video_one(pred_v[1], True, gtlen) # [array<frame_index, x1,y1,x2,y2, cls_score>]
        for tube in pred_link_v:
            pred.append((video_index, tube))

    argsort_scores = np.argsort(-np.array([np.mean(b[:,5]) for _,b in pred])) 
    pr = np.empty((len(pred)+1,2), dtype=np.float32)# precision,recall
    pr[0,0] = 1.0
    pr[0,1] = 0.0
    fn = len(gt) #sum([len(a[1]) for a in gt])
    fp = 0
    tp = 0

    gt_v_index = [g[0] for g in gt]

    for i,k in enumerate(argsort_scores):
        if i%100==0: print "%6.2f%% boxes processed, %d positives found, %d remain"%(100*float(i)/argsort_scores.size, tp, fn)
        video_index, boxes = pred[k]
        ispositive = False
        if video_index in gt_v_index:
            gt_this_index, gt_this = [], []
            for j, g in enumerate(gt):
                if g[0]==video_index:
                    gt_this.append(g[1])
                    gt_this_index.append(j)
            if len(gt_this)>0:
                if bTemporal:
                    iou = np.array([iou3dt(g,boxes[:,:5]) for g in gt_this])
                else:            
                    if boxes.shape[0]>gt_this[0].shape[0]:
                        # in case some frame don't have gt 
                        iou = np.array([iou3d(g, boxes[int(g[0,0]-1):int(g[-1,0]),:5]) for g in gt_this]) 
                    elif boxes.shape[0]<gt_this[0].shape[0]:
                        # in flow case 
                        iou = np.array([iou3d(g[int(boxes[0,0]-1):int(boxes[-1,0]),:], boxes[:,:5]) for g in gt_this]) 
                    else:
                        iou = np.array([iou3d(g, boxes[:,:5]) for g in gt_this]) 

                if iou.size>0: # on ucf101 if invalid annotation ....
                    argmax = np.argmax(iou)
                    if iou[argmax]>=iou_thresh:
                        ispositive = True
                        del gt[gt_this_index[argmax]]
        if ispositive:
            tp += 1
            fn -= 1
        else:
            fp += 1
        pr[i+1,0] = float(tp)/float(tp+fp)
        pr[i+1,1] = float(tp)/float(tp+fn)
    ap = voc_ap(pr)

    return ap


def link_video_one(vid_det, bNMS3d = False, gtlen=None):
    '''
    linking for one class in a video (in full length)
    vid_det: a list of [frame_index, [bbox cls_score]]
    gtlen: the mean length of gt in training set
    return a list of tube [array[frame_index, x1,y1,x2,y2, cls_score>]]
    '''
    vdets = [vid_det[i][1] for i in range(len(vid_det))]
    # pdb.set_trace()
    vres = link_detections(vdets) 
    if len(vres)!=0:
        if bNMS3d:
            tube = [b[:,:5] for b in vres]
            tube_scores = [np.mean(b[:,5]) for b in vres]
            dets = [(tube[t], tube_scores[t]) for t in range(len(tube))]
            # nms for tubes
            # pdb.set_trace()
            keep = nms3dt(dets,0.3) # bug for nms3dt
            if np.array(keep).size:
                vres_keep = [vres[k] for k in keep]
                # max subarray with penalization -|Lc-L|/Lc
                if gtlen:
                    vres = temporal_check(vres_keep, gtlen)
                else:
                    vres = vres_keep

    return vres


def _compute_edge_one_cls(d1, d2, w_iou=1.0, w_scores=1.0):
    # d: <x1> <y1> <x2> <y2> <class score>
    N1 = d1.shape[0]
    N2 = d2.shape[0]
    scores = np.zeros((N1,N2),dtype=np.float32) ########
    for i in range(N1):
        if w_iou>0:
            b1 = d1[i,:4]
            bbiou = iou2d(d2[:,:4],b1)
            scores[i,:] += w_iou* bbiou
        if w_scores>0:            
            sum_score = d1[i,4]+d2[:,4]
            # bbiou[bbiou>=0.2] = 1.0
            # bbiou[bbiou<0.2] = 0
            # sum_score = sum_score * bbiou # supress the class score if IOU is 0
            scores[i,:] += w_scores* sum_score 
    return scores

def link_detections(detections, w_iou=1.0, w_scores=1.0): 
    # detections: list of bounding boxes <x1> <y1> <x2> <y2> <class score>
    # check no empty detections
    ind_notempty = []
    nfr = len(detections)
    for i in range(nfr):
        if np.array(detections[i]).size:
            ind_notempty.append(i)
    # no detections at all
    if not ind_notempty:
        return []
    # miss some frames
    elif len(ind_notempty)!=nfr: 
       #  print 'some detections are empty, copy from nearest detections...'       
        for i in range(nfr):
            if not np.array(detections[i]).size:
                # copy the nearest detections
                ind_dis = np.abs(np.array(ind_notempty) - i)
                nn = np.argmin(ind_dis)
                detections[i] = detections[ind_notempty[nn]]
    
    detect = detections
    nframes = len(detect)
    res = []

    isempty_vertex = np.zeros((nframes,),dtype=np.bool)
    edge_scores = [_compute_edge_one_cls(detect[i],detect[i+1], w_iou=w_iou,w_scores=w_scores) for i in range(nframes-1)]
    copy_edge_scores = edge_scores
    while not np.any(isempty_vertex):
        # initialize
        scores = [np.zeros((d.shape[0],),dtype=np.float32) for d in detect]
        index = [np.nan*np.ones((d.shape[0],),dtype=np.float32) for d in detect]
        # viterbi
        for i in range(nframes-2,-1,-1):
            edge_score = edge_scores[i]+scores[i+1]
            scores[i] = np.max(edge_score,axis=1)
            index[i] = np.argmax(edge_score,axis=1)
        # decode
        idx = -np.ones((nframes,),dtype=np.int32)
        idx[0] = np.argmax(scores[0])
        for i in range(0, nframes-1):
            idx[i+1] = index[i][idx[i]]
        # remove covered boxes and build output structures
        this = np.empty((nframes,6),dtype=np.float32)
        this[:,0] = 1+np.arange(nframes)
        for i in range(nframes):
            j = idx[i]
            iouscore = 0
            if i<nframes-1:
                iouscore = copy_edge_scores[i][j,idx[i+1]] - detections[i][j, 4] - detections[i+1][idx[i+1], 4]

            if i<nframes-1: edge_scores[i] = np.delete(edge_scores[i], j, 0)
            if i>0: edge_scores[i-1] = np.delete(edge_scores[i-1], j, 1)
            this[i,1:5] = detect[i][j,:4]
            this[i, 5] = detect[i][j, 4] #+ 1*iouscore##### where is the IOU score0.5*
            detect[i] = np.delete(detect[i],j,0)
            isempty_vertex[i] = detect[i].size==0 # it is true when there is no detection in any frame
        res.append( this ) 
        if len(res)==3:
            break
        
    return res

def get_max_subset(x_org, gtL):
    x = x_org - np.mean(x_org)
    bestSoFar = 0
    bestNow = 0
    bestStartIndexSoFar = -1
    bestStopIndexSoFar = -1
    bestStartIndexNow = -1
    for i in xrange(x.shape[0]):
        value = bestNow + x[i]
        if value > 0:
            if bestNow == 0:
                bestStartIndexNow = i
            bestNow = value
        else:
            bestNow = 0
        if bestNow > bestSoFar:
            bestSoFar = bestNow
            bestStopIndexSoFar = i
            bestStartIndexSoFar = bestStartIndexNow
#    # search suitable length surrounding: approximate method
#    L_d = bestStopIndexSoFar-bestStartIndexSoFar
#    lcost = - (|gt_L - L_d| / gt_L)
    if gtL>(bestStopIndexSoFar-bestStartIndexSoFar):
        ext = (gtL - (bestStopIndexSoFar-bestStartIndexSoFar))//2
        bestStartIndexSoFar -= ext
        bestStopIndexSoFar += ext      
    elif gtL<(bestStopIndexSoFar-bestStartIndexSoFar):
        ext = ((bestStopIndexSoFar-bestStartIndexSoFar) - gtL)//2
        bestStartIndexSoFar += ext
        bestStopIndexSoFar -= ext

    if bestStartIndexSoFar<0: bestStartIndexSoFar=0
    if bestStopIndexSoFar>x.shape[0]: bestStopIndexSoFar=x.shape[0]
    return bestSoFar, bestStartIndexSoFar, bestStopIndexSoFar

def temporal_check(tubes, gt_L):
    # nframes x 6 array <frame> <x1> <y1> <x2> <y2> <score>
    # objective: max ( mean(score[L_d]) - (|gt_L - L_d| / gt_L) )
    save_tubes = []
    for tube in tubes:  #bbiou = iou2d(d2[:,1:5],b1)
        nframes = tube.shape[0]
        edge_scores = np.array([iou2d(tube[i,1:5],tube[i+1,1:5]) for i in range(nframes-1)]) # +tube[i,5]
        # if both overlap and cls score are low, then reverse the score, they should be remove from the tube
        ind = np.where(edge_scores<0.3)[0] + 1  
        score = tube[:, 5]
        score[ind] = -score[ind]
        best_v, beststart, bestend = get_max_subset(score, gt_L)
        trimed_b = tube[int(beststart):int(bestend), :]
        save_tubes.append(trimed_b)
    return save_tubes
