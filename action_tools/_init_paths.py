# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Set up paths for Fast R-CNN."""
import os
import os.path as osp
import sys
import numpy as np

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

#Add caffe to PYTHONPATH
if 'Ubuntu' in os.uname()[-2]: 
    caffe_path = '/scratch/gpuhost9/vsydorov/libraries/caffe/caffe_latest_rcnn_ubuntu/python'
else: # Fedora
    caffe_path = '/home/lear/xpeng/code/caffe-fast-rcnn-faster-rcnn-upstream-33f2445/python'
assert osp.exists(caffe_path)
add_path(caffe_path)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, '..', 'lib')
add_path(lib_path)


def nms2d(boxes, overlap=0.3):
    # boxes = x1,y1,x2,y2,score
    if boxes.size==0:
        return np.array([],dtype=np.int32)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    scores = boxes[:,4]
    areas = (x2-x1+1)*(y2-y1+1)
    I = np.argsort(scores)
    indices = np.zeros(scores.shape, dtype=np.int32)
    counter = 0
    while I.size>0:
        i = I[-1]
        indices[counter] = i
        counter += 1
        xx1 = np.maximum(x1[i],x1[I[:-1]])
        yy1 = np.maximum(y1[i],y1[I[:-1]])        
        xx2 = np.minimum(x2[i],x2[I[:-1]])
        yy2 = np.minimum(y2[i],y2[I[:-1]]) 
        inter = np.maximum(0.0,xx2-xx1+1)*np.maximum(0.0,yy2-yy1+1)
        iou = inter / ( areas[i]+areas[I[:-1]]-inter)
        I = I[np.where(iou<=overlap)[0]]
    return indices[:counter]    

def area2d(b):
    return (b[:,2]-b[:,0]+1)*(b[:,3]-b[:,1]+1)

def overlap2d(b1, b2):
    xmin = np.maximum( b1[:,0], b2[:,0] )
    xmax = np.minimum( b1[:,2]+1, b2[:,2]+1)
    width = np.maximum(0, xmax-xmin)
    ymin = np.maximum( b1[:,1], b2[:,1] )
    ymax = np.minimum( b1[:,3]+1, b2[:,3]+1)
    height = np.maximum(0, ymax-ymin)   
    return width*height          

def iou2d(b1, b2):
    if b1.ndim == 1: b1 = b1[None,:]
    if b2.ndim == 1: b2 = b2[None,:]
    assert b2.shape[0]==1
    o = overlap2d(b1, b2)
    return o / ( area2d(b1) + area2d(b2) - o ) 
    
        
def iou3d(b1, b2):
    assert b1.shape[0]==b2.shape[0]
    assert np.all(b1[:,0]==b2[:,0])
    o = overlap2d(b1[:,1:5],b2[:,1:5])
    return np.mean( o/(area2d(b1[:,1:5])+area2d(b2[:,1:5])-o) )  
    #return np.mean(np.array([iou2d(b1[i,1:5],b2[i,1:5]) for i in range(b1.shape[0])]))
    
def iout(b1, b2):
    tmin = max(b1[0,0], b2[0,0])
    tmax = min(b1[-1,0], b2[-1,0])
    if tmax<=tmin: return 0.0    
    temporal_inter = tmax-tmin+1
    temporal_union = max(b1[-1,0], b2[-1,0]) - min(b1[0,0], b2[0,0]) + 1 
    return temporal_inter / temporal_union
        
def iou3dt(b1, b2):

    tmin = max(b1[0,0], b2[0,0])
    tmax = min(b1[-1,0], b2[-1,0])
    if tmax<=tmin: return 0.0    
    temporal_inter = tmax-tmin+1
    temporal_union = max(b1[-1,0], b2[-1,0]) - min(b1[0,0], b2[0,0]) + 1 
    return iou3d( b1[np.where(b1[:,0]==tmin)[0][0]:np.where(b1[:,0]==tmax)[0][0]+1,:] , b2[np.where(b2[:,0]==tmin)[0][0]:np.where(b2[:,0]==tmax)[0][0]+1,:]  ) * temporal_inter / temporal_union
     
def nms3dt_given_ious(scores, ious, overlap=0.5):
    if scores.size==0: return np.array([],dtype=np.int32)
    I = np.argsort(scores)
    indices = np.zeros(I.size, dtype=np.int32)
    counter = 0
    while I.size>0:
        i = I[-1]
        indices[counter] = i
        counter += 1
        I  = I[np.where(ious[I[:-1],i]<=overlap)[0]]
    return indices[:counter]

def nms3dt(detections, overlap=0.5): # list of (tube,score)
    if len(detections)==0: return np.array([],dtype=np.int32)
    I = np.argsort([d[1] for d in detections ])
    indices = np.zeros(I.size, dtype=np.int32)
    counter = 0
    while I.size>0:
        i = I[-1]
        indices[counter] = i
        counter += 1
        ious = np.array([ iou3dt(detections[ii][0],detections[i][0]) for ii in I[:-1] ])
        I  = I[np.where(ious<=overlap)[0]]
    return indices[:counter]

def pr_to_ap(pr):
    prdif = pr[1:,1]-pr[:-1,1]
    prsum = pr[1:,0]+pr[:-1,0]
    return np.sum(prdif*prsum*0.5)
