#!/usr/bin/env python
import numpy as np
import _init_paths
caffe_root = '/home/lear/xpeng/code/caffe-fast-rcnn-faster-rcnn-upstream-33f2445/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import pdb
#e.g. python net_surgey.py /home/lear/xpeng/code/caffe_2015_3_12/models/modelzoo/VGG_ILSVRC/VGG_ILSVRC_19_layers.caffemodel /home/lear/xpeng/code/caffe_2015_3_12/models/modelzoo/VGG_ILSVRC/VGG_ILSVRC_19_layers_deploy.prototxt /home/lear/xpeng/code/caffe_2015_3_12/my_models/ucf101_tcnnft/VGG_19_surgey.caffemodel /home/lear/xpeng/code/caffe_2015_3_12/my_models/ucf101_tcnnft/s01/VGG_19_surgey_deploy.prototxt

#python net_surgery_rgbflow.py ../output/faster_rcnn_end2end/UCF101_RGB_1_split_0/RGB_1_VGG_16_iter_70000.model ../output/faster_rcnn_end2end/UCF101_FLOW_5_split_0/FLOW_5_VGG_16_iter_70000.caffemodel prototxt/test_ucf101_rgb1.prototxt test_ucf101_flow5.prototxt model/ucf101_rgb1flow5_sfusion.caffemodel test_ucf101_rgb1flow5_sfusion.prototxt

script, in_model, in_model2, in_delopy, in_delopy2, out_model,out_deploy = sys.argv
caffe.set_mode_cpu()

pdb.set_trace()
net = caffe.Net(in_delopy,in_model,caffe.TEST)
print("blobs {}\nparams {}".format(net.blobs.keys(), net.params.keys()))

net2 = caffe.Net(in_delopy2,in_model2,caffe.TEST)
print("blobs {}\nparams {}".format(net2.blobs.keys(), net2.params.keys()))

#modify filters
net_out = caffe.Net(out_deploy,caffe.TEST)

other_layer = [['conv1_1a','conv1_2a','conv2_1a', 'conv2_2a', 'conv3_1a', 'conv3_2a', 'conv3_3a', \
 'conv4_1a', 'conv4_2a', 'conv4_3a', 'conv5_1a', 'conv5_2a', 'conv5_3a','rpn_conv/3x3a', \
 'rpn_cls_scorea','rpn_bbox_preda','fc6a','fc7a', 'bbox_preda', 'cls_scorea'],
['conv1_1b','conv1_2b','conv2_1b', 'conv2_2b', 'conv3_1b', 'conv3_2b', 'conv3_3b', \
 'conv4_1b', 'conv4_2b', 'conv4_3b', 'conv5_1b', 'conv5_2b', 'conv5_3b','rpn_conv/3x3b',\
 'rpn_cls_scoreb','rpn_bbox_predb','fc6b','fc7b', 'bbox_predb',  'cls_scoreb']]
#  

for layer in other_layer[0]:
  net_out.params[layer][0].data[...] = net.params[layer[:-1]][0].data[...]
  net_out.params[layer][1] = net.params[layer[:-1]][1]

for layer in other_layer[1]:
  net_out.params[layer][0].data[...] = net2.params[layer[:-1]][0].data[...]
  net_out.params[layer][1] = net2.params[layer[:-1]][1]

net_out.save(out_model)



