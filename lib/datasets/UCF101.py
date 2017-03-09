
import datasets
import os, copy
from datasets.imdb import imdb
import numpy as np
import scipy.sparse

import cPickle
import subprocess
import pdb


class UCF101(imdb):
    def __init__(self, image_set, PHASE, mat_annot_file = None):
        imdb.__init__(self, image_set, PHASE)
        if PHASE=='TRAIN':
            self._image_set = './action_experiments/listfiles/' + image_set + '.trainlist' # you need a full path for image list and data path
        else:
            self._image_set = './action_experiments/listfiles/' + image_set + '.testlist'
        self._USE_MAT_GT = mat_annot_file!=None        
        self._annot_path = "/home/lear/xpeng/data/UCF101/UCF101_24_Annotations" # you only have annotations in RGB data folder
        if 'RGB' in image_set and 'FLOW' in image_set:
            self._data_path = '/home/lear/xpeng/data/UCF101/flows_color'
        else:
            self._MOD = image_set.split('_')[1]
            self._LEN = image_set.split('_')[2]
            self._data_path = None
            if self._MOD=='RGB': self._data_path = '/home/lear/xpeng/data/UCF101/frames_240'
            if self._MOD=='FLOW': self._data_path = '/home/lear/xpeng/data/UCF101/flows_color'

        self._classes = ('__background__', 
                         'Basketball', 'BasketballDunk', 'Biking', 'CliffDiving', 'CricketBowling', 
                         'Diving', 'Fencing', 'FloorGymnastics', 'GolfSwing', 'HorseRiding',
                         'IceDancing', 'LongJump', 'PoleVault', 'RopeClimbing', 'SalsaSpin',
                         'SkateBoarding', 'Skiing', 'Skijet', 'SoccerJuggling', 'Surfing',
                         'TennisSwing', 'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._img_format = '' # it will update after load image set index
        self._image_index = self._load_image_set_index() # get the temporal anotation

        self.test_videos = sorted([l.split()[0][:-4] for l in file('./action_experiments/listfiles/UCF101_video_testlist01.txt')])
        self.train_videos = sorted([l.split()[0][:-4] for l in file('./action_experiments/listfiles/UCF101_video_trainlist01.txt')])
        self.videos = sorted([l.split()[0][:-4] for l in file("./action_experiments/listfiles/UCF101_video_trainlist01.txt")]+ self.test_videos )        
        self.video_to_label = {v: self._class_to_ind[v.split('/')[0]] for v in self.videos }

        if mat_annot_file:
            self._mat_gt = self.get_mat_gt(mat_annot_file)
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb

    def image_path_at(self, i):
        return os.path.join(self._data_path, self._image_index[i])

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        if not os.path.exists(self._image_set):
            print 'Path does not exist: {}'.format(self._image_set)
            print 'Preparing {}'.format(self._image_set)
            self.prepare_traintest()

        with open(self._image_set) as f:
            image_index = [x.strip() for x in f.readlines()] 

        aimage = os.path.basename(image_index[0])
        name, ext = aimage.split('.')
        self._img_format = '{:0%d}.%s' % (len(name), ext)

        return image_index

    # roi database preparation
    def gt_roidb(self):
        cache_file = self.cache_path
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb
        
        roidb = [self._load_UCF101_annotation(index) for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return roidb

    #####################################################################
    def get_mat_gt(self, mat_annot_file):
        # parse annot from mat file
        import scipy.io as sio
        f = sio.loadmat(mat_annot_file)['annot'][0]

        mat_gt = {}
        n_total = f.shape[0]
        for i in range(n_total):
            videoname = str(f[i][1][0])
            mat_gt[videoname] = {}
            ef = f[i][2][0][0][0][0,0]
            sf = f[i][2][0][0][1][0,0]
            for framenr in range(sf, ef+1):
                mat_gt[videoname][framenr] = f[i][2][0][0][3][framenr-sf,:].astype(np.int32)

        return mat_gt


    def _load_UCF101_annotation(self, index):
        index = index.split(',')[-1] # to support 2 stream filelist input
        videoname = os.path.dirname(index) 
        frm = int(index.split('/')[-1].split('.')[0])
        if self._USE_MAT_GT:
            if videoname in self._mat_gt and frm in self._mat_gt[videoname]:
                boxes = self._mat_gt[videoname][frm]
                if boxes.ndim==1:
                    boxes = boxes[np.newaxis, :]
                    boxes[:,2] += (boxes[:,0]-1)
                    boxes[:,3] += (boxes[:,1]-1)
                    if not (boxes[:, 2] >= boxes[:, 0]).all():
                        print index
                        print boxes
                else:
                    pdb.set_trace()
            else:
                # print '{} {} has no box'.format(videoname, frm)
                boxes = np.empty((0,4), dtype=np.int32) 
        else:
            boxes = self.get_annot_image_boxes(videoname, frm).astype(np.uint16)
        num_objs = boxes.shape[0]
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        label = self._class_to_ind[videoname.split('/')[0]]
        for ix in range(num_objs):
            gt_classes[ix] = label
            overlaps[ix, label] = 1.0

        overlaps = scipy.sparse.csr_matrix(overlaps)
        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False}

    # annotation: warning few images do not have annotation
    def get_annot_image_boxes(self, videoname, n):
        video_annot = self.get_annot_video(videoname)
        if len(video_annot)==0: return np.empty( (0,4), dtype=np.float32)
        video_annot = np.concatenate(video_annot, axis=0) # stack all tubes
        return video_annot[np.where(video_annot[:,0]==n)[0],1:5]
        
    def get_image_format(self, videoname):
        
        return os.path.join(self._data_path, videoname, self._img_format) # 

    def get_image_file(self, videoname, n):
        return self.get_image_format(videoname) % (n)

    def get_image(self, videoname, n):
        return np.array(Image.open(self.get_image_file(videoname, n)))

    def get_resolution(self, videoname):
        return self.get_image(videoname, 1).shape[:2]

    def get_nframes(self, videoname):
        return len([i for i in os.listdir(os.path.dirname(self.get_image_format(videoname))) if i.endswith("jpg")])

    def get_annot_video(self, videoname, label=None):
        def area2d(b):
            return (b[:,2]-b[:,0]+1)*(b[:,3]-b[:,1]+1)
            
        if videoname in ["TennisSwing/v_TennisSwing_g09_c01","TennisSwing/v_TennisSwing_g09_c02","TennisSwing/v_TennisSwing_g14_c04"]: return [] # this file is corrupted
        _maplabel = {"basketball_shooting":"Basketball", "biking":"Biking", "diving":"Diving", "golf_swing":"GolfSwing", "horse_riding":"HorseRiding", "soccer_juggling":"SoccerJuggling", "tennis_swing":"TennisSwing", "trampoline_jumping":"TrampolineJumping","volleyball_spiking":"VolleyballSpiking", "walking":"WalkingWithDog"}
        _maplabel = {_maplabel[k]:k for k in _maplabel.keys()}
        if label is None: label = videoname.split("/")[0]
        oldlabel = _maplabel[label] if label in _maplabel.keys() else label
        filename = os.path.join(self._annot_path, videoname+".xgtf")
        import xml.etree.ElementTree as ET
        tree = ET.parse(filename)
        root = tree.getroot()
        assert root[1].tag.endswith("data") and root[1][0].tag.endswith("sourcefile"), pdb.set_trace()
        toread = root[1][0]
        assert toread[0].tag.endswith("file"), pdb.set_trace()
        res = []
        for annot in toread[1:]: # for each human
            if annot.attrib["name"]!="PERSON": continue
            assert annot[0].attrib["name"]=="Location"
            bf = annot[0][0].attrib["framespan"].split(":")[0]
            ef = annot[0][-1].attrib["framespan"].split(":")[1]
            # read location
            location = np.empty( (int(ef)-int(bf)+1,5), dtype=np.float32)
            i = int(bf)
            for bb in annot[0]: 
                beginframe, endframe = bb.attrib["framespan"].split(":")
                if int(beginframe)<i: continue # for instance Basketball/v_Basketball_g08_c02
                if int(beginframe)>i:
                    location = location[:(j-int(bf)),:]
                    break # for instance Basketball/v_Basketball_g06_c02
                for j in range(int(beginframe), int(endframe)+1):
                    location[j-int(bf),0] = j
                    location[j-int(bf),1] = int(bb.attrib["x"])
                    location[j-int(bf),2] = int(bb.attrib["y"])
                    location[j-int(bf),3] = int(bb.attrib["x"])+int(bb.attrib["width"])
                    location[j-int(bf),4] = int(bb.attrib["y"])+int(bb.attrib["height"])
                    i += 1
            # set location inside
            h,w = 240, 320 # self.get_resolution(videoname)
            location[:,1] = np.minimum(w-1, np.maximum(0, location[:,1]))
            location[:,2] = np.minimum(h-1, np.maximum(0, location[:,2]))
            location[:,3] = np.minimum(w-1, np.maximum(0, location[:,3]))
            location[:,4] = np.minimum(h-1, np.maximum(0, location[:,4]))
            #assert location[-1,0]==int(ef) , pdb.set_trace()       
            # read framespan for the action
            
            for tlabel in annot[1:]:
                if tlabel.attrib["name"].replace(" ","")==oldlabel: # right action # replace " " by "" : BasketballDunk/v_BasketballDunk_g03_c06
                    for bval in tlabel:
                        if bval.attrib["value"]=="true":
                            beginframe, endframe = [int(x) for x in bval.attrib["framespan"].split(":")]
                            endframe = min(endframe, self.get_nframes(videoname))
                            if endframe < beginframe: continue
                            res.append( location[(beginframe-int(bf)):(endframe+1-int(bf)), :] )
                            if res[-1].size==0: del res[-1]
                            elif np.any(area2d(res[-1][:,1:])==0):
                                print videoname, "deleting empty box annotation"
                                del res[-1]
                else:  # other action
                    # check all false
                    allfalse = True
                    for tt in tlabel:
                        if tt.attrib["value"]=="true": allfalse = False
                    #assert allfalse, pdb.set_trace()
                    if not allfalse and label==videoname.split("/")[0]: print "warning", tlabel.attrib["name"], videoname
        return res
    
    def get_test_video_annotations(self):
        assert self._phase=='TEST'
        res = {}
        for v in self.test_videos:
            assert not v in res
            res[v] = {}
            if self._USE_MAT_GT:
                pass
            else:
                tubes = self.get_annot_video(v)
            res[v] = {'tubes': tubes, 'gt_classes': self.video_to_label[v]}
        return res
    def get_train_video_annotations(self):
        assert self._phase=='TRAIN'
        res = {}
        for v in self.train_videos:
            assert not v in res
            res[v] = {}
            if self._USE_MAT_GT:
                pass
            else:
                tubes = self.get_annot_video(v)
            res[v] = {'tubes': tubes, 'gt_classes': self.video_to_label[v]}
        return res
