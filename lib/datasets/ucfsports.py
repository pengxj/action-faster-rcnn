
import datasets
import os
from datasets.imdb import imdb
import numpy as np
import scipy.sparse
import cPickle
import pdb


class ucfsports(imdb):
    def __init__(self, image_set, PHASE):
        imdb.__init__(self, image_set, PHASE)
        if PHASE=='TRAIN':
            self._image_set = './action_experiments/listfiles/' + image_set + '.trainlist' # you need a full path for image list and data path
        else:
            self._image_set = './action_experiments/listfiles/' + image_set + '.testlist'

        self._MOD = image_set.split('_')[1]
        self._LEN = image_set.split('_')[2]
        self._data_path = None
        self._annot_path = '/home/lear/xpeng/data/ucf_sports_actions/UCFsports/data' # you only have annotations in RGB data folder
        if self._MOD=='RGB': self._data_path = '/home/lear/xpeng/data/ucf_sports_actions/UCFsports/data'
        if self._MOD=='FLOW': self._data_path = '/home/lear/xpeng/data/ucf_sports_actions/broxflow'

        self._classes = ('__background__', 
                         'Diving', 'Golf', 'Kicking', 'Lifting', 'Riding', 
                         'Run', 'SkateBoarding', 'Swing1', 'Swing2', 'Walk')

        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_index = self._load_image_set_index()

        self.videos = [l.split() for l in file(os.path.join(self._annot_path,'videos.txt'))] # videos include train/test info, video names, etc.
        self.test_videos = [v[0] for v in self.videos if v[2]=="test"] 
        self.video_to_label = {v[0]: self._class_to_ind[v[1]] for v in self.videos }

        self._roidb_handler = self.gen_roidb

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return os.path.join(self._data_path, self._image_index[i])

    def prepare_traintest(self): # generating train/test file list according to self.videos
            pass

    def get_human_annot_file(self,videoname):
        return os.path.join(self._annot_path, videoname, "humans.txt") 


    def get_human_annot(self,videoname):
        return np.loadtxt( self.get_human_annot_file(videoname) , dtype=np.int32)

    def get_nhumans(self,videoname):
        return self.get_human_annot(videoname).shape[1]//4

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
        return image_index


    def gen_roidb(self):
        cache_file = self.cache_path
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb
        
        roidb = [self._load_ucfsports_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)
        return roidb

    def _load_ucfsports_annotation(self, index):
        """
        Load image and bounding boxes info 
        """
        videoname = index.split('/')[0]
        frm = int(index.split('/')[-1].split('.')[0])

        annots = self.get_human_annot(videoname)
        num_objs = annots.shape[1]//4 # num_objs is num humans

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        for ix in range(num_objs):
            tmpbox = annots[frm-1, ix*4+1:ix*4+5]
            boxes[ix,:] = [tmpbox[0], tmpbox[1], tmpbox[0]+tmpbox[2], tmpbox[1]+tmpbox[3]]
            cls = self.video_to_label[videoname]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
       
        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False}

    def get_test_video_annotations(self):
        assert self._phase=='TEST'
        res = {}
        for v in self.test_videos:
            assert not v in res
            res[v] = {}
            annots = self.get_human_annot(v)
            num_objs = annots.shape[1]//4
            tubes = []
            for i in range(num_objs):
                annots[:, 1+2+4*i] += annots[:, 1+4*i]
                annots[:, 1+3+4*i] += annots[:, 1+4*i]
                idx = [0] + range(1+4*i, 5+4*i)
                tube = annots[:,idx]
                tubes.append(tube)
            res[v] = {'tubes': tubes, 'gt_classes': self.video_to_label[v]}
        return res