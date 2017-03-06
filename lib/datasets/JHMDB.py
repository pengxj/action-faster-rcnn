import sys, os
from PIL import Image
from scipy.io import loadmat
import datasets
from datasets.imdb import imdb
import numpy as np
import scipy.sparse
import cPickle
import pdb

class JHMDB(imdb):
    def __init__(self, image_set, PHASE):
        imdb.__init__(self, image_set, PHASE)
        if PHASE=='TRAIN':
            self._image_set = './action_experiments/listfiles/' + image_set + '.trainlist' # you need a full path for image list and data path
        else:
            self._image_set = './action_experiments/listfiles/' + image_set + '.testlist'

        self._annot_path = "/home/lear/pweinzae/scratch/data/JHMDB/original/puppet_mask" # you only have annotations in RGB data folder        
        self._SPLIT = int(image_set.split('_')[-1])

        if 'RGB' in image_set and 'FLOW' in image_set: # for 2stream fusion
            self._data_path = '/home/lear/xpeng/data/JHMDB/flows_color'
        else:
            self._MOD = image_set.split('_')[1]
            self._LEN = image_set.split('_')[2]
            self._data_path = None
            if self._MOD=='RGB': self._data_path = '/home/lear/xpeng/data/JHMDB/frames'
            if self._MOD=='FLOW': self._data_path = '/home/lear/xpeng/data/JHMDB/flows_color'

        self._classes = ('__background__', 
                         'brush_hair', 'catch', 'clap', 'climb_stairs', 'golf', 
                         'jump', 'kick_ball', 'pick', 'pour', 'pullup', 'push',
                         'run', 'shoot_ball', 'shoot_bow', 'shoot_gun', 'sit',
                         'stand', 'swing_baseball', 'throw', 'walk', 'wave') # 22
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_index = self._load_image_set_index()

        self.test_videos = self.get_test_videos(self._SPLIT)

        # Default to roidb handler
        self._roidb_handler = self.gt_roidb
    # videos / images 
    def image_path_at(self, i):
        return os.path.join(self._data_path, self._image_index[i])

    def prepare_traintest(self): # generating train/test file list according to self.videos
            pass

    # train / test splits 
    def get_train_videos(self, split):
        assert split<3
        tr_videos = [os.path.join(label,l.split()[0][:-4]) for label in self._classes[1:] 
                                   for l in file("./action_experiments/listfiles/JHMDB_splits/%s_test_split%d.txt"%(label,split+1)) 
                                   if l.split()[1][0]=="1"]
        return tr_videos
    
    def get_test_videos(self, split):
        assert split<3
        ts_videos = [os.path.join(label,l.split()[0][:-4]) for label in self._classes[1:] 
                                 for l in file("./action_experiments/listfiles/JHMDB_splits/%s_test_split%d.txt"%(label,split+1)) 
                                 if l.split()[1][0]=="2"]
        return ts_videos

    # annotation: warning few images do not have annotation
    def _get_puppet_mask_file(self, videoname):
        return os.path.join(self._annot_path, videoname, "puppet_mask.mat")

    def get_annot_image_mask(self, videoname, n):
        assert os.path.exists(self._get_puppet_mask_file(videoname))
        m = loadmat(self._get_puppet_mask_file(videoname))["part_mask"]
        if n-1<m.shape[2]: return m[:,:,n-1]>0
        else: return m[:,:,-1]>0

    def get_annot_image_boxes(self, videoname, n):
#        pdb.set_trace()
        mask = self.get_annot_image_mask(videoname, n)
        m = self.mask_to_bbox(mask)
        if m is None:
            pdb.set_trace() 
            m = np.zeros((0,4), dtype=np.float32)
        if m.shape[0]>1:
            pdb.set_trace()
        return m

    def mask_to_bbox(self,mask):
         # you are aware that only 1 box for each frame
        return np.array(Image.fromarray(mask.astype(np.uint8)).getbbox(), dtype=np.float32).reshape(1,4)-np.array([0,0,1,1])

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

    # roi database preparation
    def gt_roidb(self):

        cache_file = self.cache_path
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb
            
        roidb = [self._load_JHMDB_annotation(index)
                for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)
        return roidb

    # def gt_roidb(self):
    #     """
    #     Return the database of ground-truth regions of interest.

    #     This function loads/saves from/to a cache file to speed up future calls.
    #     """
    #     gt_roidb = [self._load_JHMDB_annotation(index)
    #                 for index in self.image_index]
    #     return gt_roidb

    def _load_JHMDB_annotation(self, index):
        """
        Load image and bounding boxes info 
        """
        index = index.split(',')[-1] # to support 2 stream filelist input
        videoname = os.path.dirname(index)
        # pdb.set_trace()
        frm = int(index.split('/')[-1].split('.')[0])

        num_objs = 1 # num_objs is num humans, for JHMDB only one instance in a frame

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        boxes[0,:] = self.get_annot_image_boxes(videoname, frm)
        cls = self._class_to_ind[videoname.split('/')[0]]
        gt_classes[0] = cls
        overlaps[0, cls] = 1.0       
       
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
            tubes = []
            # only one object in JHMDB
            mask = loadmat(self._get_puppet_mask_file(v))["part_mask"]
            tube = np.empty((mask.shape[2], 5), dtype=np.int32)
            for i in range(mask.shape[2]):
                box = self.mask_to_bbox(mask[:,:, i])
                tube[i, 0] = i + 1 
                tube[i, 1:] = box
            tubes.append(tube)
            res[v] = {'tubes': tubes, 'gt_classes': self._class_to_ind[v.split('/')[0]]}
        return res