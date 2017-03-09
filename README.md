# action-faster-rcnn
This repository is a strongly modified version for action detection originally from [py-faster-rnn](https://github.com/rbgirshick/py-faster-rcnn.git). It wraps three popular action detection dataset classes: UCF-Sports, JHMDB, and UCF101. Also, it provides useful action detection evaluation scripts for both frame level and video level. 

### Installation
1. Clone this reporsitory
  ```Shell
  git clone --recursive https://github.com/pengxj/action-faster-rcnn.git
  ```
  
2. Build the Cython modules which mainly compiles the nms module
  ```Shell 
  cd $THIS_ROOT/lib
  make
  ```
  
3. Build Caffe and pycaffe
  ```Shell
  cd $THIS_ROOT/caffe-fast-rcnn-faster-rcnn-upstream-33f2445
  # Now follow the Caffe installation instructions here:
  #   http://caffe.berkeleyvision.org/installation.html
  ```
  
4. Dive into the code(will be detailed later)
  ```Shell
  training script: ./action_experiments/scripts/train_action_det.sh
  evaluation scripts: action_tools/action_util.py ucfsports_eval.py jhmdb_eval.py ucf101_eval.py fusion_eval.py
  ```

### Citation

If you find this repository useful in your research, please consider citing:

    @inproceedings{peng2016multi,
    title={Multi-region two-stream R-CNN for action detection},
    author={Peng, Xiaojiang and Schmid, Cordelia},
    booktitle={European Conference on Computer Vision},
    pages={744--759},
    year={2016},
    organization={Springer}}
  
  
    @inproceedings{renNIPS15fasterrcnn,
        Author = {Shaoqing Ren and Kaiming He and Ross Girshick and Jian Sun},
        Title = {Faster {R-CNN}: Towards Real-Time Object Detection
                 with Region Proposal Networks},
        Booktitle = {Advances in Neural Information Processing Systems ({NIPS})},
        Year = {2015}
    }
