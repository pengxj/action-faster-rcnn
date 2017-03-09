# action-faster-rcnn
This repository is a strongly modified version for action detection originally from [py-faster-rnn](https://github.com/rbgirshick/py-faster-rcnn.git). It wraps three popular action detection dataset classes: UCF-Sports, JHMDB, and UCF101. Also, it provides useful action detection evaluation scripts for both frame level and video level.

## Installation

1. Clone this reporsitory
``` Shell
git clone --recursive https://github.com/pengxj/action-faster-rcnn.git
```

2. Build the Cython modules which mainly to compile the nms module
```Shell 
cd $FRCN_ROOT/lib
make
```
