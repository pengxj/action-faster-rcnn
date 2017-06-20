# set the binaries that have to be built
TARGETS := video_broxflow
#CXX := /scratch2/clear/local/bin64/g++

# set the build configuration set 
BUILD := release
#BUILD := debug
#BIT := 64
BIT := 64

# set bin and build dirs
BUILDDIR := .build_$(BUILD)$(BIT)
BINDIR := $(BUILD)$(BIT)

# some path variables
#SANDBOX := $(HOME)/sandbox/src
#COMMON := $(SANDBOX)/common

#ALEXDIR := $(HOME)/progs/alex
#ALEXDIR := /scratch2/clear/hewang/src

## include directories
#INCLUDEDIRS := \
#	/home/clear/nchesnea/heng/include \
##	/home/clear/hewang/test_opencv/dense_trajectory_release_v1.1 \
##	$(ALEXDIR)/common/src \
##	/scratch2/clear/local/include \

## library directories
#LIBDIRS := \
#	/home/clear/nchesnea/heng/lib 
##	/scratch2/clear/hewang/usr/local/lib
##	/scratch2/clear/revaud/local/lib \
##	/scratch2/clear/revaud/local/opencv2.4/lib


# using Heng's lib

# include directories
INCLUDEDIRS := \
	/idiap/home/xpeng/user/usr/opencv2.4.13/include/opencv2

# library directories
LIBDIRS := \
	/idiap/home/xpeng/user/usr/opencv2.4.13/lib\
	/idiap/home/xpeng/localscratch/github/action-faster-rcnn/opencv_cpu_broxflow_for_video

# libraries (without the prefix "lib") 
LIBS := \
	boost_program_options boost_regex boost_system boost_filesystem \
	opencv_core opencv_highgui opencv_video opencv_imgproc opencv_legacy opencv_ml opencv_calib3d opencv_features2d opencv_videostab opencv_nonfree\
	avformat avdevice avutil avcodec swscale of
#	lapack cblas atlas blas \

# set which libraries are used by which executable
LDLIBS = $(addprefix -L, $(LIBDIRS)) $(addprefix -l, $(LIBS))

CFLAGS=-Wall -g -O3 -msse4
LDFLAGS=-g -Wall -O3 -msse4
LIBFLAGS=-lm -ljpeg -lpng
LIBAFLAGS=-static /usr/lib64/libjpeg.a /usr/lib64/libpng.a /usr/lib64/libz.a /usr/lib64/libm.a

# set some flags and compiler/linker specific commands
CXXFLAGS = -m$(BIT) -pipe -D __STDC_CONSTANT_MACROS -D STD=std -Wall $(CXXFLAGS_$(BUILD)) $(addprefix -I, $(INCLUDEDIRS))
CXXFLAGS_debug := -ggdb
CXXFLAGS_release := -O3 -DNDEBUG -ggdb
LDFLAGS = -m$(BIT) -pipe -Wall $(LDFLAGS_$(BUILD))
LDFLAGS_debug := -ggdb
LDFLAGS_release := -O3 -ggdb

include make/generic.mk
