import os, sys, pdb
import cPickle as pickle
import numpy as np

scr, infile, outfile, LEN = sys.argv

half_len =  int(LEN)//2

test_images = [f.rstrip() for f in file(infile)]
preVideo = os.path.dirname(test_images[0])


out_obj = open(outfile, 'wb')
final_out = []
out_images = []
for i in range(len(test_images)):
    curVideo = os.path.dirname(test_images[i])
    out_images.append(test_images[i])
    if preVideo!=curVideo:
        tmp_image = out_images[-1]
        del out_images[-1]
        for k in range(half_len):
            del out_images[0]
            del out_images[-1]
        preVideo = curVideo
        # print out_images, len(out_images)
        final_out.append(out_images)
        out_images = []
        out_images.append(tmp_image)

# the last video
for k in range(half_len):
    del out_images[0]
    del out_images[-1]
final_out.append(out_images)

for out in final_out:
    for out_image in out:
        out_obj.write(out_image+'\n')
out_obj.close()


