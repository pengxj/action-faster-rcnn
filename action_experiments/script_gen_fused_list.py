import os, sys, pdb
import cPickle as pickle
import numpy as np

def get_image_format(imfile):
        name, ext = imfile.split('.')
        return '{:0%d}.%s' % (len(name), ext)

scr, infile1, infile2, outfile = sys.argv

test_images_a = [f.rstrip() for f in file(infile1)]
test_images_b = [f.rstrip() for f in file(infile2)]
out_obj = open(outfile, 'wb')

base_images = []
if len(test_images_b)<len(test_images_a):
	base_images = test_images_b
	format_a = get_image_format(os.path.basename(test_images_a[0])) # make sure this is color image
else:
	base_images = test_images_a
	format_a = get_image_format(os.path.basename(test_images_b[0]))



for i in range(len(base_images)):
	image_path = base_images[i]
	if 'UCF101' in infile1 or 'JHMDB' in infile1:
		dir_a = os.path.dirname(image_path)
	elif 'UCF-Sports' in infile1:
		dir_a = os.path.join(os.path.dirname(image_path),'jpeg')

	num_frm = int(os.path.basename(image_path).split('.')[0])
	image_path_a = format_a.format(num_frm)
	out_obj.write('{},{}\n'.format(os.path.join(dir_a, image_path_a), image_path))

out_obj.close()
