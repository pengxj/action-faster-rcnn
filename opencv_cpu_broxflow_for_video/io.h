#ifndef __IO_H__
#define __IO_H__

#include <stdlib.h>

#include "image.h"

/* read a flow file and returns a pointer with two images containing the flow along x and y axis */
image_t** readFlowFile(const char* filename);

/* write a flow to a file */
void writeFlowFile(const char* filename, const image_t *flowx, const image_t *flowy);

/* load a color image from a file in jpg or ppm*/
color_image_t *color_image_load(const char *fname);

#endif