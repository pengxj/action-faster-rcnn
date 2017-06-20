#ifndef FLO_H_
#define FLO_H_

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <ctype.h>
#include <unistd.h>

#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <list>
#include <string>

//typedef unsigned char uchar;
#define TAG_FLOAT 202021.25  // check for this when READING the file
#define UNKNOWN_FLOW_THRESH 1e9
#define MAXCOLS 60

int verbose = 1;
int ncols = 0;
int colorwheel[MAXCOLS][3];

// return whether flow vector is unknown
bool unknown_flow(float u, float v) {
	return (fabs(u) >  UNKNOWN_FLOW_THRESH) 
	|| (fabs(v) >  UNKNOWN_FLOW_THRESH)
	|| isnan(u) || isnan(v);
}

//bool unknown_flow(float *f) {
//	return unknown_flow(f[0], f[1]);
//}

void setcols(int r, int g, int b, int k)
{
    colorwheel[k][0] = r;
    colorwheel[k][1] = g;
    colorwheel[k][2] = b;
}

void makecolorwheel()
{
    // relative lengths of color transitions:
    // these are chosen based on perceptual similarity
    // (e.g. one can distinguish more shades between red and yellow 
    //  than between yellow and green)
    int RY = 15;
    int YG = 6;
    int GC = 4;
    int CB = 11;
    int BM = 13;
    int MR = 6;
    ncols = RY + YG + GC + CB + BM + MR;
    //printf("ncols = %d\n", ncols);
    if (ncols > MAXCOLS)
	exit(1);
    int i;
    int k = 0;
    for (i = 0; i < RY; i++) setcols(255,	   255*i/RY,	 0,	       k++);
    for (i = 0; i < YG; i++) setcols(255-255*i/YG, 255,		 0,	       k++);
    for (i = 0; i < GC; i++) setcols(0,		   255,		 255*i/GC,     k++);
    for (i = 0; i < CB; i++) setcols(0,		   255-255*i/CB, 255,	       k++);
    for (i = 0; i < BM; i++) setcols(255*i/BM,	   0,		 255,	       k++);
    for (i = 0; i < MR; i++) setcols(255,	   0,		 255-255*i/MR, k++);
}

void computeColor(float fx, float fy, uchar *pix)
{
    if (ncols == 0)
	makecolorwheel();

    float rad = sqrt(fx * fx + fy * fy);
    float a = atan2(-fy, -fx) / M_PI;
    float fk = (a + 1.0) / 2.0 * (ncols-1);
    int k0 = (int)fk;
    int k1 = (k0 + 1) % ncols;
    float f = fk - k0;
    //f = 0; // uncomment to see original color wheel
    for (int b = 0; b < 3; b++) {
	float col0 = colorwheel[k0][b] / 255.0;
	float col1 = colorwheel[k1][b] / 255.0;
	float col = (1 - f) * col0 + f * col1;
	if (rad <= 1)
	    col = 1 - rad * (1 - col); // increase saturation with radius
	else
	    col *= .75; // out of range
	pix[2 - b] = (int)(255.0 * col);
    }
}

//// read a flow file into a 2-channel opencv image
//IplImage* ReadFlowFile(const char* filename)
//{
//	if (filename == NULL)
//		std::cerr << "ReadFlowFile: empty filename" << std::endl;

//	char *dot = strrchr(filename, '.');
//	if (strcmp(dot, ".flo") != 0)
//		std::cerr << "ReadFlowFile (" << filename << "): extension .flo expected" << std::endl;

//	FILE *stream = fopen(filename, "rb");
//	if (stream == 0)
//		std::cerr << "ReadFlowFile: could not open " << filename << std::endl;
//	
//	int width, height;
//	float tag;

//	if ((int)fread(&tag,	sizeof(float), 1, stream) != 1 ||
//		(int)fread(&width,  sizeof(int),   1, stream) != 1 ||
//		(int)fread(&height, sizeof(int),   1, stream) != 1)
//		std::cerr << "ReadFlowFile: problem reading file " << filename << std::endl; 

//	if (tag != TAG_FLOAT) // simple test for correct endian-ness
//		std::cerr << "ReadFlowFile(" << filename << "): wrong tag (possibly due to big-endian machine?)" << std::endl;

//	// another sanity check to see that integers were read correctly (99999 should do the trick...)
//	if (width < 1 || width > 99999)
//		std::cerr << "ReadFlowFile(" << filename << "): illegal width " << width << std::endl;

//	if (height < 1 || height > 99999)
//		std::cerr << "ReadFlowFile(" << filename << "): illegal height " << height << std::endl;
//	
//	IplImage* img = cvCreateImage(cvSize(width, height), IPL_DEPTH_32F, 2);
//	int nBands = 2;
////	CShape sh(width, height, nBands);
////	img.ReAllocate(sh);

//	//printf("reading %d x %d x 2 = %d floats\n", width, height, width*height*2);
//	int n = nBands * width;
//	//float* ptr = (float*)malloc(n*sizeof(float));
//	for (int y = 0; y < height; y++) {
//		float* ptr = (float*)(img->imageData + img->widthStep*y);
//		//float* ptr = &img.Pixel(0, y, 0);
//		if ((int)fread(ptr, sizeof(float), n, stream) != n)
//			std::cerr << "ReadFlowFile(" << filename << "): file is too short" << std::endl;
//	}

//	if (fgetc(stream) != EOF)
//		std::cerr << "ReadFlowFile(" << filename << "): file is too long" << std::endl;

//	fclose(stream);

//	return img;
//}

void MotionToColor(IplImage* flow, IplImage* img, float maxmotion = 0)
{
	//CShape sh = motim.Shape();
	int width = flow->width, height = flow->height;
	//colim.ReAllocate(CShape(width, height, 3));
	int x, y;
	// determine motion range:
	float maxx = -999, maxy = -999;
	float minx =  999, miny =  999;
	float maxrad = -1;
	std::list<float> xValues(0);
	std::list<float> yValues(0);
//	std::list<float> radValues(0);

	for (y = 0; y < height; y++) {
		const float* f = (const float*)(flow->imageData + flow->widthStep*y);
		for (x = 0; x < width; x++) {
			float fx = f[2*x];
			float fy = f[2*x+1];
			xValues.push_back(fx);
			yValues.push_back(fy);

			if (unknown_flow(fx, fy))
				continue;
//			maxx = std::max<float>(maxx, fx);
//			maxy = std::max<float>(maxy, fy);
//			minx = std::min<float>(minx, fx);
//			miny = std::min<float>(miny, fy);
//			float rad = sqrt(fx * fx + fy * fy);
//			maxrad = std::max<float>(maxrad, rad);
//			radValues.push_back(maxrad);
		}
	}

	xValues.sort();
	yValues.sort();
//	radValues.sort();

	// discard the smallest 5% and biggest 5%
	int size = xValues.size()/20;
	for(int m = 0; m < size; m++) {
		xValues.pop_front();
		yValues.pop_front();
	//	radValues.pop_front();
	}
	minx = xValues.front();
	miny = xValues.front();

	for(int m = 0; m < size; m++) {
		xValues.pop_back();
		yValues.pop_back();
//		radValues.pop_back();
	}
	maxx = xValues.back();
	maxy = yValues.back();
//	maxrad = radValues.back();	

	for (y = 0; y < height; y++) {
		float* f = (float*)(flow->imageData + flow->widthStep*y);
		for (x = 0; x < width; x++) {
			float fx = f[2*x] = std::min<float>(std::max<float>(f[2*x], minx), maxx);
			float fy = f[2*x+1] = std::min<float>(std::max<float>(f[2*x+1], miny), maxy);

			if (unknown_flow(fx, fy))
				continue;

			float rad = sqrt(fx * fx + fy * fy);
			maxrad = std::max<float>(maxrad, rad);
		}
	}

//	std::cerr << "max motion: " << maxrad << "  motion range: u = " << minx << " .. " << maxx << ";  v = " << miny << " .. " << maxy << std::endl;

	if (maxmotion > maxrad) // i.e., specified on commandline
		maxrad = maxmotion;

	if (maxrad == 0) // if flow == 0 everywhere
		maxrad = 1;

//	if (verbose)
//		fprintf(stderr, "normalizing by %g\n", maxrad);

	for (y = 0; y < height; y++) {
		const float* f = (const float*)(flow->imageData + flow->widthStep*y);
		for (x = 0; x < width; x++) {
			uchar* im = &((uchar*)(img->imageData + img->widthStep*y))[3*x];
			float fx = f[2*x];
			float fy = f[2*x+1];

			//uchar *pix = &colim.Pixel(x, y, 0);
			if (unknown_flow(fx, fy)) {
				im[0] = im[1] = im[2] = 0;
			} else {
				computeColor(fx/maxrad, fy/maxrad, im);
			}
		}
	}
}

#endif /*FLO_H_*/
