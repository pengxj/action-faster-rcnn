#ifndef OF_H_
#define OF_H_
#include "CTensor.h"
/*
 * function for computing optical flow
 *
 * Arguments "image1" and "image2" are color images (3-dimensional arrays).
 * They are required to have the same dimension.
 * The computed optical flow is stored in "aResult". 
 * The array will be resized to fit the image dimension.
 *
 * The other arguments are tuning parameters
 * (the number in parentheses is the default value)
 *
 *       sigma:        (0.8) presmoothing of the input images
 *       alpha:        (80)  smoothness of the flow field
 *       gamma:        (5)   influence of the gradient constancy assumption
 *
 * The default values work best with image values in the range [0,255].
 *
 * Thomas Brox
 * U.C. Berkeley
 * Apr, 2010
 * All rights reserved
 */
void opticalFlow(CTensor<float>& aImage1, CTensor<float>& aImage2, CTensor<float>& aResult, float sigma=0.8f, float alpha=80.f, float gamma=5.f) ;



#endif /* OF_H_ */
