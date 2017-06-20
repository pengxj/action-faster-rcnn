#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <malloc.h>

#include "opticalflow.h"
#include "opticalflow_aux.h"
#include "solver.h"
#include "image.h"

#include <xmmintrin.h>
typedef __v4sf v4sf;

convolution_t *deriv, *deriv_flow;
float quarter_alpha, half_delta_over3, half_beta, half_gamma_over3;

/* perform flow computation at one level of the pyramid */
void compute_one_level(image_t *wx, image_t *wy, color_image_t *im1, color_image_t *im2, image_t *desc_flow_x, image_t *desc_flow_y, image_t *desc_weight, const optical_flow_params_t *params){ 
    const int width = wx->width, height = wx->height, stride=wx->stride;
    int i_inner_iteration;
  
    image_t *du = image_new(width,height), *dv = image_new(width,height), // the flow increment
        *mask = image_new(width,height), // mask containing 0 if a point goes outside image boundary, 1 otherwise
        *smooth_horiz = image_new(width,height), *smooth_vert = image_new(width,height), // horiz: (i,j) contains the diffusivity coeff from (i,j) to (i+1,j) 
        *uu = image_new(width,height), *vv = image_new(width,height), // flow plus flow increment
        *a11 = image_new(width,height), *a12 = image_new(width,height), *a22 = image_new(width,height), // system matrix A of Ax=b for each pixel
        *b1 = image_new(width,height), *b2 = image_new(width,height); // system matrix b of Ax=b for each pixel

    color_image_t *w_im2 = color_image_new(width,height), // warped second image
        *Ix = color_image_new(width,height), *Iy = color_image_new(width,height), *Iz = color_image_new(width,height), // first order derivatives
        *Ixx = color_image_new(width,height), *Ixy = color_image_new(width,height), *Iyy = color_image_new(width,height), *Ixz = color_image_new(width,height), *Iyz = color_image_new(width,height); // second order derivatives
  
    // warp second image
    image_warp(w_im2, mask, im2, wx, wy);
    // compute derivatives
    get_derivatives(im1, w_im2, deriv, Ix, Iy, Iz, Ixx, Ixy, Iyy, Ixz, Iyz);
    // erase du and dv
    image_erase(du);
    image_erase(dv);
    // initialize uu and vv
    memcpy(uu->data,wx->data,wx->stride*wx->height*sizeof(float));
    memcpy(vv->data,wy->data,wy->stride*wy->height*sizeof(float));
    // inner fixed point iterations
    for(i_inner_iteration = 0 ; i_inner_iteration < params->n_inner_iteration ; i_inner_iteration++){
        //  compute robust function and system
        compute_smoothness(smooth_horiz, smooth_vert, uu, vv, deriv_flow, quarter_alpha );
        compute_data_and_match(a11, a12, a22, b1, b2, mask, wx, wy, du, dv, uu, vv, Ix, Iy, Iz, Ixx, Ixy, Iyy, Ixz, Iyz, desc_weight, desc_flow_x, desc_flow_y, half_delta_over3, half_beta, half_gamma_over3);
        sub_laplacian(b1, wx, smooth_horiz, smooth_vert);
        sub_laplacian(b2, wy, smooth_horiz, smooth_vert);
        // solve system
        sor_coupled(du, dv, a11, a12, a22, b1, b2, smooth_horiz, smooth_vert, params->n_solver_iteration, params->sor_omega);
        // update flow plus flow increment
        int i;
        v4sf *uup = (v4sf*) uu->data, *vvp = (v4sf*) vv->data, *wxp = (v4sf*) wx->data, *wyp = (v4sf*) wy->data, *dup = (v4sf*) du->data, *dvp = (v4sf*) dv->data;
        for( i=0 ; i<height*stride/4 ; i++){
            (*uup) = (*wxp) + (*dup);
            (*vvp) = (*wyp) + (*dvp);
            uup+=1; vvp+=1; wxp+=1; wyp+=1;dup+=1;dvp+=1;
	   }
    }
    // add flow increment to current flow
    memcpy(wx->data,uu->data,uu->stride*uu->height*sizeof(float));
    memcpy(wy->data,vv->data,vv->stride*vv->height*sizeof(float)); 
    // free memory
    image_delete(du); image_delete(dv);
    image_delete(mask);
    image_delete(smooth_horiz); image_delete(smooth_vert);
    image_delete(uu); image_delete(vv);
    image_delete(a11); image_delete(a12); image_delete(a22);
    image_delete(b1); image_delete(b2);
    color_image_delete(w_im2); 
    color_image_delete(Ix); color_image_delete(Iy); color_image_delete(Iz);
    color_image_delete(Ixx); color_image_delete(Ixy); color_image_delete(Iyy); color_image_delete(Ixz); color_image_delete(Iyz);
}

/* set flow parameters to default */
void optical_flow_params_default(optical_flow_params_t *params){
    if(!params){
        fprintf(stderr,"Error optical_flow_params_default: argument is null\n");
        exit(1);
    }
    params->alpha = 6.0f;
    params->beta = 390.0f;
    params->gamma = 5.0f;
    params->delta = 0.5f;
    params->sigma = 0.6f;
    params->bk = 1.0f;
    params->eta = 0.95f;
    params->min_size = 25;
    params->n_inner_iteration = 5;  
    params->n_solver_iteration = 25;
    params->sor_omega = 1.60f;
}

/* set flow parameters to sintel one */
void optical_flow_params_sintel(optical_flow_params_t *params){
    if(!params){
        fprintf(stderr,"Error optical_flow_params_sintel: argument is null\n");
        exit(1);
    }
    params->alpha = 7.55f;
    params->beta = 390.0f;
    params->gamma = 4.95f;
    params->delta = 0.0;
    params->sigma = 0.7f;
    params->bk = 0.6;
    params->eta = 0.95f;
    params->min_size = 25;
    params->n_inner_iteration = 5;  
    params->n_solver_iteration = 25;
    params->sor_omega = 1.60f;
}

/* set flow parameters to middlebury one */
void optical_flow_params_middlebury(optical_flow_params_t *params){
    if(!params){
        fprintf(stderr,"Error optical_flow_params_middlebury: argument is null\n");
        exit(1);
    }
    params->alpha = 4.4f;
    params->beta = 0.2f;
    params->gamma = 5.95f;
    params->delta = 3.45f;
    params->sigma = 0.55f;
    params->bk = 0.8f;
    params->eta = 0.95f;
    params->min_size = 25;
    params->n_inner_iteration = 5;  
    params->n_solver_iteration = 25;
    params->sor_omega = 1.60f;
}

/* set flow parameters to kitti one */
void optical_flow_params_kitti(optical_flow_params_t *params){
    if(!params){
        fprintf(stderr,"Error optical_flow_params_kitti: argument is null\n");
        exit(1);
    }
    params->alpha = 5.4f;
    params->beta = 390.0f;
    params->gamma = 6.9f;
    params->delta = 0.1f;
    params->sigma = 0.45f;
    params->bk = 1.7f;
    params->eta = 0.95f;
    params->min_size = 25;
    params->n_inner_iteration = 5;  
    params->n_solver_iteration = 25;
    params->sor_omega = 1.60f;
}

/* Compute the optical flow between im1 and im2 and store it as two 1-channel images in wx for flow along x-axis and wy for flow along y-axis. match_x, match_y and match_z contains eventually the input matches (NULL for no match) at any scale. */
void optical_flow(image_t *wx, image_t *wy, const color_image_t *im1, const color_image_t *im2, optical_flow_params_t *params, const image_t *match_x, const image_t *match_y, const image_t *match_z){
  
    // Check parameters
    if(!params){
        params = (optical_flow_params_t*) malloc(sizeof(optical_flow_params_t));
        if(!params){
          fprintf(stderr,"error color_image_convolve_hv(): not enough memory\n");
          exit(1);
        }
        optical_flow_params_default(params);
    }

    // initialize global variables
    quarter_alpha = 0.25f*params->alpha;
    half_gamma_over3 = params->gamma*0.5f/3.0f;
    half_delta_over3 = params->delta*0.5f/3.0f;
    half_beta = params->beta*0.5f;
    float deriv_filter[3] = {0.0f, -8.0f/12.0f, 1.0f/12.0f};
    deriv = convolution_new(2, deriv_filter, 0);
    float deriv_filter_flow[2] = {0.0f, -0.5f};
    deriv_flow = convolution_new(1, deriv_filter_flow, 0);

    // presmooth images
    int width = im1->width, height = im1->height, filter_size;
    color_image_t *smooth_im1 = color_image_new(width, height), *smooth_im2 = color_image_new(width, height);
    float *presmooth_filter = gaussian_filter(params->sigma, &filter_size);
    convolution_t *presmoothing = convolution_new(filter_size, presmooth_filter, 1);
    color_image_convolve_hv(smooth_im1, im1, presmoothing, presmoothing);
    color_image_convolve_hv(smooth_im2, im2, presmoothing, presmoothing); 
    convolution_delete(presmoothing);
    free(presmooth_filter);
  
    // check descriptors
    image_t *desc_flow_x, *desc_flow_y, *desc_weight, *desc_flow_x_original=NULL, *desc_flow_y_original=NULL, *desc_weight_original=NULL;
    desc_flow_x = image_new(0,0);
    desc_flow_y = image_new(0,0);
    desc_weight = image_new(0,0);
    if(params->beta){
        if(match_x == NULL){
            params->beta = 0.0f;
            half_beta = 0.0f;
        }else{
            desc_flow_x_original = image_cpy(match_x);
            desc_flow_y_original = image_cpy(match_y);
            desc_weight_original = image_cpy(match_z);
        }
    }  

    // building pyramid
    color_image_pyramid_t *pyr1 = color_image_pyramid_create(smooth_im1, 1.0f/params->eta, params->min_size, 0.0f),
                        *pyr2 = color_image_pyramid_create(smooth_im2, 1.0f/params->eta, params->min_size, 0.0f);

    // loop over levels
    int k;
    for(k=pyr1->size-1; k>=0 ; k--){
        if(params->bk>0.0f) half_beta = 0.5f*params->beta * pow(((float)k)/((float)pyr1->size-1),params->bk);
        if(k == pyr1->size-1){ 
            // first level	  
            // allocate wx and wy
            resize_if_needed_newsize(wx, pyr1->images[k]->width, pyr1->images[k]->height);
            resize_if_needed_newsize(wy, pyr1->images[k]->width, pyr1->images[k]->height);
            image_erase(wx); image_erase(wy);
	    }else{ 
            // resize flow to the new pyramid level size and multiply it by 1/eta
            image_t *tmp = image_new(pyr1->images[k]->width, pyr1->images[k]->height);
            image_resize_bilinear_newsize(tmp, wx, pyr1->images[k]->width, pyr1->images[k]->height);
            resize_if_needed_newsize(wx, pyr1->images[k]->width, pyr1->images[k]->height);
            memcpy(wx->data, tmp->data, tmp->stride*tmp->height*sizeof(float)); 
            image_mul_scalar(wx, 1.0f/params->eta);
            image_resize_bilinear_newsize(tmp, wy, pyr1->images[k]->width, pyr1->images[k]->height);
            resize_if_needed_newsize(wy, pyr1->images[k]->width, pyr1->images[k]->height);
            memcpy(wy->data, tmp->data, tmp->stride*tmp->height*sizeof(float));
            image_mul_scalar(wy, 1.0f/params->eta);     
            image_delete(tmp);
        }

        // resize descriptors
        if(params->beta){
            resize_if_needed_newsize(desc_flow_x, pyr1->images[k]->width, pyr1->images[k]->height);
            resize_if_needed_newsize(desc_flow_y, pyr1->images[k]->width, pyr1->images[k]->height);
            resize_if_needed_newsize(desc_weight, pyr1->images[k]->width, pyr1->images[k]->height);
            descflow_resize(desc_flow_x,desc_flow_y,desc_weight,desc_flow_x_original,desc_flow_y_original,desc_weight_original);
	    }
     
        compute_one_level(wx, wy, pyr1->images[k], pyr2->images[k], desc_flow_x, desc_flow_y, desc_weight, params);
      
    }
    
    color_image_pyramid_delete(pyr1); color_image_pyramid_delete(pyr2);
    
    // do a last iteration without descriptor if bk==0
    if(params->beta>0.0f && params->bk==0.0f){
        half_beta = 0.0f;
        compute_one_level(wx, wy, smooth_im1, smooth_im2, desc_flow_x, desc_flow_y, desc_weight, params);
        half_beta = 0.5f*params->beta;
    }

    // free memory
    color_image_delete(smooth_im1);
    color_image_delete(smooth_im2);
    image_delete(desc_flow_x); image_delete(desc_flow_y); image_delete(desc_weight);
    convolution_delete(deriv);
    convolution_delete(deriv_flow);
    if(params->beta){image_delete(desc_flow_x_original); image_delete(desc_flow_y_original); image_delete(desc_weight_original);}
}
