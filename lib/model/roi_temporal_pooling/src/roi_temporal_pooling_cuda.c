#include <THC/THC.h>
#include <math.h>
#include "roi_temporal_pooling_kernel.h"

extern THCState *state;

int roi_temporal_pooling_forward_cuda(int pooled_length, int pooled_height, int pooled_width, float temporal_scale, float ctx_ratio,
                        THCudaTensor * features, THCudaTensor * rois, THCudaTensor * output, THCudaIntTensor * argmax)
{
    // Grab the input tensor
    float * data_flat = THCudaTensor_data(state, features);
    float * rois_flat = THCudaTensor_data(state, rois);

    float * output_flat = THCudaTensor_data(state, output);
    int * argmax_flat = THCudaIntTensor_data(state, argmax);

    // Number of ROIs
    int num_rois = THCudaTensor_size(state, rois, 0);
    int size_rois = THCudaTensor_size(state, rois, 1);
    if (size_rois != 3)
    {
        return 0;
    }

    // batch size
    // int batch_size = THCudaTensor_size(state, features, 0);
    // if (batch_size != 1)
    // {
    //     return 0;
    // }
    // data length
    int data_length = THCudaTensor_size(state, features, 2);
    // data height
    int data_height = THCudaTensor_size(state, features, 3);
    // data width
    int data_width = THCudaTensor_size(state, features, 4);
    // Number of channels
    int num_channels = THCudaTensor_size(state, features, 1);

    cudaStream_t stream = THCState_getCurrentStream(state);

    ROITemporalPoolForwardLaucher(
        data_flat, temporal_scale, ctx_ratio, num_rois, data_length, data_height,
        data_width, num_channels, pooled_length,
        pooled_height, pooled_width, rois_flat,
        output_flat, argmax_flat, stream);

    return 1;
}

int roi_temporal_pooling_backward_cuda(int pooled_length, int pooled_height, int pooled_width, float temporal_scale, float ctx_ratio,
                        THCudaTensor * top_grad, THCudaTensor * rois, THCudaTensor * bottom_grad, THCudaIntTensor * argmax)
{
    // Grab the input tensor
    float * top_grad_flat = THCudaTensor_data(state, top_grad);
    float * rois_flat = THCudaTensor_data(state, rois);

    float * bottom_grad_flat = THCudaTensor_data(state, bottom_grad);
    int * argmax_flat = THCudaIntTensor_data(state, argmax);

    // Number of ROIs
    int num_rois = THCudaTensor_size(state, rois, 0);
    int size_rois = THCudaTensor_size(state, rois, 1);
    if (size_rois != 3)
    {
        return 0;
    }

    // batch size
    int batch_size = THCudaTensor_size(state, bottom_grad, 0);
    // if (batch_size != 1)
    // {
    //     return 0;
    // }
    // data length
    int data_length = THCudaTensor_size(state, bottom_grad, 2);
    // data height
    int data_height = THCudaTensor_size(state, bottom_grad, 3);
    // data width
    int data_width = THCudaTensor_size(state, bottom_grad, 4);
    // Number of channels
    int num_channels = THCudaTensor_size(state, bottom_grad, 1);

    cudaStream_t stream = THCState_getCurrentStream(state);
    ROITemporalPoolBackwardLaucher(
        top_grad_flat, temporal_scale, ctx_ratio, batch_size, num_rois, data_length, data_height,
        data_width, num_channels, pooled_length, pooled_height,
        pooled_width, rois_flat,
        bottom_grad_flat, argmax_flat, stream);

    return 1;
}
