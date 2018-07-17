// #ifdef __cplusplus
// extern "C" {
// #endif

#include <stdio.h>
#include <vector>
#include <math.h>
#include <float.h>
#include "roi_temporal_pooling_kernel.h"


#define DIVUP(m, n) ((m) / (m) + ((m) % (n) > 0))

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

__global__ void ROITemporalPoolForward(const int nthreads, const float* bottom_data,
    const float temporal_scale, const float ctx_ratio, const int length, const int height, const int width,
    const int channels, const int pooled_length, const int pooled_height, const int pooled_width,
    const float* bottom_rois, float* top_data, int* argmax_data)
{
    CUDA_KERNEL_LOOP(index, nthreads)
    {
        // (n, c, pl, ph, pw) is an element in the pooled output
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int pl = (index / pooled_width / pooled_height ) % pooled_length;
        int c  = (index / pooled_width / pooled_height / pooled_length) % channels;
        int n  = index / pooled_width / pooled_height / pooled_length / channels;

        // bottom_rois += n * 3;
        int roi_batch_ind = bottom_rois[n * 3 + 0];
        int roi_start_l = round(bottom_rois[n * 3 + 1] * temporal_scale);
        int roi_end_l = round(bottom_rois[n * 3 + 2] * temporal_scale);
        int roi_start_h = 0;
        int roi_end_h = height -1;
        int roi_start_w = 0;
        int roi_end_w = width -1;

        // Add context for temporal segment
        int roi_length_tmp = fmaxf(roi_end_l - roi_start_l + 1, 1);
        roi_start_l = roi_start_l - 0.5*(ctx_ratio-1)*roi_length_tmp;
        roi_end_l = roi_end_l + 0.5*(ctx_ratio-1)*roi_length_tmp;

        // Force malformed ROIs to be 1x1, Note that we just have temporal annotation.
        int roi_length = fmaxf(roi_end_l - roi_start_l + 1, 1);
        float bin_size_l = (float)(roi_length) / (float)(pooled_length);
        int roi_height = fmaxf(roi_end_h - roi_start_h + 1, 1);
        float bin_size_h = (float)(roi_height) / (float)(pooled_height);
        int roi_width = fmaxf(roi_end_w - roi_start_w + 1, 1);
        float bin_size_w = (float)(roi_width) / (float)(pooled_width);

        int lstart = (int)(floor((float)(pl) * bin_size_l));
        int lend = (int)(ceil((float)(pl + 1) * bin_size_l));
        int hstart = (int)(floor((float)(ph) * bin_size_h));
        int hend = (int)(ceil((float)(ph + 1) * bin_size_h));
        int wstart = (int)(floor((float)(pw) * bin_size_w));
        int wend = (int)(ceil((float)(pw + 1) * bin_size_w));

        // Add roi offsets and clip to input boundaries
        lstart = fminf(fmaxf(lstart + roi_start_l, 0), length);
        lend = fminf(fmaxf(lend + roi_start_l, 0), length);
        hstart = fminf(fmaxf(hstart + roi_start_h, 0), height);
        hend = fminf(fmaxf(hend + roi_start_h, 0), height);
        wstart = fminf(fmaxf(wstart + roi_start_w, 0), width);
        wend = fminf(fmaxf(wend + roi_start_w, 0), width);
        bool is_empty = (lend <= lstart) || (hend <= hstart) || (wend <= wstart);

        // Define an empty pooling region to be zero
        float maxval = is_empty ? 0 : -FLT_MAX;
        // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
        int maxidx = -1;
        // bottom_data += roi_batch_ind * channels * length * height * width;

        int bottom_data_batch_offset = roi_batch_ind * channels * length * height * width;
        int bottom_data_offset = bottom_data_batch_offset + c * length * height * width;
        
        for (int l = lstart; l < lend; ++l) {
            for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {
                    int bottom_index = (l * height + h) * width + w;
                    if (bottom_data[bottom_data_offset + bottom_index] > maxval) {
                        maxval = bottom_data[bottom_data_offset + bottom_index];
                        maxidx = bottom_data_offset + bottom_index;
                    }
                }
            }
        }
        top_data[index] = maxval;
        if (argmax_data != NULL)
            argmax_data[index] = maxidx;
    }
}

int ROITemporalPoolForwardLaucher(
    const float* bottom_data, const float temporal_scale, const float ctx_ratio, const int num_rois, const int length, const int height,
    const int width, const int channels, const int pooled_length, const int pooled_height,
    const int pooled_width, const float* bottom_rois,
    float* top_data, int* argmax_data, cudaStream_t stream)
{
    const int kThreadsPerBlock = 1024;
    int output_size = num_rois * pooled_length * pooled_height * pooled_width * channels;
    cudaError_t err;

    ROITemporalPoolForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
      output_size, bottom_data, temporal_scale, ctx_ratio, length, height, width, channels, pooled_length, pooled_height,
      pooled_width, bottom_rois, top_data, argmax_data);

    // dim3 blocks(DIVUP(output_size, kThreadsPerBlock),
    //             DIVUP(output_size, kThreadsPerBlock));
    // dim3 threads(kThreadsPerBlock);
    //
    // ROITemporalPoolForward<<<blocks, threads, 0, stream>>>(
    //   output_size, bottom_data, temporal_scale, height, width, channels, pooled_height,
    //   pooled_width, bottom_rois, top_data, argmax_data);

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}


__global__ void ROITemporalPoolBackward(const int nthreads, const float* top_diff,
    const int* argmax_data, const int num_rois, const float temporal_scale, const float ctx_ratio,
    const int length, const int height, const int width, const int channels,
    const int pooled_length, const int pooled_height, const int pooled_width, float* bottom_diff,
    const float* bottom_rois) {
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
        // (n, c, l, h, w) is an element in the bottom diff
        int w = index % width;
        int h = (index / width) % height;
        int l = (index / width / height ) % length;
        int c  = (index / width / height / length) % channels;
        int n  = index / width / height / length / channels;

        int roi_height = height;
        int roi_width = width;
        float gradient = 0;
        // Accumulate gradient over all ROIs that pooled this element
        for (int roi_n = 0; roi_n < num_rois; ++roi_n)
        {
            const float* offset_bottom_rois = bottom_rois + roi_n * 3;
            int roi_batch_ind = offset_bottom_rois[0];
            // Skip if ROI's batch index doesn't match n
            if (n != roi_batch_ind) {
                continue;
            }

            int roi_start_l = round(offset_bottom_rois[1] * temporal_scale);
            int roi_end_l = round(offset_bottom_rois[2] * temporal_scale);
            int roi_start_h = 0;
            int roi_end_h = roi_height - 1;
            int roi_start_w = 0;
            int roi_end_w = roi_width - 1;

            // Add context for temporal segment
            int roi_length_tmp = fmaxf(roi_end_l - roi_start_l + 1, 1);
            roi_start_l = roi_start_l - 0.5*(ctx_ratio-1)*roi_length_tmp;
            roi_end_l = roi_end_l + 0.5*(ctx_ratio-1)*roi_length_tmp;

            // Skip if ROI doesn't include (l, h, w)
            const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
                               h >= roi_start_h && h <= roi_end_h && l >= roi_start_l && l <= roi_end_l);
            if (!in_roi) {
                continue;
            }

            int offset = roi_n * pooled_length * pooled_height * pooled_width * channels;
            const float* offset_top_diff = top_diff + offset;
            const int* offset_argmax_data = argmax_data + offset;

            // Compute feasible set of pooled units that could have pooled
            // this bottom unit

            // Force malformed ROIs to be 1x1
            int roi_length = fmaxf(roi_end_l - roi_start_l + 1, 1);
            // int roi_height = fmaxf(roi_end_h - roi_start_h + 1, 1);
            // int roi_width = fmaxf(roi_end_w - roi_start_w + 1, 1);
            
            float bin_size_l = (float)(roi_length) / (float)(pooled_length);
            float bin_size_h = (float)(roi_height) / (float)(pooled_height);
            float bin_size_w = (float)(roi_width) / (float)(pooled_width);

            int plstart = floor((float)(l - roi_start_l) / bin_size_l);
            int plend = ceil((float)(l - roi_start_l + 1) / bin_size_l);
            int phstart = floor((float)(h - roi_start_h) / bin_size_h);
            int phend = ceil((float)(h - roi_start_h + 1) / bin_size_h);
            int pwstart = floor((float)(w - roi_start_w) / bin_size_w);
            int pwend = ceil((float)(w - roi_start_w + 1) / bin_size_w);

            plstart = fminf(fmaxf(plstart, 0), pooled_length);
            plend = fminf(fmaxf(plend, 0), pooled_length);
            phstart = fminf(fmaxf(phstart, 0), pooled_height);
            phend = fminf(fmaxf(phend, 0), pooled_height);
            pwstart = fminf(fmaxf(pwstart, 0), pooled_width);
            pwend = fminf(fmaxf(pwend, 0), pooled_width);

            for (int pl = plstart; pl < plend; ++pl) {
                for (int ph = phstart; ph < phend; ++ph) {
                    for (int pw = pwstart; pw < pwend; ++pw) {
                        int top_index = (( c * pooled_length + pl ) * pooled_height + ph ) * pooled_width + pw;
                        if (offset_argmax_data[top_index] == index)
                        {
                            gradient += offset_top_diff[top_index];
                        }
                    }
                }
            }
        }
        bottom_diff[index] = gradient;
  }
}

int ROITemporalPoolBackwardLaucher(const float* top_diff, const float temporal_scale, const float ctx_ratio, const int batch_size, const int num_rois,
    const int length, const int height, const int width, const int channels, const int pooled_length,
    const int pooled_height, const int pooled_width, const float* bottom_rois,
    float* bottom_diff, const int* argmax_data, cudaStream_t stream)
{
    const int kThreadsPerBlock = 1024;
    int output_size = batch_size * length * height * width * channels;
    cudaError_t err;

    ROITemporalPoolBackward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
      output_size, top_diff, argmax_data, num_rois, temporal_scale, ctx_ratio, length, height, width, channels, pooled_length, pooled_height,
      pooled_width, bottom_diff, bottom_rois);

    // dim3 blocks(DIVUP(output_size, kThreadsPerBlock),
    //             DIVUP(output_size, kThreadsPerBlock));
    // dim3 threads(kThreadsPerBlock);
    //
    // ROITemporalPoolBackward<<<blocks, threads, 0, stream>>>(
    //   output_size, top_diff, argmax_data, num_rois, temporal_scale, height, width, channels, pooled_height,
    //   pooled_width, bottom_diff, bottom_rois);

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}


// #ifdef __cplusplus
// }
// #endif
