#ifndef _ROI_TEMPORAL_POOLING_KERNEL
#define _ROI_TEMPORAL_POOLING_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

int ROITemporalPoolForwardLaucher(
    const float* bottom_data, const float temporal_scale, const float ctx_ratio, const int num_rois, const int length, const int height,
    const int width, const int channels, const int pooled_length, const int pooled_height,
    const int pooled_width, const float* bottom_rois,
    float* top_data, int* argmax_data, cudaStream_t stream);


int ROITemporalPoolBackwardLaucher(
    const float* top_diff, const float temporal_scale, const float ctx_ratio, const int batch_size, const int num_rois,
    const int length, const int height, const int width, const int channels, const int pooled_length, const int pooled_height,
    const int pooled_width, const float* bottom_rois,
    float* bottom_diff, const int* argmax_data, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif

