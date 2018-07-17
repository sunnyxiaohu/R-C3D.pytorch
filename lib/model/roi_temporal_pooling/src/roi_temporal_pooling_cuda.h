int roi_temporal_pooling_forward_cuda(int pooled_length, int pooled_height, int pooled_width, float temporal_scale, float ctx_ratio,
                        THCudaTensor * features, THCudaTensor * rois, THCudaTensor * output, THCudaIntTensor * argmax);

int roi_temporal_pooling_backward_cuda(int pooled_length, int pooled_height, int pooled_width, float temporal_scale, float ctx_ratio,
                        THCudaTensor * top_grad, THCudaTensor * rois, THCudaTensor * bottom_grad, THCudaIntTensor * argmax);
