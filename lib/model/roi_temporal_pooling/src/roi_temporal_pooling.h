int roi_temporal_pooling_forward(int pooled_length, int pooled_height, int pooled_width, float temporal_scale, float ctx_ratio,
                        THFloatTensor * features, THFloatTensor * rois, THFloatTensor * output);
