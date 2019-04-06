#include <TH/TH.h>
#include <math.h>

int roi_temporal_pooling_forward(int pooled_length, int pooled_height, int pooled_width, float temporal_scale, float ctx_ratio,
                        THFloatTensor * features, THFloatTensor * rois, THFloatTensor * output)
{
    // Grab the input tensor
    float * data_flat = THFloatTensor_data(features);
    float * rois_flat = THFloatTensor_data(rois);

    float * output_flat = THFloatTensor_data(output);

    // Number of ROIs
    int num_rois = THFloatTensor_size(rois, 0);
    int size_rois = THFloatTensor_size(rois, 1);
    // batch size
    int batch_size = THFloatTensor_size(features, 0);
    if(batch_size != 1)
    {
        return 0;
    }
    if (size_rois != 3)
    {
        return 0;
    }
    // data length
    int data_length = THFloatTensor_size(features, 1);
    // data height
    int data_height = THFloatTensor_size(features, 2);
    // data width
    int data_width = THFloatTensor_size(features, 3);
    // Number of channels
    int num_channels = THFloatTensor_size(features, 4);

    // Set all element of the output tensor to -inf.
    THFloatStorage_fill(THFloatTensor_storage(output), -1);

    // Note that RoI just has temporal border
    int roi_height = data_height;
    int roi_width = data_width;

    // For each ROI R = [batch_index x1 x2]: max pool over R
    int index_roi = 0;
    int index_output = 0;
    int n;
    for (n = 0; n < num_rois; ++n)
    {
        int roi_batch_ind = rois_flat[index_roi + 0];
        int roi_start_l = round(rois_flat[index_roi + 1] * temporal_scale);
        int roi_end_l = round(rois_flat[index_roi + 2] * temporal_scale);
        int roi_start_h = 0;
        // int roi_end_h = roi_height -1;
        int roi_start_w = 0;
        // int roi_end_w = roi_height -1;
        //      CHECK_GE(roi_batch_ind, 0);
        //      CHECK_LT(roi_batch_ind, batch_size);
       
        // Add context for temporal segment
        int roi_length_tmp = fmaxf(roi_end_l - roi_start_l + 1, 1);
        roi_start_l = roi_start_l - 0.5*(ctx_ratio-1)*roi_length_tmp;
        roi_end_l = roi_end_l + 0.5*(ctx_ratio-1)*roi_length_tmp;

        int roi_length = fmaxf(roi_end_l - roi_start_l + 1, 1);
        float bin_size_l = (float)(roi_length) / (float)(pooled_length);
        float bin_size_h = (float)(roi_height) / (float)(pooled_height);
        float bin_size_w = (float)(roi_width) / (float)(pooled_width);

        int index_data = roi_batch_ind * data_length * data_height * data_width * num_channels;
        const int output_area = pooled_width * pooled_height * pooled_length;

        int c, pl, ph, pw;
        for (pl = 0; pl < pooled_length; ++pl)
        {
            for (ph = 0; ph < pooled_height; ++ph)
            {
                for (pw = 0; pw < pooled_width; ++pw)
                {
                    int lstart = (floor((float)(pl) * bin_size_l));
                    int hstart = (floor((float)(ph) * bin_size_h));
                    int wstart = (floor((float)(pw) * bin_size_w));
                    int lend = (ceil((float)(pl + 1) * bin_size_l));
                    int hend = (ceil((float)(ph + 1) * bin_size_h));
                    int wend = (ceil((float)(pw + 1) * bin_size_w));

                    lstart = fminf(fmaxf(lstart + roi_start_l, 0), data_length);
                    lend = fminf(fmaxf(lend + roi_start_l, 0), data_length);
                    hstart = fminf(fmaxf(hstart + roi_start_h, 0), data_height);
                    hend = fminf(fmaxf(hend + roi_start_h, 0), data_height);
                    wstart = fminf(fmaxf(wstart + roi_start_w, 0), data_width);
                    wend = fminf(fmaxf(wend + roi_start_w, 0), data_width);

                    const int pool_index = index_output + ((pl * pooled_height + ph) * pooled_width + pw);
                    int is_empty = (lend <= lstart) || (hend <= hstart) || (wend <= wstart);
                    if (is_empty)
                    {
                        for (c = 0; c < num_channels * output_area; c += output_area)
                        {
                            output_flat[pool_index + c] = 0;
                        }
                    }
                    else
                    {
                        int l, h, w, c;
                        for (l = lstart; l < lend; ++l)
                        {
                            for (h = hstart; h < hend; ++h)
                            {
                                for (w = wstart; w < wend; ++w)
                                {
                                    for (c = 0; c < num_channels; ++c)
                                    {
                                        const int index =((l * data_height + h) * data_width + w) * num_channels + c;
                                        if (data_flat[index_data + index] > output_flat[pool_index + c * output_area])
                                        {
                                            output_flat[pool_index + c * output_area] = data_flat[index_data + index];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // Increment ROI index
        index_roi += size_rois;
        index_output += pooled_length * pooled_height * pooled_width * num_channels;
    }
    return 1;
}
