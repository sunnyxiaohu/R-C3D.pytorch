import torch
from torch.autograd import Function
from .._ext import roi_temporal_pooling
import pdb

class RoITemporalPoolFunction(Function):
    def __init__(ctx, pooled_length, pooled_height, pooled_width, temporal_scale, ctx_ratio):
        ctx.pooled_length = pooled_length
        ctx.pooled_width = pooled_width
        ctx.pooled_height = pooled_height
        ctx.temporal_scale = temporal_scale
        ctx.ctx_ratio = ctx_ratio
        ctx.feature_size = None

    def forward(ctx, features, rois): 
        ctx.feature_size = features.size()           
        batch_size, num_channels, data_length, data_height, data_width = ctx.feature_size
        num_rois = rois.size(0)
        output = features.new(num_rois, num_channels, ctx.pooled_length, ctx.pooled_height, ctx.pooled_width).zero_()
        ctx.argmax = features.new(num_rois, num_channels, ctx.pooled_length, ctx.pooled_height, ctx.pooled_width).zero_().int()
        ctx.rois = rois
        if not features.is_cuda:
            _features = features.permute(0, 2, 3, 4, 1)
            roi_temporal_pooling.roi_temporal_pooling_forward(ctx.pooled_length, ctx.pooled_height, ctx.pooled_width, ctx.temporal_scale,
                                            ctx.ctx_ratio, _features, rois, output)
        else:
            roi_temporal_pooling.roi_temporal_pooling_forward_cuda(ctx.pooled_length, ctx.pooled_height, ctx.pooled_width, ctx.temporal_scale,
                                                 ctx.ctx_ratio, features, rois, output, ctx.argmax)

        return output

    def backward(ctx, grad_output):
        assert(ctx.feature_size is not None and grad_output.is_cuda)
        batch_size, num_channels, data_length, data_height, data_width = ctx.feature_size
        grad_input = grad_output.new(batch_size, num_channels, data_length, data_height, data_width).zero_()

        roi_temporal_pooling.roi_temporal_pooling_backward_cuda(ctx.pooled_length, ctx.pooled_height, ctx.pooled_width, ctx.temporal_scale,
                                              ctx.ctx_ratio, grad_output, ctx.rois, grad_input, ctx.argmax)

        return grad_input, None
