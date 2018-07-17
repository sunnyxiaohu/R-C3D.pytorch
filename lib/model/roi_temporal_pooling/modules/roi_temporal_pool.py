from torch.nn.modules.module import Module
from ..functions.roi_temporal_pool import RoITemporalPoolFunction


class _RoITemporalPooling(Module):
    def __init__(self, pooled_length, pooled_height, pooled_width, temporal_scale, ctx_ratio=1.0):
        super(_RoITemporalPooling, self).__init__()

        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.pooled_length = int(pooled_length)
        self.temporal_scale = float(temporal_scale)
        self.ctx_ratio = float(ctx_ratio)

    def forward(self, features, rois):
        return RoITemporalPoolFunction(self.pooled_length, self.pooled_height, self.pooled_width, self.temporal_scale, self.ctx_ratio)(features, rois)
