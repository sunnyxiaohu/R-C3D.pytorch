from torch.nn.modules.module import Module
from ..functions.roi_temporal_pool import roi_temporal_pool


class RoITemporalPool(Module):
    def __init__(self, pooled_size, temporal_scale):
        super(RoITemporalPool, self).__init__()

        self.pooled_size = pooled_size
        self.temporal_scale = float(temporal_scale)

    def forward(self, features, rois):
        return roi_temporal_pool(features, rois, self.pooled_size, self.temporal_scale)
