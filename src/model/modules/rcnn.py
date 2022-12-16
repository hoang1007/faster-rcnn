from torch import nn
from torchvision import ops

from utils import init_weight


class RCNN(nn.Module):
    def __init__(self, roi_size, num_channels, n_classes, spatial_scale, dropout=0.2):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(num_channels * roi_size * roi_size, 4096),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.roi_cls = nn.Linear(4096, n_classes)
        self.roi_reg = nn.Linear(4096, n_classes * 4)

        self.roi_align = ops.RoIAlign(
            (roi_size, roi_size), spatial_scale, sampling_ratio=-1
        )
        # self.roi_align = ops.RoIPool(roi_size, spatial_scale=spatial_scale)

    def forward(self, feature_map, rois):
        # feature_map.shape = (1, C, H, W)
        # rois.shape = (N, 4)
        assert len(rois.shape) == 2 and rois.size(1) == 4

        # pooled.shape = (N, C, roi_size, roi_size)
        pooled = self.roi_align(feature_map, [rois])

        pooled = pooled.view(pooled.size(0), -1)
        fc_out = self.fc(pooled)

        roi_bbox_pred = self.roi_reg(fc_out)
        roi_cls_scores = self.roi_cls(fc_out)

        return roi_bbox_pred, roi_cls_scores

    def init_weight(self):
        init_weight(self.roi_cls, std=0.01)
        init_weight(self.roi_reg, std=0.01)

        for model in self.fc:
            try:
                init_weight(model, std=0.01)
            except:
                continue
