import torch
from torch import nn
from .anchor_generator import AnchorGenerator
from .proposal import ProposalLayer

from utils import init_weight


class RPNLayer(nn.Module):
    def __init__(
        self, in_channels, feat_stride, proposal_layer_params, anchor_generator_params
    ):
        super().__init__()

        self.feat_stride = feat_stride

        self.anchor_generator = AnchorGenerator(feat_stride, **anchor_generator_params)
        self.proposal_layer = ProposalLayer(**proposal_layer_params)

        self.conv = nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1)
        self.classifier = nn.Conv2d(
            512, self.anchor_generator.num_anchors_per_location * 2, kernel_size=1
        )
        self.regressor = nn.Conv2d(
            512, self.anchor_generator.num_anchors_per_location * 4, kernel_size=1
        )

    def init_weight(self):
        init_weight(self.conv, std=0.01)
        init_weight(self.classifier, std=0.01)
        init_weight(self.regressor, std=0.01)

    def forward(self, feature_map, im_info):
        anchors = self.anchor_generator(feature_map)

        fm = torch.relu(self.conv(feature_map))

        cls_scores = self.classifier(fm)  # shape (1, A * 2, H, W)
        bbox_pred = self.regressor(fm)  # shape (1, A * 4, H, W)

        cls_scores = cls_scores.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).contiguous().view(-1, 4)

        cls_probs = torch.softmax(cls_scores.detach(), dim=1)
        fg_probs = cls_probs[:, 1]

        rois = self.proposal_layer(bbox_pred, fg_probs, anchors, im_info)

        return bbox_pred, cls_scores, rois, anchors
