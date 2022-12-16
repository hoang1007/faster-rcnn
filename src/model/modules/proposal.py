import torch
from torch import nn
from torchvision import ops

from utils import bbox_transform_inv, clip_boxes


class ProposalLayer(nn.Module):
    def __init__(
        self,
        train_pre_nms_topN,
        train_post_nms_topN,
        test_pre_nms_topN,
        test_post_nms_topN,
        nms_thresh,
        min_box_size,
    ):
        """
        Lọc các loc bằng nms và đưa ra các RoIs từ các loc đã lọc.

        Args:
            rpn_bbox_pred: Dự đoán của box regessors trên từng anchor. Shape (N, 4)
            fg_probs: Xác suất để box là foreground. Shape (N, 1)
            anchors: Các mỏ neo trên ảnh. Shape (N, 4)
            im_info: Đại diện cho (H, W, scale) của ảnh

        Returns:
            rois: RoIs được đề xuất của ảnh
        """
        super().__init__()

        self._pre_nms_topN = {"train": train_pre_nms_topN, "test": test_pre_nms_topN}
        self._post_nms_topN = {"train": train_post_nms_topN, "test": test_post_nms_topN}
        self.nms_thresh = nms_thresh
        self.min_box_size = min_box_size

    def _get_nms_topN(self):
        key = "train" if self.training else "test"

        return self._pre_nms_topN[key], self._post_nms_topN[key]

    def forward(self, rpn_bbox_pred, fg_probs, anchors, im_info):
        if rpn_bbox_pred.requires_grad:
            rpn_bbox_pred = rpn_bbox_pred.detach()

        pre_nms_topN, post_nms_topN = self._get_nms_topN()

        proposals = bbox_transform_inv(anchors, rpn_bbox_pred)
        proposals = clip_boxes(proposals, im_info[0], im_info[1])

        keep = self._filter_boxes(proposals, self.min_box_size * im_info[2])
        proposals = proposals[keep]
        fg_probs = fg_probs[keep]

        fg_probs, order = torch.sort(fg_probs, descending=True)

        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
            fg_probs = fg_probs[:pre_nms_topN]

        proposals = proposals[order]

        nms_keep_ids = ops.nms(proposals, fg_probs, self.nms_thresh)

        if post_nms_topN > 0:
            nms_keep_ids = nms_keep_ids[:post_nms_topN]

        proposals = proposals[nms_keep_ids]

        return proposals

    def _filter_boxes(self, boxes, min_size):
        """
        Remove all boxes with any size smaller than min_size
        Return:
            indices of filltered boxes
        """
        ws = boxes[:, 2] - boxes[:, 0] + 1
        hs = boxes[:, 3] - boxes[:, 1] + 1

        keep = torch.where(torch.logical_and(ws >= min_size, hs >= min_size))[0]

        return keep
