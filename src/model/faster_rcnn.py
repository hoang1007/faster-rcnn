import torch
from torch import nn
from torchvision import ops

from .modules import RPNLayer, RCNN
from utils import bbox_transform_inv, clip_boxes


class FasterRCNN(nn.Module):
    def __init__(self, backbone: nn.Module, rpn: RPNLayer, rcnn: RCNN):
        super().__init__()

        self.set_val_mode()

        self.backbone = backbone
        self.rpn = rpn
        self.rcnn = rcnn

    def backbone_forward(self, img: torch.Tensor):
        assert len(img.shape) == 3, "Only support 1 image per batch"

        batched_img = img.unsqueeze(0)

        feature_map = self.backbone(batched_img)
        h, w = img.shape[-2:]
        scale = round(w / feature_map.size(-1))

        im_info = (h, w, 1 / scale)

        return feature_map, im_info

    def forward(self, img):
        feature_map, im_info = self.backbone_forward(img)

        rpn_bbox_pred, rpn_cls_scores, rpn_rois, anchors = self.rpn(
            feature_map, im_info
        )
        roi_bbox_pred, roi_cls_scores = self.rcnn(feature_map, rpn_rois)

        return {
            "rpn_bbox_pred": rpn_bbox_pred,
            "rpn_cls_scores": rpn_cls_scores,
            "roi_bbox_pred": roi_bbox_pred,
            "roi_cls_scores": roi_cls_scores,
            "rpn_rois": rpn_rois,
            "anchors": anchors,
        }

    def set_val_mode(self, mode: str = None):
        """
        mode: `EVAL` or `VISUALIZE`
        """

        if mode is None:
            self._val_mode = "EVAL"
        elif mode in ("EVAL", "VISUALIZE"):
            self._val_mode = mode
        else:
            raise ValueError("Invalid val mode")

    def _get_preset(self):
        if self._val_mode == "EVAL":
            return {"nms_thresh": 0.3, "score_thresh": 0.05}
        elif self._val_mode == "VISUALIZE":
            return {"nms_thresh": 0.3, "score_thresh": 0.5}
        else:
            raise ValueError("Invalid val mode")

    def _suppress(self, pred_boxes, box_scores, num_classes):
        preset = self._get_preset()
        boxlist = []
        labellist = []
        scorelist = []

        pred_boxes = pred_boxes.reshape(-1, num_classes, 4)

        for clazz in range(1, num_classes):  # ignore background
            boxes_ = pred_boxes[:, clazz]
            scores_ = box_scores[:, clazz]

            mask = scores_ > preset["score_thresh"]
            boxes_ = boxes_[mask]
            scores_ = scores_[mask]

            keep = ops.nms(boxes_, scores_, preset["nms_thresh"])

            boxes_ = boxes_[keep]
            scores_ = scores_[keep]
            labels_ = clazz * pred_boxes.new_ones(boxes_.size(0), dtype=torch.long)

            boxlist.append(boxes_)
            labellist.append(labels_)
            scorelist.append(scores_)

        return (
            torch.cat(boxlist, dim=0),
            torch.cat(labellist, dim=0),
            torch.cat(scorelist, dim=0),
        )

    def predict(
        self,
        img,
        num_classes,
        delta_means=(0, 0, 0, 0),
        delta_stds=(0.1, 0.1, 0.2, 0.2),
    ):

        is_training = False
        if self.training:
            self.train(False)
            is_training = True

        delta_means = (
            torch.tensor(delta_means).view(1, -1).to(img.device).repeat(1, num_classes)
        )
        delta_stds = (
            torch.tensor(delta_stds).view(1, -1).to(img.device).repeat(1, num_classes)
        )

        with torch.no_grad():
            outputs = self(img)

            roi_bbox_pred = outputs["roi_bbox_pred"] * delta_stds + delta_means
            roi_bbox_pred = roi_bbox_pred.view(-1, num_classes, 4)

            rois = outputs["rpn_rois"].view(-1, 1, 4).expand_as(roi_bbox_pred)

            pred_boxes = bbox_transform_inv(
                rois.reshape(-1, 4), roi_bbox_pred.reshape(-1, 4)
            )

            img_height, img_width = img.shape[-2:]
            pred_boxes = clip_boxes(pred_boxes, img_height, img_width)

            pred_boxes = pred_boxes.view(-1, num_classes * 4)

            box_probs = torch.softmax(outputs["roi_cls_scores"], dim=-1)

            pred_boxes, pred_labels, box_scores = self._suppress(
                pred_boxes, box_probs, num_classes
            )

        if is_training:
            self.train(True)  # restore train mode
        return pred_boxes, pred_labels, box_scores

    def init_weight(self):
        print("Weights are initialized")
        self.rpn.init_weight()
        self.rcnn.init_weight()
