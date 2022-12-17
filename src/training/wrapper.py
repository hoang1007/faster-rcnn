import torch
import torch.nn.functional as F

from model import FasterRCNN
from model.modules import AnchorTargetGenerator, ProposalTargetGenerator


class TrainerWrapper:
    def __init__(
        self,
        model: FasterRCNN,
        num_classes: int,
        anchor_target_args: dict,
        proposal_target_args: dict,
    ):
        self._model = model

        self.num_classes = num_classes

        self.anchor_target_layer = AnchorTargetGenerator(**anchor_target_args)
        self.proposal_target_layer = ProposalTargetGenerator(
            num_classes=num_classes, **proposal_target_args
        )

    def __call__(
        self, img: torch.Tensor, gt_boxes: torch.Tensor, gt_labels: torch.Tensor
    ):
        feature_map, im_info = self._model.backbone_forward(img)

        rpn_bbox_pred, rpn_cls_scores, rpn_rois, anchors = self._model.rpn(
            feature_map, im_info
        )

        rpn_bbox_targets, rpn_labels = self.anchor_target_layer(
            anchors, gt_boxes, im_info
        )
        roi_bbox_targets, sampled_rois, roi_labels = self.proposal_target_layer(
            rpn_rois, gt_boxes, gt_labels
        )

        roi_bbox_pred, roi_cls_scores = self._model.rcnn(feature_map, sampled_rois)

        rpn_cls_loss, rpn_reg_loss = self._compute_loss(
            rpn_bbox_pred, rpn_cls_scores, rpn_bbox_targets, rpn_labels
        )

        roi_cls_loss, roi_reg_loss = self._compute_loss(
            roi_bbox_pred, roi_cls_scores, roi_bbox_targets, roi_labels
        )

        return {
            "rpn_cls_loss": rpn_cls_loss,
            "rpn_reg_loss": rpn_reg_loss,
            "roi_cls_loss": roi_cls_loss,
            "roi_reg_loss": roi_reg_loss,
        }

    def _compute_loss(self, bbox_pred, cls_scores, bbox_targets, labels, beta=10):
        cls_loss = F.cross_entropy(cls_scores, labels, ignore_index=-1)

        fg_mask = labels > 0
        sampled_mask = labels >= 0

        reg_loss = F.smooth_l1_loss(
            bbox_pred[fg_mask], bbox_targets[fg_mask], beta=1, reduction="sum"
        )

        reg_loss = beta * reg_loss / sampled_mask.sum()

        return cls_loss, reg_loss

    def train_mode(self, enable=True):
        self._model.train(mode=enable)

    def to(self, device):
        self._model.to(device)

    @property
    def model(self):
        return self._model

    def load_state_dict(self, state_dict):
        self._model.load_state_dict(state_dict)

    def get_optimizers(
        self,
        lr,
        weight_decay,
        bias_decay,
        double_bias,
        momentum,
        lr_decay,
        monitor_mode,
        patience,
        use_adam=False,
    ):
        params = []

        for key, val in self.model.named_parameters():
            if val.requires_grad:
                if "bias" in key:
                    params.append(
                        {
                            "params": [val],
                            "lr": lr * (double_bias + 1),
                            "weight_decay": bias_decay and weight_decay or 0,
                        }
                    )
                else:
                    params.append(
                        {
                            "params": [val],
                            "lr": lr,
                            "weight_decay": getattr(val, "weight_decay", weight_decay),
                        }
                    )

        if use_adam:
            optimizer = torch.optim.Adam(params)
        else:
            optimizer = torch.optim.SGD(params, momentum=momentum)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=monitor_mode,
            factor=lr_decay,
            patience=patience,
            min_lr=1e-7
        )

        return optimizer, scheduler
