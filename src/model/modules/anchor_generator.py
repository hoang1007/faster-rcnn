import torch
from torch import nn

from utils import box_info_to_boxes


class AnchorGenerator(nn.Module):
    def __init__(self, feat_stride: int, scales: list, aspect_ratios=[0.5, 1, 2]):
        """
        Tạo ra các anchor trên ảnh

        Args:
            feat_stride: Khoảng cách giữa tâm của hai anchor liền kề
            scales: Tỉ lệ kích thước so với base anchor
            aspect_ratios: Tỉ lệ height / width của anchors

        Forward Args:
            feature_map (Tensor): Feature map của ảnh sau khi đưa qua backbone. Shape (1, C, H, W)
        """
        super().__init__()

        if not isinstance(scales, torch.Tensor):
            scales = torch.tensor(scales)
        if not isinstance(aspect_ratios, torch.Tensor):
            aspect_ratios = torch.tensor(aspect_ratios)

        self.feat_stride = feat_stride
        self.scales = scales
        self.aspect_ratios = aspect_ratios

        self.register_buffer("base_anchors", self.mkbase_anchors())

    @property
    def num_anchors_per_location(self):
        return self.scales.numel() * self.aspect_ratios.numel()

    def mkbase_anchors(self):
        A = self.num_anchors_per_location

        base_area = self.feat_stride**2

        hs = torch.sqrt(base_area * self.aspect_ratios)  # shape (num_aspect_ratios,)
        ws = base_area / hs  # shape (num_aspect_ratios,)

        scales = self.scales.reshape(1, -1)
        hs = (hs.reshape(-1, 1) * scales).flatten()
        ws = (ws.reshape(-1, 1) * scales).flatten()

        x_ctrs = torch.ones(A) * (self.feat_stride - 1) / 2
        y_ctrs = x_ctrs.clone()

        base_anchors = box_info_to_boxes(x_ctrs, y_ctrs, ws, hs)

        return base_anchors.round()

    def forward(self, feature_map: torch.Tensor):
        feat_height, feat_width = feature_map.shape[-2:]

        shiftx = (
            torch.arange(0, feat_width, device=feature_map.device) * self.feat_stride
        )
        shifty = (
            torch.arange(0, feat_height, device=feature_map.device) * self.feat_stride
        )

        shiftx, shifty = torch.meshgrid(shiftx, shifty, indexing="ij")

        shifts = torch.vstack(
            (shiftx.ravel(), shifty.ravel(), shiftx.ravel(), shifty.ravel())
        ).transpose(0, 1)

        # shifts.shape == (H * W, 4)
        # base_anchors.shape == (A, 4)
        # => anchors.shape == (H * W * A, 4)
        anchors = self.base_anchors.view(1, -1, 4) + shifts.view(-1, 1, 4)
        anchors = anchors.contiguous().view(-1, 4)

        return anchors
