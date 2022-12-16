from model.backbones import resnet_backbone
from model.modules import RPNLayer, RCNN
from .faster_rcnn import FasterRCNN


def faster_rcnn(config: dict):
    if config["backbone"] == "resnet50":
        bb_model, roi_head = resnet_backbone(layer=50)
        out_channels = 1024
    elif config["backbone"] == "resnet101":
        bb_model, roi_head = resnet_backbone(layer=101)
        out_channels = 1024

    feat_stride = 16
    rpn = RPNLayer(out_channels, feat_stride, config["proposal"], config["anchor_gen"])
    rcnn = RCNN(
        config["rcnn"]["roi_size"],
        out_channels,
        config["num_classes"],
        spatial_scale=1.0 / feat_stride,
        dropout=0.1,
    )

    model = FasterRCNN(bb_model, rpn, rcnn)

    return model
