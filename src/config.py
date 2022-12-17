NUM_CLASSES = 21

model = {
    "backbone": "resnet50",
    "num_classes": NUM_CLASSES,
    "proposal": {
        "train_pre_nms_topN": 12000,
        "test_pre_nms_topN": 6000,
        "train_post_nms_topN": 2000,
        "test_post_nms_topN": 300,
        "nms_thresh": 0.7,
        "min_box_size": 16,
    },
    "anchor_gen": {"scales": (8, 16, 32), "aspect_ratios": (0.5, 1, 2)},
    "rcnn": {"roi_size": 7},
}

anchor_target = {
    "batch_size": 128,
    "allowed_border": 10,
    "clobber_positives": False,
    "positive_overlap": 0.7,
    "negative_overlap": 0.3,
    "fg_fraction": 0.5,
}

proposal_target = {
    "use_gt": True,
    "batch_size": 128,
    "fg_fraction": 0.25,
    "fg_thresh": 0.5,
    "bg_thresh_high": 0.5,
    "bg_thresh_low": 0.1,
    "bbox_normalize_means": (0.0, 0.0, 0.0, 0.0),
    "bbox_normalize_stds": (0.1, 0.1, 0.2, 0.2),
}

optimizer = {
    "double_bias": True,
    "bias_decay": False,
    "weight_decay": 0.0001,
    "momentum": 0.9,
    "lr": 0.001,
    "lr_gamma": 0.1,
    "lr_decay": 0.1,
    "monitor_mode": "max",
    "patience": 3,
}
