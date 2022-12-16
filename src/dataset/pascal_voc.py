import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import VOCDetection
import albumentations as A
from albumentations.pytorch import ToTensorV2

VOC_MEAN = (0.485, 0.456, 0.406)
VOC_STD = (0.229, 0.224, 0.225)


class VOCDataset(Dataset):
    def __init__(self, root="vocdata", year="2007", image_set="trainval"):
        if image_set == "train":
            self.transform = A.Compose(
                (
                    A.HorizontalFlip(p=0.5),
                    A.Normalize(mean=VOC_MEAN, std=VOC_STD),
                    A.RandomBrightnessContrast(p=0.2),
                    ToTensorV2(),
                ),
                bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
            )
        else:
            self.transform = A.Compose(
                (
                    A.Normalize(mean=VOC_MEAN, std=VOC_STD),
                    ToTensorV2(),
                ),
                bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
            )

        self._data = VOCDetection(
            root=root,
            year=year,
            image_set=image_set,
            download=True,
        )

        self.classes = (
            "__background__",  # always index 0
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        )
        self._class2idx = {name: idx for idx, name in enumerate(self.classes)}

    def __getitem__(self, idx):
        img, info = self.data[idx]
        img = np.array(img)

        gt_boxes, labels = [], []

        for obj_info in info["annotation"]["object"]:
            label_name = obj_info["name"]
            bndbox = [int(k) for k in obj_info["bndbox"].values()]

            gt_boxes.append(bndbox)
            labels.append(self._class2idx[label_name])

        transformed = self.transform(image=img, bboxes=gt_boxes, labels=labels)

        img = transformed["image"]
        gt_boxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
        labels = torch.tensor(transformed["labels"], dtype=torch.long)

        return img, gt_boxes, labels

    def __len__(self):
        return len(self._data)
