import config
from model import faster_rcnn
from training import Trainer
from argparse import ArgumentParser

from utils import visualize, read_image
from dataset.pascal_voc import VOC_CLASSES


def main():
    parser = ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--image_path", type=str)
    parser.add_argument("--score_thresh", type=float, default=0.5)

    args = parser.parse_args()

    model = faster_rcnn(config.model)

    trainer = Trainer(
        config.optimizer,
        model_checkpoint=args.ckpt_dir,
    )

    model = trainer.restore_model(model)

    img = read_image(args.image_path)

    pred_boxes, pred_labels, pred_scores = model.predict(img, config.NUM_CLASSES)

    visualize(img, pred_boxes, pred_labels, pred_scores, VOC_CLASSES, args.score_thresh)


if __name__ == "__main__":
    main()
