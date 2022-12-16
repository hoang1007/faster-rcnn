import config
from model import faster_rcnn
from dataset import VOCDataset
from training import Trainer, TrainerWrapper
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--restore", type=bool, default=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--use_adam", type=bool, default=False)

    args = parser.parse_args()

    model = faster_rcnn(config.model)
    model_wrapper = TrainerWrapper(
        model, config.NUM_CLASSES, config.anchor_target, config.proposal_target
    )

    train_data = VOCDataset(image_set="train")
    val_data = VOCDataset(image_set="val")

    trainer = Trainer(
        config.optimizer,
        model_checkpoint=args.ckpt_dir,
        device=args.device,
        max_grad_norm=args.grad_clip,
        use_adam=args.use_adam,
        val_test_first=True,
    )

    trainer(
        model_wrapper,
        train_data,
        val_data,
        num_epochs=args.epochs,
        restore=args.restore,
    )

if __name__ == "__main__":
    main()