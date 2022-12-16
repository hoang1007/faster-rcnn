import os
import pickle
import torch
from torch import nn
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics import MeanMetric
from tqdm import tqdm
from statistics import mean
import wandb

from .wrapper import TrainerWrapper
from model import FasterRCNN


class Trainer:
    def __init__(
        self,
        optimizer_config: dict,
        model_checkpoint=None,
        device="cpu",
        use_adam=False,
        max_grad_norm=None,
        disp_interval=50,
        val_test_first=True,
    ):
        self.optimizer_config = optimizer_config
        self.model_checkpoint = model_checkpoint
        self.device = device
        self.log_interval = disp_interval
        self.use_adam = use_adam
        self.val_test_first = val_test_first
        self.max_grad_norm = max_grad_norm

        self.mAP = MeanAveragePrecision()

        self.losses = {
            "rpn_cls_loss": MeanMetric(),
            "rpn_reg_loss": MeanMetric(),
            "roi_cls_loss": MeanMetric(),
            "roi_reg_loss": MeanMetric(),
            "total_loss": MeanMetric(),
        }

    def __call__(
        self, wrapper: TrainerWrapper, train_data, val_data, num_epochs, restore=True
    ):
        optimizer, scheduler, start_epoch = self._load_params(wrapper, restore)
        wandb.init(project="Faster RCNN", resume=restore)

        if self.val_test_first:
            mAP = self.validation_epoch(wrapper, val_data, max_batches=5)
            self.mAP.reset()
            print("mAP on first test run:", mAP)

        for epoch in range(start_epoch, num_epochs + 1):
            self.training_epoch(wrapper, train_data, epoch, optimizer, scheduler)

            wandb.log({name: loss.compute() for name, loss in self.losses.items()}, step=epoch)

            mAP = self.validation_epoch(wrapper, val_data)
            wandb.log({"mAP": mAP}, step=epoch)
            print("mAP:", mAP)

            self.backup(wrapper.model, optimizer, scheduler, epoch)

            for loss in self.losses.values():
                loss.reset()
            self.mAP.reset()

    def training_epoch(
        self, wrapper: TrainerWrapper, train_data, epoch, optimizer, scheduler
    ):
        wrapper.train_mode(True)
        with tqdm(train_data) as train_bar:
            train_bar.set_description(f"Epoch {epoch}")
            for batch_idx, batch in enumerate(train_bar):
                losses = self.training_step(wrapper, batch, batch_idx)

                loss = sum(list(losses.values()))

                optimizer.zero_grad()
                loss.backward()

                if self.max_grad_norm:
                    nn.utils.clip_grad_norm_(
                        wrapper.model.parameters(), self.max_grad_norm
                    )
                optimizer.step()
        scheduler.step()

    @torch.no_grad()
    def validation_epoch(self, wrapper: TrainerWrapper, val_data, max_batches=None):
        wrapper.train_mode(False)
        wrapper.model.set_val_mode("EVAL")

        for batch_idx, batch in enumerate(val_data):
            self.validation_step(wrapper, batch, batch_idx)

            if max_batches is not None and batch_idx > max_batches:
                break

        mAP = self.mAP.compute()["map"]
        return mAP

    def training_step(self, wrapper: TrainerWrapper, batch, batch_idx):
        assert wrapper.model.training

        img, gt_bboxes, labels = batch

        img = img.to(self.device)
        gt_bboxes = gt_bboxes.to(self.device)
        labels = labels.to(self.device)

        losses = wrapper(img, gt_bboxes, labels)

        return losses

    def validation_step(self, wrapper: TrainerWrapper, batch, batch_idx):
        img, gt_boxes, labels = batch

        img = img.to(self.device)
        gt_boxes = gt_boxes.to(self.device)
        labels = labels.to(self.device)

        pred_boxes, pred_labels, box_scores = wrapper.model.predict(
            img, wrapper.num_classes
        )

        preds = [
            dict(
                boxes=pred_boxes,
                scores=box_scores,
                labels=pred_labels,
            )
        ]

        targets = [
            dict(
                boxes=gt_boxes,
                labels=labels,
            )
        ]

        self.mAP.update(preds, targets)

    def _load_params(self, wrapper: TrainerWrapper, restore):
        wrapper.to(self.device)

        if (
            restore
            and os.path.isdir(self.model_checkpoint)
            and len(os.listdir(self.model_checkpoint)) != 0
        ):
            model_state, optimizer_state, scheduler_state, start_epoch = self.restore()

            wrapper.load_state_dict(model_state)

            optimizer, scheduler = wrapper.get_optimizers(
                lr=self.optimizer_config.get("learning_rate"),
                weight_decay=self.optimizer_config.get("weight_decay"),
                bias_decay=self.optimizer_config.get("bias_decay"),
                double_bias=self.optimizer_config.get("double_bias"),
                momentum=self.optimizer_config.get("train_momentum"),
                lr_decay=self.optimizer_config.get("lr_decay"),
                step_lr_decay=self.optimizer_config.get("step_lr_decay"),
                use_adam=self.use_adam,
            )

            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)

            start_epoch += 1
        else:
            start_epoch = 1

            wrapper.model.init_weight()
            optimizer, scheduler = wrapper.get_optimizers(
                lr=self.optimizer_config.get("learning_rate"),
                weight_decay=self.optimizer_config.get("weight_decay"),
                bias_decay=self.optimizer_config.get("bias_decay"),
                double_bias=self.optimizer_config.get("double_bias"),
                momentum=self.optimizer_config.get("train_momentum"),
                lr_decay=self.optimizer_config.get("lr_decay"),
                step_lr_decay=self.optimizer_config.get("step_lr_decay"),
                use_adam=self.use_adam,
            )

        return optimizer, scheduler, start_epoch

    def backup(self, model, optimizer, scheduler, epoch):
        ckpt_path = os.path.join(self.model_checkpoint, "checkpoint.pth")

        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "epoch": epoch,
            },
            ckpt_path,
        )

        wandb.save(ckpt_path)

    def restore(self):
        ckpt_path = os.path.join(self.model_checkpoint, "checkpoint.pth")

        checkpoint = torch.load(wandb.restore(ckpt_path), map_location=self.device)

        return (
            checkpoint["model_state"],
            checkpoint["optimizer_state"],
            checkpoint["scheduler_state"],
            checkpoint["epoch"],
        )

    def restore_model(self, model: FasterRCNN):
        ckpt_path = os.path.join(self.model_checkpoint, "checkpoint.pth")

        checkpoint = torch.load(wandb.restore(ckpt_path), map_location=self.device)

        model.load_state_dict(checkpoint["model_state"])

        return model
