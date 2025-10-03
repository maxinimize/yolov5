import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict

from utils.attacks.attacker import Attacker
from utils.loss import ComputeLoss
from utils.general import non_max_suppression

# ART
try:
    # newer versions of ART using the generic object detector
    from art.estimators.object_detection import PyTorchObjectDetector as ARTDetector
except Exception:
    # older versions of ART
    from art.estimators.object_detection import PyTorchYolo as ARTDetector

from art.attacks.evasion import ProjectedGradientDescent


class YoloV5ForART(nn.Module):
    """
    wrap YOLOv5 to fit ART's (images, targets) -> losses interface.
    """
    def __init__(self, yolo_model: nn.Module, img_size: int):
        super().__init__()
        self.model = yolo_model
        self.img_size = img_size
        self.compute_loss = ComputeLoss(self.model)

    @torch.enable_grad()
    def forward(self, images: torch.Tensor, targets=None):
        """
        images: (N,C,H,W) float in [0,1]
        targets: list[dict] with keys 'boxes'(xyxy, pixel), 'labels' (int64)
        return: if targets is None, return torchvision style outputs;
        """
        # YOLOv5 forward
        preds = self.model(images, augment=False)

        if targets is None:
            # if targets is None, return torchvision style outputs;
            dets = non_max_suppression(preds, conf_thres=0.001, iou_thres=0.6, max_det=300)
            out = []
            for det in dets:
                if det is None or len(det) == 0:
                    out.append({"boxes": torch.zeros((0, 4), device=images.device),
                                "labels": torch.zeros((0,), dtype=torch.int64, device=images.device),
                                "scores": torch.zeros((0,), device=images.device)})
                else:
                    boxes = det[:, :4]          # xyxy
                    scores = det[:, 4]
                    labels = det[:, 5].long()
                    out.append({"boxes": boxes, "labels": labels, "scores": scores})
            return out

        # ART → YOLOv5：list[dict](xyxy)  ->  Tensor(N,6)  xywh
        n, _, H, W = images.shape
        yolo_tgts = []
        for i, tgt in enumerate(targets):
            if tgt["boxes"] is None or len(tgt["boxes"]) == 0:
                continue
            b = torch.as_tensor(tgt["boxes"], device=images.device, dtype=torch.float32)  # (K,4) xyxy
            c = torch.as_tensor(tgt["labels"], device=images.device, dtype=torch.int64)   # (K,)
            # xyxy -> xywh (pixel)
            xy = (b[:, 0:2] + b[:, 2:4]) * 0.5
            wh = (b[:, 2:4] - b[:, 0:2]).clamp(min=1e-6)
            # -> [0,1]
            xy[:, 0] /= W; xy[:, 1] /= H
            wh[:, 0] /= W; wh[:, 1] /= H
            # -> [img_idx, cls, x, y, w, h]
            img_idx = torch.full((b.size(0), 1), i, device=images.device, dtype=torch.float32)
            cls = c.to(dtype=torch.float32).unsqueeze(1)
            yolo_tgts.append(torch.cat([img_idx, cls, xy, wh], dim=1))

        if len(yolo_tgts) == 0:
            yolo_targets = images.new_zeros((0, 6))
        else:
            yolo_targets = torch.cat(yolo_tgts, dim=0)

        # compute loss
        total_loss, loss_items = self.compute_loss(preds, yolo_targets)
        # only return total_loss with gradient (loss_items are detached, cannot be used for backprop)
        return {"loss": total_loss}


class ARTPGD(Attacker):
    """
    fit attacker.forward(x, targets)
      epsilon -> PGD.eps
      lr      -> PGD.eps_step
      epoch   -> PGD.max_iter
    """
    def __init__(self, model, config=None, target=None,
                 epsilon=0.05, lr=0.005, epoch=20, img_size=640):
        super().__init__(model, config, epsilon)
        self.device = next(model.parameters()).device

        # might not be necessary as batchnorm/dropout will be disabled in ART wrapper later
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # wrap yolo model for ART
        wrapped = YoloV5ForART(self.model, img_size)

        # ART Detector
        self.estimator = ARTDetector(
            model=wrapped,
            input_shape=(3, img_size, img_size),
            clip_values=(0.0, 1.0),
            channels_first=True,
            device_type="gpu",
            attack_losses=("loss",),
        )
        # disable batchnorm/dropout
        if hasattr(self.estimator, "set_batchnorm"):
            self.estimator.set_batchnorm(False)
        if hasattr(self.estimator, "set_dropout"):
            self.estimator.set_dropout(False)
 
        # PGD
        self.attack = ProjectedGradientDescent(
            estimator=self.estimator,
            norm=np.inf,
            eps=epsilon,
            eps_step=lr,
            max_iter=epoch,
            targeted=False,
            num_random_init=0,
            verbose=False,
        )

    def forward(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        x: (N,C,H,W) in [0,1] float
        targets: (M,6) [img_idx, cls, x, y, w, h] 归一化 xywh
        """
        n, _, h, w = x.shape
        y_art = self._to_art_labels(targets, h, w, n)

        x_np = x.detach().cpu().numpy()
        x_adv_np = self.attack.generate(x=x_np, y=y_art)
        x_adv = torch.from_numpy(x_adv_np).to(self.device).type_as(x)
        return x_adv

    @staticmethod
    def _to_art_labels(targets: torch.Tensor, H: int, W: int, batch_size: int) -> List[Dict[str, np.ndarray]]:
        out: List[Dict[str, np.ndarray]] = []
        t = targets.detach().cpu()
        for i in range(batch_size):
            ti = t[t[:, 0] == i]
            if ti.numel() == 0:
                out.append({"boxes": np.zeros((0, 4), dtype=np.float32),
                            "labels": np.zeros((0,), dtype=np.int64)})
                continue
            # xywh -> xyxy
            xywh = ti[:, 2:6].clone()
            xywh[:, 0] *= W; xywh[:, 1] *= H
            xywh[:, 2] *= W; xywh[:, 3] *= H
            xyxy = torch.zeros_like(xywh)
            xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
            xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
            xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
            xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2

            labels = ti[:, 1].to(torch.int64)
            out.append({
                "boxes": xyxy.numpy().astype(np.float32),
                "labels": labels.numpy().astype(np.int64),
            })
        return out
