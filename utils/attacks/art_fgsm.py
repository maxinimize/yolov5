import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict

from utils.attacks.attacker import Attacker
from .art_pgd import YoloV5ForART  # reuse the wrapper

# ART
try:
    from art.estimators.object_detection import PyTorchObjectDetector as ARTDetector
except Exception:
    from art.estimators.object_detection import PyTorchYolo as ARTDetector

from art.attacks.evasion import FastGradientMethod


class ARTFGSM(Attacker):
    """
    FGSM epsilon -> FGSM.eps
    """
    def __init__(self, model, config=None, target=None,
                 epsilon=0.05, img_size=640):
        super().__init__(model, config, epsilon)
        self.device = next(model.parameters()).device

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        wrapped = YoloV5ForART(self.model, img_size)

        self.estimator = ARTDetector(
            model=wrapped,
            input_shape=(3, img_size, img_size),
            clip_values=(0.0, 1.0),
            channels_first=True,
            device_type="gpu",
            attack_losses=("loss",),
        )
        if hasattr(self.estimator, "set_batchnorm"):
            self.estimator.set_batchnorm(False)
        if hasattr(self.estimator, "set_dropout"):
            self.estimator.set_dropout(False)

        self.attack = FastGradientMethod(
            estimator=self.estimator,
            eps=epsilon,
            targeted=False,
            num_random_init=0,
        )

    def forward(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        x: (N,C,H,W) in [0,1]
        targets: (M,6) [img_idx, cls, x, y, w, h] (xywh 归一化)
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