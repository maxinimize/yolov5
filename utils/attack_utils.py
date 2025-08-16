import torch
import yaml
from pathlib import Path
from models.yolo import Model
from ultralytics.utils.patches import torch_load
from utils.general import LOGGER, check_suffix, intersect_dicts
from utils.downloads import attempt_download
from utils.torch_utils import de_parallel
from copy import deepcopy

def setup_attack_model(attack_weights, device, nc, training=False, imgsz=640):
    """
    Set up the attack model based on the provided parameters.
    
    Args:
        attack_weights (str): Path to the attack model weights.
        device (torch.device): Device.
        nc (int): Number of classes.
        training (bool): Whether in training mode.

    Returns:
        attack_model: Configured attack model.
    """
    if attack_weights and attack_weights != "":
        check_suffix(attack_weights, ".pt")
        attack_weights_path = attempt_download(attack_weights) # download if not found locally

        ckpt = torch_load(attack_weights_path, map_location="cpu")  # load checkpoint to CPU to avoid CUDA memory leak

        hyp = ckpt.get("hyp", {})
        cfg = None
        
        attack_model = Model(cfg or ckpt["model"].yaml, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)
        exclude = ["anchor"] if (cfg or hyp.get("anchors")) else []
        csd = ckpt["model"].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, attack_model.state_dict(), exclude=exclude)  # intersect
        attack_model.load_state_dict(csd, strict=False)  # load    
        LOGGER.info(f"Transferred {len(csd)}/{len(attack_model.state_dict())} items for attack model")

        # load default hyperparameters
        ROOT = Path(__file__).parents[1]  # YOLOv5 root directory
        hyp_path = ROOT / "data/hyps/hyp.scratch-low.yaml"
        
        with open(hyp_path, errors="ignore") as f:
            hyp = yaml.safe_load(f)  # load hyps dict
        
        # Model attributes
        nl = de_parallel(attack_model).model[-1].nl  # number of detection layers (to scale hyps)
        hyp["box"] *= 3 / nl  # scale to layers
        hyp["cls"] *= nc / 80 * 3 / nl  # scale to classes and layers
        hyp["obj"] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        attack_model.nc = nc  # attach number of classes to model
        attack_model.hyp = hyp  # attach hyperparameters to model
    else:
        raise ValueError("attack_weights must be provided for setup_attack_model()")
    return attack_model