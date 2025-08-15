"""
Combine network weights into a single weight file. 
"""

import os
import re
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm 
from argparse import ArgumentParser

from mobileposer.models_new.net import PoseNet
from mobileposer.models_new.joints import Joints
from mobileposer.models_new.poser import Poser
from mobileposer.models_new.velocity import Velocity

from mobileposer.constants import MODULES
from mobileposer.utils.file_utils import get_best_checkpoint


def load_module_weights(module_name, weight_path):
    try:
        if module_name == "joints":
            return Joints.load_from_checkpoint(weight_path, winit=True)
        elif module_name == "poser":
            return Poser.load_from_checkpoint(weight_path)
        elif module_name == "velocity":
            return Velocity.load_from_checkpoint(weight_path)
        else:
            raise ValueError(f"Unknown module: {module_name}")
    except Exception as e:
        print(f"Error loading {module_name} weights from {weight_path}: {e}")
        return None


def get_module_path(module_name, checkpoint, finetune=None, name=None, combo_id=None):
    module_path = Path("data/checkpoints") / name / combo_id
    if args.finetune and module_name in ["poser", "joints"]:
        module_path = module_path / f"finetuned_{finetune}" / module_name
    else:
        module_path = module_path / module_name
    return module_path


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--weights", nargs="+", help="List of weight paths.")
    parser.add_argument("--finetune", type=str, default=None)
    parser.add_argument("--checkpoint", type=int, help="Checkpoint number.", default=1)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--combo_id", type=str, default='lw_rp_h')
    args = parser.parse_args()

    checkpoints = {}
    for module_name in MODULES.keys():
        # 如果module_name非["poser", "joints"]
        if module_name not in ["poser", "joints", "velocity"]:
            continue
        module_path = get_module_path(module_name, args.checkpoint, args.finetune, args.name, args.combo_id)
        best_ckpt = get_best_checkpoint(module_path)
        if best_ckpt:
            checkpoints[module_name] = load_module_weights(module_name, module_path / best_ckpt)
            print(f"Module: {module_name.ljust(15)} | Best Checkpoint: {best_ckpt}")
        else:
            print(f"No checkpoint found for {module_name} in {module_path}")

    # load combined model and save
    model_name = "base_model.pth" if not args.finetune else "model_finetuned.pth"
    model = PoseNet(**checkpoints)
    model_path = Path("data/checkpoints") / args.name / args.combo_id / model_name
    torch.save(model.state_dict(), model_path)
    print(f"Model written to {model_path}.")
