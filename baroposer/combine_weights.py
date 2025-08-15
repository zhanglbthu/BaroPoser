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
from config import *

# from mobileposer.models import MobilePoserNet, Poser, Joints, Velocity, FootContact, Velocity_new
from mobileposer.constants import IMUPOSER, MOBILEPOSER, HEIGHTPOSER
from model.imuposer_local.net import IMUPoserNet
from model.mobileposer.net import MobilePoserNet
from model.heightposer.net import HeightPoserNet
from mobileposer.utils.file_utils import get_file_number, get_best_checkpoint

def load_module_weights(modules, module_name, weight_path):
    try:
        model = modules[module_name](combo_id=model_config.combo_id)
        checkpoint = torch.load(weight_path)
        model.load_state_dict(checkpoint["state_dict"])
        return model
        
    except Exception as e:
        print(f"Error loading {module_name} weights from {weight_path}: {e}")
        return None

def get_module_path(module_name, name=None, combo_id=None):
    
    module_path = Path("data/checkpoints") / name / combo_id / module_name
    
    return module_path


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--weights", nargs="+", help="List of weight paths.")
    parser.add_argument("--finetune", type=str, default=None)
    parser.add_argument("--checkpoint", type=int, help="Checkpoint number.", default=1)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--combo_id", type=str, default=None)
    args = parser.parse_args()

    checkpoints = {}
    
    # 取args.name以_分割，取第一个元素
    model_type = args.name.split("_")[0]
    if model_type == "imuposer":
        modules = IMUPOSER
    elif model_type == "mobileposer":
        modules = MOBILEPOSER
    elif model_type == "heightposer":
        modules = HEIGHTPOSER
    
    for module_name in modules.keys():
        module_path = get_module_path(module_name, args.name, model_config.combo_id)
        best_ckpt = get_best_checkpoint(module_path)
        if best_ckpt:
            checkpoints[module_name] = load_module_weights(modules, module_name, module_path / best_ckpt)
            print(f"Module: {module_name.ljust(15)} | Best Checkpoint: {best_ckpt}")
        else:
            print(f"No checkpoint found for {module_name} in {module_path}")

    # load combined model and save
    model_name = "base_model.pth" if not args.finetune else "model_finetuned.pth"
    
    if model_type == "imuposer":
        model = IMUPoserNet(**checkpoints)
    elif model_type == "mobileposer":
        model = MobilePoserNet(**checkpoints)
    elif model_type == "heightposer":
        model = HeightPoserNet(**checkpoints)
    else:
        raise ValueError(f"Unknown model name: {model_type}")
        
    model_path = Path("data/checkpoints") / args.name / model_config.combo_id / model_name
    torch.save(model.state_dict(), model_path)
    print(f"Model written to {model_path}.")
