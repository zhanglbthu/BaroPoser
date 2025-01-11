import os
import numpy as np
import pickle
import torch
from argparse import ArgumentParser
from tqdm import tqdm
import glob

from mobileposer.articulate.model import ParametricModel
from mobileposer.articulate import math
from mobileposer.config import paths, datasets

from pathlib import Path
import articulate as art
from utils.data_utils import _foot_contact, _get_heights, _foot_min, _get_ground
import sys

target_fps = 30

def resample(tensor, target_fps):
    indices = torch.arange(0, tensor.shape[0], 60 / target_fps)
    
    start_indices = torch.floor(indices).long()
    end_indices = torch.ceil(indices).long()
    
    end_indices[end_indices >= tensor.shape[0]] = tensor.shape[0] - 1
    
    start = tensor[start_indices]
    end = tensor[end_indices]
    
    floats = indices - start_indices
    for _ in range(len(tensor.shape) - 1):
        floats = floats.unsqueeze(-1)
        
    weights = torch.ones_like(start) * floats

    torch_lerped = start * (1 - weights) + end * weights

    return torch_lerped

def resample_dataset(data_dir, resample_dir, target_fps):
    # 遍历data_dir下的所有文件以.pt结尾的文件
    for file in data_dir.glob("*.pt"):
        print(f"Resampling {file}")
        data = torch.load(file)
        resampled_data = {}
        for key, _ in data.items():
            resampled_data[key] = [resample(tensor, target_fps) for tensor in data[key]]
        new_file = resample_dir / file.name
        torch.save(resampled_data, new_file)

def create_directories():
    paths.processed_datasets.mkdir(exist_ok=True, parents=True)
    paths.eval_dir.mkdir(exist_ok=True, parents=True)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train", default=False, action="store_true")
    args = parser.parse_args()
    
    if args.train:
        resample_dir = paths.processed_datasets.parent / f"{target_fps}"
        os.makedirs(resample_dir, exist_ok=True)
        data_dir = paths.processed_datasets
    else:
        resample_dir = paths.processed_datasets.parent / f"{target_fps}_eval"
        os.makedirs(resample_dir, exist_ok=True)
        data_dir = paths.processed_datasets / "eval"
        
    resample_dataset(data_dir, resample_dir, target_fps)    