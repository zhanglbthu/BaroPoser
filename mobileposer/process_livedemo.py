import time
import socket
import torch
from pygame.time import Clock
from argparse import ArgumentParser

import articulate as art
import os
from config import *
from articulate.utils.unity import MotionViewer

from articulate.utils.wearable import WearableSensorSet
from auxiliary import calibrate_q, quaternion_inverse

from utils.model_utils import load_mobileposer_model, load_imuposer_model, load_heightposer_model
from mobileposer.data import PoseDataset
import numpy as np
import matplotlib
from utils.data_utils import _foot_min

colors = matplotlib.colormaps['tab10'].colors
body_model = art.ParametricModel(paths.smpl_file, device='cuda')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='data/checkpoints/heightposer/lw_rp/base_model.pth')
    os.makedirs(paths.temp_dir, exist_ok=True)
    os.makedirs(paths.live_record_dir, exist_ok=True)
    args = parser.parse_args()
    
    device = torch.device("cuda")
    clock = Clock()
    
    # load model
    model_name = model_config.name.split('_')[0]
    if model_name == 'imuposer':
        model = load_imuposer_model(model_path=args.model, combo_id=model_config.combo_id)
    elif model_name == 'mobileposer':
        model = load_mobileposer_model(model_path=args.model, combo_id=model_config.combo_id)
    elif model_name == 'heightposer':
        model = load_heightposer_model(model_path=args.model, combo_id=model_config.combo_id)
    else:
        raise ValueError(f"Model {model_name} not supported.")
    print('Model loaded.')
    
    # load livedemo data
    data_path = "data/livedemo/raw/test_simple.pt"
    save_path = "data/livedemo/processed"
    os.makedirs(save_path, exist_ok=True)
    data = torch.load(data_path)
    
    model.eval()
    
    acc = data['acc'].to(device)
    ori = data['ori'].to(device)
    
    imu_set = amass.combos_mine[model_config.combo_id]
    acc = acc[:, imu_set] / amass.acc_scale
    ori = ori[:, imu_set]
    
    input = torch.cat([acc.flatten(1), ori.flatten(1)], dim=1)
    
    if model_config.winit:
        # 初始化24个关节的姿态，均为单位矩阵
        pose_t = torch.eye(3).repeat(1, 24, 1, 1).to(device)
        pose_p = model.predict(input, pose_t[0])
    else:
        online_results = [model.forward_online(f) for f in torch.cat((input, input[-1].repeat(model_config.future_frames, 1)))]
        pose_p = torch.stack(online_results[model_config.future_frames:], dim=0)
    
    torch.save({'pose_p': pose_p}, save_path + "/pose_p.pt")