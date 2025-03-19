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
    parser.add_argument('--name', type=str, default='default')
    args = parser.parse_args()
    
    device = torch.device("cuda")
    clock = Clock()
    
    # load model
    model_name = args.name
    if model_name == 'imuposer':
        model_path = 'data/checkpoints/imuposer/lw_rp/base_model.pth'
        model = load_imuposer_model(model_path=model_path, combo_id=model_config.combo_id)
    elif model_name == 'mobileposer':
        model_path = 'data/checkpoints/mobileposer/lw_rp/base_model.pth'
        model = load_mobileposer_model(model_path=model_path, combo_id=model_config.combo_id)
    else:
        raise ValueError(f"Model {model_name} not supported.")
    print('Model loaded.')
    
    # load livedemo data
    data_name = "freestyle_20250317_164707"
    data_path = "data/livedemo/record/" + data_name + ".pt"
    save_dir = "data/livedemo/processed/" + data_name
    os.makedirs(save_dir, exist_ok=True)
    save_path = save_dir + "/" + model_name + ".pt"
    data = torch.load(data_path)
    
    model.eval()
    
    acc = data['acc'].to(device)
    ori = data['ori'].to(device)
    
    imu_set = amass.combos_mine[model_config.combo_id]
    acc = acc[:, imu_set] / amass.acc_scale
    ori = ori[:, imu_set]
    
    input = torch.cat([acc.flatten(1), ori.flatten(1)], dim=1)
    
    online_results = [model.forward_online(f) for f in torch.cat((input, input[-1].repeat(model_config.future_frames, 1)))]
    pose_p = torch.stack(online_results)[model_config.future_frames:].view(-1, 24, 3, 3)
    
    torch.save({'pose': pose_p}, save_path)