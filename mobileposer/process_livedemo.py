import time
import socket
import torch
from pygame.time import Clock

import articulate as art
import os
from config import *
from articulate.utils.noitom import *
from articulate.utils.unity import MotionViewer

from articulate.utils.wearable import WearableSensorSet
from auxiliary import calibrate_q, quaternion_inverse

from utils.model_utils import load_mobileposer_model 
from mobileposer.data import PoseDataset
import numpy as np
import matplotlib
from utils.data_utils import _foot_min

colors = matplotlib.colormaps['tab10'].colors
body_model = art.ParametricModel(paths.smpl_file, device='cuda')

class ViewerManager:
    def __init__(self, init_ground=0.):
        self.init_ground = init_ground
    
    def plot_terrian(self, viewer: MotionViewer, f_pos, ground, color, width, offset = False):
        
        x_center = (f_pos[0][0] + f_pos[1][0]) / 2
        z_center = (f_pos[0][2] + f_pos[1][2]) / 2
        
        start_y = self.init_ground.clone().cpu().numpy()
        
        if offset:
            start_y -= 1e-2
            
        end_y = ground.item()
        
        start = [x_center, start_y, z_center]
        end = [x_center, end_y, z_center]
        
        viewer.draw_terrian(start, end, color=color, width=width, render=False)

if __name__ == '__main__':
    os.makedirs(paths.temp_dir, exist_ok=True)
    os.makedirs(paths.live_record_dir, exist_ok=True)
    
    device = torch.device("cuda")
    clock = Clock()
    
    # set network
    ckpt_path = "data/checkpoints/mobileposer/lw_rp/base_model.pth"
    model = load_mobileposer_model(ckpt_path, combo_id=model_config.combo_id)
    print('Model loaded.')
    
    # load livedemo data
    data_path = "data/livedemo/record/test_simple.pt"
    data = torch.load(data_path)
    
    viewer_manager = ViewerManager()
    
    model.eval()
    
    acc = data['acc'].to(device)
    ori = data['ori'].to(device)
    
    print('acc shape:', acc.shape)
    print('ori shape:', ori.shape)
    
    imu_set = amass.combos_mine[model_config.combo_id]
    acc = acc[:, imu_set] / amass.acc_scale
    ori = ori[:, imu_set]
    
    input = torch.cat([acc.flatten(1), ori.flatten(1)], dim=1)
    
    online_results = [model.forward_online(f) for f in torch.cat((input, input[-1].repeat(model_config.future_frames, 1)))]
    pose_p = torch.stack(online_results[model_config.future_frames:], dim=0)
    frame = pose_p.shape[0]
    
    with torch.no_grad(), MotionViewer(1) as viewer:
        for i in range(frame):
            clock.tick(30)
            viewer.clear_line(render=False), viewer.clear_point(render=False), viewer.clear_terrian(render=False)
            
            pose = pose_p[i].view(24, 3, 3).cpu().numpy()
            tran = torch.zeros(3).cpu().numpy()
            
            viewer.update_all([pose], [tran], render=False)
            
            viewer.render()
            
            print('\r', clock.get_fps(), end='')