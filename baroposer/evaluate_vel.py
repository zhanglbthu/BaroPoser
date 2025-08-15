import os
import numpy as np
import torch
from argparse import ArgumentParser
import tqdm 

from mobileposer.config import *
from mobileposer.helpers import * 
import mobileposer.articulate as art
from mobileposer.constants import MODULES
from mobileposer.utils.model_utils import load_pose_model
from mobileposer.data import PoseDataset
from mobileposer.models_new.net import PoseNet
from pathlib import Path
from mobileposer.utils.file_utils import (
    get_best_checkpoint
)
from mobileposer.config import model_config
from process import _foot_ground_probs
from mobileposer.articulate.model import ParametricModel
import matplotlib.pyplot as plt

body_model = ParametricModel(paths.smpl_file)

plot_dir = 'plot/imuposer'
axis = 11
plot_dir = os.path.join(plot_dir, f'axis_{axis}')
os.makedirs(plot_dir, exist_ok=True)

def evaluate_vel(vel_t, vel_p, axis=0, idx=0):
    difference = (vel_t - vel_p)
    
    axis_diff = difference[:, axis]
    
    plt.plot(axis_diff.cpu().numpy())
    plt.title(f"Velocity Error for Axis {axis}")
    plt.xlabel("Frame")
    plt.ylabel("Error")
    plt.grid(True)
    
    plt.savefig(plot_dir + f'/error_{idx}.png')
    plt.close()

def evaluate_vel_abs(vel_t, vel_p, axis=0, idx=0):
    # 提取指定轴上的数据
    gt_axis = vel_t[:, axis].cpu().numpy()  
    vel_p_axis = vel_p[:, axis].cpu().numpy() 

    # 绘制两条曲线
    plt.plot(gt_axis, label='Ground Truth', color='b')  # 绘制 gt 曲线，使用蓝色
    plt.plot(vel_p_axis, label='Predicted', color='r')  # 绘制 vel_p 曲线，使用红色

    # 添加标题和标签
    plt.title(f'Comparison on Axis {axis}')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()  # 显示图例
    plt.grid(True)

    # 保存为图片
    plt.savefig(plot_dir + f'/abs_{idx}.png')
    plt.close()  # 关闭当前图形，释放内存

@torch.no_grad()
def evaluate_pose(model: PoseNet, dataset, save_dir=None):
    # specify device
    device = model_config.device

    # load data
    xs, ys, js = zip(*[(imu.to(device), (pose.to(device), tran), (joint.to(device), vel.to(device))) for imu, pose, joint, tran, vel in dataset])
    
    model.eval()
    with torch.no_grad():
        for idx, (x, y, j) in enumerate(tqdm.tqdm(zip(xs, ys, js), total=len(xs))):
            # x: [N, 60], y: ([N, 144], [N, 3])
            
            model.reset()
            _, vel_t = j
            vel_t = vel_t[:, amass.vel_joint].view(-1, 12)

            vel_p = model.predict_vel(x)
            evaluate_vel(vel_t, vel_p, axis=axis, idx=idx)
            evaluate_vel_abs(vel_t, vel_p, axis=axis, idx=idx)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='dip')
    parser.add_argument('--name', type=str, default='default')
    parser.add_argument('--combo_id', type=str, default='lw_rp_h')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # record combo
    print(f"combo: {amass.combos}")

    # load model
    model = load_pose_model(args.model)
    
    fold = 'test'
    
    dataset = PoseDataset(fold=fold, evaluate=args.dataset, combo_id=args.combo_id, 
                          wheights=model_config.wheights)
    
    save_dir = Path('data') / 'eval' / model_config.name / args.combo_id / args.dataset
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # evaluate pose
    print(f"Starting evaluation: {args.dataset.capitalize()}")
    # evaluate_pose(model, dataset, evaluate_tran=True, save_dir=save_dir)
    evaluate_pose(model=model, dataset=dataset, save_dir=save_dir)