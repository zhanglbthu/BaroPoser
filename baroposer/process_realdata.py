import torch
from pygame.time import Clock
from argparse import ArgumentParser
import articulate as art
import os
from config import *
from utils.model_utils import load_mobileposer_model, load_imuposer_model, load_heightposer_model
import matplotlib
from tqdm import tqdm
from sensor_height import KalmanFilter, get_k_bias, get_p_bias, get_bias
import numpy as np

colors = matplotlib.colormaps['tab10'].colors
body_model = art.ParametricModel(paths.smpl_file, device='cuda')
vi_mask = torch.tensor([1961, 5424, 876, 4362, 411, 3021])

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='default')
    parser.add_argument('--data_name', type=str, default='default')
    parser.add_argument('--p_bias', type=float, default=6.95)
    args = parser.parse_args()
    
    device = torch.device("cuda")
    clock = Clock()
    
    # load model
    model_name = args.model.split('_')[0]
    if model_name == 'imuposer':
        model_path = 'data/checkpoints/imuposer/lw_rp/base_model.pth'
        model = load_imuposer_model(model_path=model_path, combo_id=model_config.combo_id)
    elif model_name == 'mobileposer':
        model_path = 'data/checkpoints/mobileposer/lw_rp/base_model.pth'
        model = load_mobileposer_model(model_path=model_path, combo_id=model_config.combo_id)
    elif model_name == 'heightposer':
        model_path = 'data/checkpoints/' + args.model + '/lw_rp/base_model.pth'
        model = load_heightposer_model(model_path=model_path, combo_id=model_config.combo_id)
    else:
        raise ValueError(f"Model {args.model} not supported.")
    print('Model loaded.')
    
    subjects = ['sub1', 'sub2', 'sub3', 'sub4', 'sub5']
    p_bias_list = [6.84, 6.81, 6.85, 6.83, 6.82]
    real_data_dir = 'data/livedemo/real_data/processed'
    t_pose = torch.eye(3).unsqueeze(0).repeat(24, 1, 1).to("cuda")
    _, glb_joint, vertex = body_model.forward_kinematics(t_pose.unsqueeze(0), calc_mesh=True)
    init_thigh = vertex[0, 4362, 1]
    # p_bias = args.p_bias
    
    for idx, subject in enumerate(subjects):
        # subject_dir = 'data/livedemo/real_data/sub1'
        subject_dir = os.path.join(real_data_dir, subject)
        sub_align = realdata.time_align[subject]
        p_bias = p_bias_list[idx]
        for action in os.listdir(subject_dir):
            # 判断action路径是否为目录
            if action not in sub_align:
                continue
            print(f"Processing subject: {subject}, action: {action}")
            
            # load livedemo data
            data_path = os.path.join(subject_dir, action, 'test.pt')
            data = torch.load(data_path)
            
            save_dir = os.path.join(subject_dir, action, 'results')
            os.makedirs(save_dir, exist_ok=True)
            save_path = save_dir + "/" + args.model + ".pt"
            
            model.eval()
            
            acc = data['aM'][0].to(device).view(-1, 6, 3)
            ori = data['RMB'][0].to(device).view(-1, 6, 3, 3)
            pose = data['pose'][0].to(device).view(-1, 24, 3, 3)
            
            imu_set = amass.combos_mine[model_config.combo_id]
            acc = acc[:, imu_set] / amass.acc_scale
            ori = ori[:, imu_set]
            
            input = torch.cat([acc.flatten(1), ori.flatten(1)], dim=1)
            
            frames = torch.cat((input, input[-1].repeat(model_config.future_frames, 1)))
            interval = sub_align[action]
            tran_p = None
            
            # 加进度条
            if model_name == 'heightposer':
                init_pose = torch.eye(3, device='cuda').repeat(24, 1, 1)
                
                root_ori = ori[:, -1]
                root_angular_vel = art.math.rotation_matrix_to_axis_angle(root_ori[:-1].transpose(1, 2).bmm(root_ori[1:]))
                root_angular_vel = torch.cat((torch.zeros([1, 3], device=device), root_angular_vel)).view(-1, 3)
                
                # compute ground truth relative height
                _, _, vertex = body_model.forward_kinematics(pose, calc_mesh=True)
                h_gt = vertex[:, vi_mask[3], 1] - vertex[:, vi_mask[0], 1]
                h_gt = h_gt[interval[0]:interval[1]].view(-1, 1)
                
                # compute sensor relative height
                pressure = data['pressure'][0].to(device).view(-1, 2)
                pressure = pressure[interval[0]:interval[1]]
                k_bias = h_gt[:10].mean() / (pressure[:10, 0] - (pressure[:10, 1] + p_bias)).mean() 
                h_sensor = (pressure[:, 0] - (pressure[:, 1] + p_bias)) * k_bias
                h_sensor = h_sensor.view(-1, 1)

                input = input[interval[0]:interval[1]]
                root_angular_vel = root_angular_vel[interval[0]:interval[1]]
                input = torch.cat([input, root_angular_vel, h_sensor], dim=1)
                
                pose_p, tran_p = model.predict(input, init_pose)

                # compute thigh height
                k_kf = k_bias * 100
                b_kf = - pressure[:10, 0].mean() * k_kf
                kf = KalmanFilter(k=k_kf, b=b_kf)
                for i in range(pressure.shape[0]):
                    a_thigh = input[i, 4].clone().cpu()
                    a = np.array([a_thigh * 100 * amass.acc_scale])
                    z = np.array([pressure[i, 0].cpu()])
                    kf.predict(a)
                    kf.update(z)
                    h_thigh = kf.get_height() * 0.01
                    # print(f"Frame {i}, Thigh height: {h_thigh} m")
                
                _, _, vertex = body_model.forward_kinematics(pose_p, calc_mesh=True)
                local_thigh_h = vertex[0][4362][1]
                
                h_thigh = torch.tensor(h_thigh, device=device)
                tran_p_y = h_thigh - (local_thigh_h - init_thigh)
                tran_p[:, 1] = tran_p_y
    
            else:
                online_results = []
                for f in tqdm(frames, desc="Running forward_online"):
                    if model_name == 'imuposer':
                        online_results.append(model.forward_online(f))
                    else:
                        online_results.append(model.forward_online(f, tran=True))
                    
                if model_name == 'imuposer':
                    pose_p = torch.stack(online_results)[model_config.future_frames:].view(-1, 24, 3, 3)
                else:
                    pose_p, tran_p = [torch.stack(_)[model_config.future_frames:] for _ in zip(*online_results)]
                    
                pose_p = pose_p[interval[0]:interval[1]]
                tran_p = tran_p[interval[0]:interval[1]] if tran_p is not None else None
            
            if tran_p is None:
                torch.save({'pose': pose_p}, save_path)
            else:
                torch.save({'pose': pose_p, 'tran': tran_p}, save_path)