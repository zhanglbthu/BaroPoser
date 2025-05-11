import os
import numpy as np
import torch
from argparse import ArgumentParser
import tqdm 

from mobileposer.config import *
from mobileposer.helpers import * 
import mobileposer.articulate as art
from mobileposer.utils.model_utils import load_imuposer_model, load_imuposer_glb_model, load_mobileposer_model, load_heightposer_model
from mobileposer.data import PoseDataset
from pathlib import Path
from mobileposer.utils.file_utils import (
    get_best_checkpoint
)
from mobileposer.config import model_config
from process import _foot_ground_probs
from mobileposer.articulate.model import ParametricModel

body_model = ParametricModel(paths.smpl_file, device=model_config.device)

class PoseEvaluator:
    def __init__(self):
        self._eval_fn = art.FullMotionEvaluator(paths.smpl_file, joint_mask=torch.tensor([1, 2, 16, 17]), fps=datasets.fps)

    def eval(self, pose_p, pose_t, joint_p=None, tran_p=None, tran_t=None):
        pose_p = pose_p.clone().view(-1, 24, 3, 3)
        pose_t = pose_t.clone().view(-1, 24, 3, 3)
        if tran_p is not None and tran_t is not None:
            tran_p = tran_p.clone().view(-1, 3)
            tran_t = tran_t.clone().view(-1, 3)
        pose_p[:, joint_set.ignored] = torch.eye(3, device=pose_p.device)
        pose_t[:, joint_set.ignored] = torch.eye(3, device=pose_t.device)

        errs = self._eval_fn(pose_p, pose_t, tran_p=tran_p, tran_t=tran_t)
        return torch.stack([errs[9], errs[3], errs[9], errs[0]*100, errs[7]*100, errs[1]*100, errs[4] / 100, errs[6]])

    @staticmethod
    def print(errors):
        for i, name in enumerate(['SIP Error (deg)', 'Angular Error (deg)', 'Masked Angular Error (deg)',
                                  'Positional Error (cm)', 'Masked Positional Error (cm)', 'Mesh Error (cm)', 
                                  'Jitter Error (100m/s^3)', 'Distance Error (cm)']):
            print('%s: %.2f (+/- %.2f)' % (name, errors[i, 0], errors[i, 1]))
            
    @staticmethod
    def print_single(errors, file=None):
        metric_names = [
            'SIP Error (deg)', 
            'Angular Error (deg)', 
            'Masked Angular Error (deg)',
            'Positional Error (cm)', 
            'Masked Positional Error (cm)', 
            'Mesh Error (cm)', 
            'Jitter Error (100m/s^3)', 
            'Distance Error (cm)'
        ]
        
        # 找出最长的指标名，以便统一对齐
        max_len = max(len(name) for name in metric_names)

        # 将每个指标的输出字符串保存到列表中，最后 join 成一行输出
        output_parts = []
        for i, name in enumerate(metric_names):
            if name in ['Angular Error (deg)', 'Mesh Error (cm)']:
                # 对这类指标使用“均值 ± 方差”的格式
                output_str = f"{name:<{max_len}}: {errors[i,0]:.2f}"
            else:
                continue
            
            output_parts.append(output_str)

        # 最终打印为一行
        print(" | ".join(output_parts), file=file)
        # 如果需要在末尾换行，print 本身就会换行，无需额外操作

def evaluate_joint(joint_t, joint_p):
    joint_t = joint_t.clone().view(-1, 24, 3)
    joint_p = joint_p.clone().view(-1, 24, 3)
    
    # align root joint
    offset_from_p2t = joint_t[:, 0] - joint_p[:, 0]
    joint_p += offset_from_p2t.unsqueeze(1)
    
    je = (joint_t - joint_p).norm(dim=2)
    
    return je.mean()

def edit_pose(pose_p):
    pose = pose_p.clone().view(-1, 24, 3, 3)
    _, joint = body_model.forward_kinematics(pose=pose)
    
    _, joint_init = body_model.forward_kinematics(pose=pose[0].unsqueeze(0))
    joint_init = joint_init[0]
    N = pose.shape[0]
    
    vel_threshold = 1e-5
    init_threshold = 5e-3
    
    static = torch.zeros((N, 1), dtype=torch.bool, device=pose.device)    
    S = np.diag([1, -1, -1])
    S = torch.tensor(S, dtype=torch.float32, device=pose.device)
    
    start = False
    
    for i in range(1, N):
        last_larm = joint[i-1, joint_set.larm].clone().view(-1)
        last_rarm = joint[i-1, joint_set.rarm].clone().view(-1)
        cur_larm = joint[i, joint_set.larm].clone().view(-1)
        cur_rarm = joint[i, joint_set.rarm].clone().view(-1)
        init_larm = joint_init[joint_set.larm].clone().view(-1)
        init_rarm = joint_init[joint_set.rarm].clone().view(-1)
        
        vel_err = torch.mean((last_larm - cur_larm) ** 2)
        init_err = torch.mean((init_larm - cur_larm) ** 2)
        rarm_err = torch.mean((init_rarm - cur_rarm) ** 2)

        if vel_err > vel_threshold and init_err > init_threshold:
            start = True
            # pose[i, joint_set.rarm] = - pose[i, joint_set.larm].clone()
        
        if vel_err < vel_threshold and init_err < init_threshold:
            static[i] = True
            
        # 如果过去一段时间内，static为True，则将当前帧的右臂姿态设置为左臂姿态的镜像
        if i > 10 and torch.all(static[i-10:i]):
            start = False
            
        if not start:
            pose[i, joint_set.rarm] = torch.matmul(S, torch.matmul(pose[i, joint_set.larm], S))
            
    return pose

@torch.no_grad()
def evaluate_pose(model, dataset, save_dir=None, debug=False, processed=None):
    # specify device
    device = model_config.device

    # load data
    xs, ys, js = zip(*[(imu.to(device), (pose.to(device), tran), (joint.to(device), vel.to(device))) for imu, pose, joint, tran, vel in dataset])

    # setup Pose Evaluator
    evaluator = PoseEvaluator()

    # track errors
    online_errs = []
    
    model.eval()
      
    with torch.no_grad():
        for idx, (x, y, j) in enumerate(tqdm.tqdm(zip(xs, ys, js), total=len(xs))):
            # x: [N, 60], y: ([N, 144], [N, 3])
            
            model.reset()

            pose_t, _ = y
            joint_t, vel_t = j
            vel_t = vel_t[:, amass.vel_joint].view(-1, len(amass.vel_joint)*3)
            
            pose_t = art.math.r6d_to_rotation_matrix(pose_t)
            pose_t = pose_t.view(-1, 24, 3, 3)  
            
            if debug:
                joint_p, joint_all_p, pose_p = model.predict(x, pose_t[0], debug=True)
                if save_dir:
                    torch.save({'pose_t': pose_t, 
                                'pose_p': pose_p, 
                                'joint_p': joint_p,
                                'joint_all_p': joint_all_p,
                                },
                                save_dir / f"{idx}.pt")
                continue
            
            if processed:
                pose_p = torch.load(processed + '/' + f"{idx}.pt")['pose_p'].to(device).view(-1, 24, 3, 3)
            else:
                if model_config.winit:
                    pose_p = model.predict(x, pose_t[0], poser_only=True)
                else:
                    x = x[:, :24]
                    online_results = [model.forward_online(f) for f in torch.cat((x, x[-1].repeat(model_config.future_frames, 1)))]
                    pose_p = torch.stack(online_results[model_config.future_frames:], dim=0)
            
            online_errs.append(evaluator.eval(pose_p, pose_t))
            
            if save_dir and not processed:
                torch.save({'pose_t': pose_t, 
                            'pose_p': pose_p, 
                            },
                           save_dir / f"{idx}.pt")

    if not debug:
        evaluator.print(torch.stack(online_errs).mean(dim=0))
        
        log_path = save_dir / 'log.txt'
        
        # 清空原有内容
        with open(log_path, 'w', encoding='utf-8') as f:
            pass
        
        for online_err in online_errs:
            with open(log_path, 'a', encoding='utf-8') as f:
                evaluator.print_single(online_err, file=f)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='data/checkpoints/imuposer_local/lw_rp/base_model.pth')
    parser.add_argument('--processed', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='dip')
    parser.add_argument('--name', type=str, default='default')
    parser.add_argument('--combo_id', type=str, default='lw_rp_h')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

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
    
    fold = 'test'
    
    dataset = PoseDataset(fold=fold, evaluate=args.dataset, combo_id=model_config.combo_id, 
                          wheights=model_config.data_heights)
    
    save_dir = Path('data') / 'eval' / model_config.name / model_config.combo_id / args.dataset
    if args.debug:
        save_dir = Path('debug')
    print(f"Saving results to: {save_dir}")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # evaluate pose
    print(f"Starting evaluation: {args.dataset.capitalize()}")
    # evaluate_pose(model, dataset, evaluate_tran=True, save_dir=save_dir)
    evaluate_pose(model=model, dataset=dataset, save_dir=save_dir, debug=args.debug, processed=args.processed)