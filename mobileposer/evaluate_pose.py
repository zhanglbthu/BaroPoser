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

body_model = ParametricModel(paths.smpl_file)

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

@torch.no_grad()
def evaluate_pose(model: PoseNet, dataset, save_dir=None):
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

            pose_p = model.predict(x, pose_t[0], gt_vel=vel_t)
            
            online_errs.append(evaluator.eval(pose_p, pose_t))
            
            if save_dir:
                torch.save({'pose_t': pose_t, 
                            'pose_p': pose_p, 
                            },
                           save_dir / f"{idx}.pt")


    evaluator.print(torch.stack(online_errs).mean(dim=0))
    
    log_path = save_dir / 'log.txt'
    
    for online_err in online_errs:
        with open(log_path, 'a', encoding='utf-8') as f:
            evaluator.print_single(online_err, file=f)


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