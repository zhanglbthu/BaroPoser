import os
import numpy as np
import torch
from argparse import ArgumentParser
from tqdm import tqdm

from mobileposer.config import *
from mobileposer.helpers import * 
import mobileposer.articulate as art
from mobileposer.utils.model_utils import load_imuposer_model, load_imuposer_glb_model, load_mobileposer_model, load_heightposer_model
from mobileposer.data import PoseDataset
from pathlib import Path
from mobileposer.utils.file_utils import (
    get_best_checkpoint
)
from mobileposer.config import model_config, realdata
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

@torch.no_grad()
def evaluate_pose(subject_dir, model='imuposer', sub_align=None):
    actions = [d for d in os.listdir(subject_dir) if os.path.isdir(os.path.join(subject_dir, d))]

    # setup Pose Evaluator
    evaluator = PoseEvaluator()

    # track errors
    online_errs = []
    
    for action in tqdm(actions):
        if action not in sub_align:
            continue
        
        interval = sub_align[action]
        
        test_data = torch.load(os.path.join(subject_dir, action, 'test.pt'))
        pose_t = test_data['pose'][0].view(-1, 24, 3, 3)
            
        result_path = os.path.join(subject_dir, action, 'results', f'{model}.pt')
        result = torch.load(result_path)
        pose_p = result['pose'].view(-1, 24, 3, 3)
        
        # pose_p = pose_p[interval[0]:interval[1]]
        pose_t = pose_t[interval[0]:interval[1]]
            
        online_errs.append(evaluator.eval(pose_p, pose_t))

    evaluator.print(torch.stack(online_errs).mean(dim=0))
        
    log_path = os.path.join(subject_dir, f'{model}_eval.txt')
        
    # 清空原有内容
    with open(log_path, 'w', encoding='utf-8') as f:
        pass
        
    for online_err in online_errs:
        with open(log_path, 'a', encoding='utf-8') as f:
            evaluator.print_single(online_err, file=f)
            
    return online_errs

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='default')
    args = parser.parse_args()
    
    model = args.model
    
    subjects = ['sub1', 'sub2', 'sub3', 'sub4', 'sub5']
    real_data_dir = 'data/livedemo/real_data/processed'
    all_errs = []
    for subject in subjects:
        print(f'Evaluating {subject}...')
        subject_dir = os.path.join(real_data_dir, subject)
        sub_align = realdata.time_align[subject]
        errs = evaluate_pose(subject_dir, model=model, sub_align=sub_align)
        all_errs.extend(errs)
    
    # print 分隔符
    print('-' * 80)
    print('len(all_errs):', len(all_errs))
    all_errs = torch.stack(all_errs).mean(dim=0)
    evaluator = PoseEvaluator()
    evaluator.print(all_errs)