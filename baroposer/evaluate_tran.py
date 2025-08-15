import os
import numpy as np
import torch
from argparse import ArgumentParser
from tqdm import tqdm

from config import *
from helpers import * 
import articulate as art
from utils.model_utils import load_imuposer_model, load_imuposer_glb_model, load_mobileposer_model, load_heightposer_model
from data import PoseDataset
from pathlib import Path
from utils.file_utils import (
    get_best_checkpoint
)
from config import model_config, realdata
from process import _foot_ground_probs
from articulate.model import ParametricModel

body_model = ParametricModel(paths.smpl_file, device=model_config.device)
device = model_config.device

@torch.no_grad()
def evaluate_pose(subject_dir, model='imuposer', sub_align=None, tran_errors=None):
    actions = [d for d in os.listdir(subject_dir) if os.path.isdir(os.path.join(subject_dir, d))]

    # log_path = os.path.join(subject_dir, f'{model}_tran.txt')
    # if os.path.exists(log_path):
    #     os.remove(log_path)
    
    for action in tqdm(actions):
        # tran_errors = {window_size: [] for window_size in list(range(1, 8))}
        if action not in sub_align:
            continue
        
        interval = sub_align[action]
        
        test_data = torch.load(os.path.join(subject_dir, action, 'test.pt'))
        tran_t = test_data['tran'][0].view(-1, 3)
        # permute tran_t: [x, y, z] to [-x, y, -z]
        tran_t = torch.stack([-tran_t[:, 0], tran_t[:, 1], -tran_t[:, 2]], dim=1)
            
        result_path = os.path.join(subject_dir, action, 'results', f'{model}.pt')
        result = torch.load(result_path)
        tran_p = result['tran'].view(-1, 3)
        tran_t = tran_t[interval[0]:interval[1]]
            
        move_distance_t = torch.zeros(tran_t.shape[0])
        v = (tran_t[1:] - tran_t[:-1]).norm(dim=1)
        for j in range(len(v)):
            move_distance_t[j + 1] = move_distance_t[j] + v[j]
        
        for window_size in tran_errors.keys():
            frame_pairs = []
            start, end = 0, 1
            while end < len(move_distance_t):
                if move_distance_t[end] - move_distance_t[start] < window_size:
                    end += 1
                else:
                    if len(frame_pairs) == 0 or frame_pairs[-1][1] != end:
                        frame_pairs.append((start, end))
                    start += 1
            
            errs = []
            for start, end in frame_pairs:
                vel_p = tran_p[end] - tran_p[start]
                vel_t = (tran_t[end] - tran_t[start]).to(device)
                errs.append((vel_t - vel_p).norm() / (move_distance_t[end] - move_distance_t[start]) * window_size)
            if len(errs) > 0:
                tran_errors[window_size].append(sum(errs) / len(errs))
        # with open(log_path, 'a', encoding='utf-8') as f:
        #     formatted = [0] + [f"{torch.tensor(_).mean().item():.2f}" for _ in tran_errors.values()]
        #     # print action and formatted errors to file
        #     # align the action name to 20 characters
        #     # action is str, formatted is list of int
        #     formatted_str = [f"{x:<6}" for x in formatted]
        #     f.write(f"{action:<20} {' '.join(formatted_str)}\n")
                
    # formatted = [0] + [f"{torch.tensor(_).mean().item():.2f}" for _ in tran_errors.values()]
    # print(formatted)
    return tran_errors
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='default')
    args = parser.parse_args()
    
    model = args.model
    
    subjects = ['sub1', 'sub2', 'sub3', 'sub4', 'sub5']
    real_data_dir = 'data/livedemo/real_data/processed'
    tran_errors = {window_size: [] for window_size in list(range(1, 8))}
    for subject in subjects:
        print(f'Evaluating {subject}...')
        subject_dir = os.path.join(real_data_dir, subject)
        sub_align = realdata.time_align[subject]
        tran_errors = evaluate_pose(subject_dir, model=model, sub_align=sub_align, tran_errors=tran_errors)
    
    for window_size in tran_errors.keys():
        # print length of tran_errors[window_size]
        print(f"Window size {window_size}: {len(tran_errors[window_size])} pairs")
    
    formatted = [0] + [f"{torch.tensor(_).mean().item():.2f}" for _ in tran_errors.values()]
    print(formatted)
