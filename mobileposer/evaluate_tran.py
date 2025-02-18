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

body_model = ParametricModel(paths.smpl_file)

@torch.no_grad()
def evaluate_tran(model, dataset, save_dir=None, debug=False):
    # specify device
    device = model_config.device

    # load data
    xs, ys, js = zip(*[(imu.to(device), (pose.to(device), tran), (joint.to(device), vel.to(device))) for imu, pose, joint, tran, vel in dataset])

    # track errors
    online_errs = []
    tran_errors = {window_size: [] for window_size in list(range(1, 8))}
    
    model.eval()
    with torch.no_grad():
        for idx, (x, y, j) in enumerate(tqdm.tqdm(zip(xs, ys, js), total=len(xs))):
            # x: [N, 60], y: ([N, 144], [N, 3])
            
            model.reset()

            pose_t, tran_t = y
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
            
            if model_config.winit:
                _, joint_p, pose_p = model.predict(x, pose_t[0], detail=True)
                
                tran_p = [model.forward_online(joints=j) for j in joint_p]
                
                    
            else:
                online_results = [model.forward_online(f, tran=True) for f in torch.cat((x, x[-1].repeat(model_config.future_frames, 1)))]
                pose_p, tran_p = [torch.stack(_)[model_config.future_frames:] for _ in zip(*online_results)]
            
            if True:
                # compute gt move distance at every frame 
                move_distance_t = torch.zeros(tran_t.shape[0])
                v = (tran_t[1:] - tran_t[:-1]).norm(dim=1)
                for j in range(len(v)):
                    move_distance_t[j + 1] = move_distance_t[j] + v[j] # distance travelled

                for window_size in tran_errors.keys():
                    # find all pairs of start/end frames where gt moves `window_size` meters
                    frame_pairs = []
                    start, end = 0, 1
                    while end < len(move_distance_t):
                        if move_distance_t[end] - move_distance_t[start] < window_size: # if not less than the window_size (in meters)
                            end += 1
                        else:
                            if len(frame_pairs) == 0 or frame_pairs[-1][1] != end:
                                frame_pairs.append((start, end))
                            start += 1

                    # calculate mean distance error 
                    errs = []
                    for start, end in frame_pairs:
                        # vel_p = tran_p_offline[end] - tran_p_offline[start]
                        vel_p = tran_p[end] - tran_p[start]
                        vel_t = (tran_t[end] - tran_t[start]).to(device)
                        errs.append((vel_t - vel_p).norm() / (move_distance_t[end] - move_distance_t[start]) * window_size)
                    if len(errs) > 0:
                        tran_errors[window_size].append(sum(errs) / len(errs))

    if not debug:
        print([0] + [torch.tensor(_).mean() for _ in tran_errors.values()])

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='data/checkpoints/imuposer_local/lw_rp_h/base_model.pth')
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
                          wheights=model_config.wheights)
    
    save_dir = Path('data') / 'eval' / model_config.name / model_config.combo_id / args.dataset
    if args.debug:
        save_dir = Path('debug')
    print(f"Saving results to: {save_dir}")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # evaluate pose
    print(f"Starting evaluation: {args.dataset.capitalize()}")
    # evaluate_pose(model, dataset, evaluate_tran=True, save_dir=save_dir)
    evaluate_tran(model=model, dataset=dataset, save_dir=save_dir, debug=args.debug)