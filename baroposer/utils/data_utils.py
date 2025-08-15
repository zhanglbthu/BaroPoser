import math
import numpy as np
import torch
from config import amass

def smooth_avg(acc=None, s=3):
    nan_tensor = (torch.zeros((s // 2, acc.shape[1], acc.shape[2])) * torch.nan)
    acc = torch.cat((nan_tensor, acc, nan_tensor))
    tensors = []
    for i in range(s):
        L = acc.shape[0]
        tensors.append(acc[i:L-(s-i-1)])

    smoothed = torch.stack(tensors).nanmean(dim=0)
    return smoothed

def _foot_min(joint):
    lheel_y = joint[:, 7, 1]
    rheel_y = joint[:, 8, 1]
    
    ltoe_y = joint[:, 10, 1]
    rtoe_y = joint[:, 11, 1]
    
    # 取四个点的最小值 [N, 1]
    points = torch.stack((lheel_y, rheel_y, ltoe_y, rtoe_y), dim=1)
    min_y, _ = torch.min(points, dim=1, keepdim=True)   
    assert min_y.shape == (joint.shape[0], 1)
    return min_y

def _get_heights(vert, ground, vi_mask):
    '''
    select pocket and wrist vertex
    '''
    pocket_height = vert[:, vi_mask[3], 1].unsqueeze(1) - ground
    wrist_height = vert[:, vi_mask[0], 1].unsqueeze(1) - ground
    
    # return [N, 2]
    return torch.stack((pocket_height, wrist_height), dim=1)

def _foot_contact(fp_list):
    """
    判断 n 帧中是否连续有至少一只脚接触地面。
    """
    for fp in fp_list:
        if fp[0] or fp[1]:  #
            continue
        else:
            return False
    return True  

def _get_ground(joint, fc_probs, length, contact_num=5):
    f_min = _foot_min(joint)
    cur_ground = f_min[0].item()
    ground = torch.full_like(f_min, cur_ground)
    
    for frame in range(1, length+1):
        g = f_min[frame - 1]
        if frame >= contact_num:
            fp_last_n = fc_probs[frame - contact_num: frame]
        else:
            fp_last_n = fc_probs[:frame]
        
        ground[frame-1] = cur_ground
        contact = _foot_contact(fp_last_n)
        residual = abs(g - cur_ground)
        if contact and residual > 1e-3:
            ground[frame-1] = g
            cur_ground = g
            
    return ground

def normalize_and_concat(glb_acc, glb_rot):
    imu_num = glb_acc.shape[1]
    j_idx = imu_num - 1
    
    glb_acc = glb_acc.view(-1, imu_num, 3)
    glb_rot = glb_rot.view(-1, imu_num, 3, 3)
    acc = torch.cat((glb_acc[:, :j_idx] - glb_acc[:, j_idx:], glb_acc[:, j_idx:]), dim=1).bmm(glb_rot[:, -1])
    ori = torch.cat((glb_rot[:, j_idx:].transpose(2, 3).matmul(glb_rot[:, :j_idx]), glb_rot[:, j_idx:]), dim=1)
    
    return acc, ori

def normalize_joint(joint, norm_rot):
    
    norm_rot = norm_rot.view(-1, 3, 3)
    j = (joint - joint[:, 15].unsqueeze(1)).bmm(norm_rot)

    j = j[:, amass.normalize_joints].flatten(1)
    
    return j