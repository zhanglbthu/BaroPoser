import math
import numpy as np
import torch

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