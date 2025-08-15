import torch
from mobileposer.articulate.model import ParametricModel
from mobileposer.config import paths, datasets

body_model = ParametricModel(paths.smpl_file, device='cuda')

def process_mobileposer_data(data):
    pose_t = data['pose_t']
    pose_p = data['pose_p_online']
    
    tran_t = data['tran_t'].to(pose_t.device)
    tran_p = data['tran_p_online']
    
    pose_t = pose_t.view(-1, 24, 3, 3)
    pose_p = pose_p.view(-1, 24, 3, 3)
    
    # # convert to cpu
    # pose_t = pose_t.cpu().numpy()
    # pose_p = pose_p.cpu().numpy()
    # tran_t = tran_t.cpu().numpy()
    # tran_p = tran_p.cpu().numpy()
    
    # 将tran_t和tran_p的第一帧align到一起
    tran_t = tran_t - tran_t[0] + tran_p[0]
    
    return pose_t, pose_p, tran_t, tran_p

if __name__ == '__main__':
    data_path = "data/eval/cmu/4.pt"
    data = torch.load(data_path)
    
    pose_t, pose_p, tran_t, tran_p = process_mobileposer_data(data)
    
    _, glb_joint = body_model.forward_kinematics(pose=pose_t, calc_mesh=False)
    
    glb_root_joint = glb_joint[:, 0]
    print(glb_root_joint)