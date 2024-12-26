import os
import numpy as np
import pickle
import torch
from argparse import ArgumentParser
from tqdm import tqdm
import glob

from mobileposer.articulate.model import ParametricModel
from mobileposer.articulate import math
from mobileposer.config import paths, datasets

from pathlib import Path


# specify target FPS
TARGET_FPS = 60

# left wrist, right wrist, left thigh, right thigh, head, pelvis, left shank, right shank
vi_mask = torch.tensor([1961, 5424, 876, 4362, 411, 3021, 1176, 4662])
ji_mask = torch.tensor([18, 19, 1, 2, 15, 0, 4, 5])
body_model = ParametricModel(paths.smpl_file)

def _syn_acc(v, smooth_n=4):
    """Synthesize accelerations from vertex positions."""
    mid = smooth_n // 2
    acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * 3600 for i in range(0, v.shape[0] - 2)])
    acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
    if mid != 0:
        acc[smooth_n:-smooth_n] = torch.stack(
            [(v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n]) * 3600 / smooth_n ** 2
             for i in range(0, v.shape[0] - smooth_n * 2)])
    return acc

def _relative_height(vert):
    '''
    Compute relative height of the body.
    '''
    relative_height = vert[:, vi_mask[0], 1] - vert[:, vi_mask[3], 1] # left wrist - right thigh
    return relative_height

def _foot_min(joint, fix=False):
    lheel_y = joint[:, 7, 1]
    rheel_y = joint[:, 8, 1]
    
    ltoe_y = joint[:, 10, 1]
    rtoe_y = joint[:, 11, 1]
    
    # 取四个点的最小值 [N, 1]
    points = torch.stack((lheel_y, rheel_y, ltoe_y, rtoe_y), dim=1)
    min_y, _ = torch.min(points, dim=1, keepdim=True)   
    assert min_y.shape == (joint.shape[0], 1)
    
    if fix:
        # min_y所有值都取第一帧
        min_y[:] = min_y[0]
    
    return min_y

def _get_heights(vert, ground):
    pocket = vert[:, vi_mask[3], 1].unsqueeze(1)
    pocket_height = vert[:, vi_mask[3], 1].unsqueeze(1) - ground
    wrist_height = vert[:, vi_mask[0], 1].unsqueeze(1) - ground
    
    # return [N, 2]
    return torch.stack((pocket_height, wrist_height), dim=1)
    
    # root_height = vert[:, vi_mask[5], 1].unsqueeze(1) - ground
    # wrist_height = vert[:, vi_mask[0], 1].unsqueeze(1) - ground
    
    # return torch.stack((root_height, wrist_height), dim=1)

def _foot_contact(fp_list):
    """
    判断 n 帧中是否连续有至少一只脚接触地面。
    
    参数:
        fp_list (list of tuples): 包含 n 帧的接触概率，每一帧的格式为 (fp[0], fp[1])。
                                fp[0] 和 fp[1] 分别表示左右脚的接触概率或状态。
    
    返回:
        bool: 如果 n 帧中至少有一只脚接触地面，则返回 True；否则返回 False。
    """
    for fp in fp_list:
        if fp[0] or fp[1]:  # 判断当前帧是否有一只脚接触地面
            continue
        else:
            # 只要有一帧没有脚接触地面，返回 False
            return False
    return True  # 所有帧都有至少一只脚接触地面

def _foot_ground_probs(joint):
    """Compute foot-ground contact probabilities."""
    dist_lfeet = torch.norm(joint[1:, 10] - joint[:-1, 10], dim=1)
    dist_rfeet = torch.norm(joint[1:, 11] - joint[:-1, 11], dim=1)
    lfoot_contact = (dist_lfeet < 0.008).int()
    rfoot_contact = (dist_rfeet < 0.008).int()
    lfoot_contact = torch.cat((torch.zeros(1, dtype=torch.int), lfoot_contact))
    rfoot_contact = torch.cat((torch.zeros(1, dtype=torch.int), rfoot_contact))
    return torch.stack((lfoot_contact, rfoot_contact), dim=1)

def gen_amass_floor():
    def _foot_ground_probs(joint):
        """Compute foot-ground contact probabilities."""
        dist_lfeet = torch.norm(joint[1:, 10] - joint[:-1, 10], dim=1)
        dist_rfeet = torch.norm(joint[1:, 11] - joint[:-1, 11], dim=1)
        lfoot_contact = (dist_lfeet < 0.008).int()
        rfoot_contact = (dist_rfeet < 0.008).int()
        lfoot_contact = torch.cat((torch.zeros(1, dtype=torch.int), lfoot_contact))
        rfoot_contact = torch.cat((torch.zeros(1, dtype=torch.int), rfoot_contact))
        return torch.stack((lfoot_contact, rfoot_contact), dim=1)
    
    def _foot_pos(joint):
        # get foot position: [N, 2, 3]
        
        lfeet = joint[:, 10]
        rfeet = joint[:, 11]
        
        return torch.stack((lfeet, rfeet), dim=1)
    
    # enable skipping processed files
    try:
        processed = [fpath.name for fpath in (paths.processed_datasets).iterdir()]
    except FileNotFoundError:
        processed = []

    for ds_name in datasets.amass_datasets:

        data_pose, data_trans, data_beta, length = [], [], [], []
        npz_frames = []
        print("\rReading", ds_name)
        
        for npz_fname in tqdm(sorted(glob.glob(os.path.join(paths.raw_amass, ds_name, "*/*_poses.npz")))):
            
            try: cdata = np.load(npz_fname)
            except: continue

            framerate = int(cdata['mocap_framerate'])
            if framerate not in [120, 60, 59]:
                continue

            # enable downsampling
            step = max(1, round(framerate / TARGET_FPS))

            data_pose.extend(cdata['poses'][::step].astype(np.float32))
            data_trans.extend(cdata['trans'][::step].astype(np.float32))
            data_beta.append(cdata['betas'][:10])
            length.append(cdata['poses'][::step].shape[0])
            npz_frames.append(npz_fname)

        if len(data_pose) == 0:
            print(f"AMASS dataset, {ds_name} not supported")
            continue

        length = torch.tensor(length, dtype=torch.int)
        shape = torch.tensor(np.asarray(data_beta, np.float32))
        tran = torch.tensor(np.asarray(data_trans, np.float32))
        pose = torch.tensor(np.asarray(data_pose, np.float32)).view(-1, 52, 3)

        # include the left and right index fingers in the pose
        pose[:, 23] = pose[:, 37]     # right hand 
        pose = pose[:, :24].clone()   # only use body + right and left fingers

        # align AMASS global frame with DIP
        amass_rot = torch.tensor([[[1, 0, 0], [0, 0, 1], [0, -1, 0.]]])
        tran = amass_rot.matmul(tran.unsqueeze(-1)).view_as(tran)
        pose[:, 0] = math.rotation_matrix_to_axis_angle(
            amass_rot.matmul(math.axis_angle_to_rotation_matrix(pose[:, 0])))
        
        print("Synthesizing IMU accelerations and orientations")
        b = 0
        
        contact_num = 5
        
        out_pose, out_shape, out_tran, out_joint, out_vrot, out_vacc, out_contact = [], [], [], [], [], [], []
        out_ground, out_heights = [], []
        uneven=False
        for i, l in tqdm(list(enumerate(length))):
            if l <= 12: b += l; print("\tdiscard one sequence with length", l); continue
            p = math.axis_angle_to_rotation_matrix(pose[b:b + l]).view(-1, 24, 3, 3)

            _, joint = body_model.forward_kinematics(p, shape[i], tran[b:b + l], calc_mesh=False)
            
            fc_probs = _foot_ground_probs(joint).clone()
            
            f_min = _foot_min(joint)
            cur_ground = f_min[0].item()
            
            # 初始化地面高度, shape和f_y一样, 值全部为cur_ground
            ground = torch.full_like(f_min, cur_ground)
            
            for frame in range(l):
                g = f_min[frame]
                
                if frame >= contact_num:
                    fp_last_n = fc_probs[frame - contact_num: frame]
                else:
                    fp_last_n = fc_probs[:frame]
                
                ground[frame] = cur_ground
                
                # 如果当前帧至少有一个脚接触地面
                contact = _foot_contact(fp_last_n)
                residual = abs(g - cur_ground)
                if contact and residual > 1e-3:
                    ground[frame] = g
                    cur_ground = g
                    if residual > 0.3:
                        print("fname:", npz_frames[i], "residual:", residual)
            
            grot, joint, vert = body_model.forward_kinematics(p, shape[i], tran[b:b + l], calc_mesh=True)

            out_pose.append(p.clone())  # N, 24, 3, 3
            out_tran.append(tran[b:b + l].clone())  # N, 3
            out_shape.append(shape[i].clone())  # 10
            out_joint.append(joint[:, :24].contiguous().clone())  # N, 24, 3
            out_vacc.append(_syn_acc(vert[:, vi_mask]))  # N, 6, 3
            out_contact.append(_foot_ground_probs(joint).clone()) # N, 2
            out_vrot.append(grot[:, ji_mask])  # N, 6, 3, 3
            # out_ground.append(ground)
            out_heights.append(_get_heights(vert, ground).squeeze())
            
            b += l
        
        print("Saving...")
        # print(out_vacc.shape, out_pose.shape)
        data = {
            'joint': out_joint,
            'pose': out_pose,
            'shape': out_shape,
            'tran': out_tran,
            'acc': out_vacc,
            'ori': out_vrot,
            'contact': out_contact,
            # 'ground': out_ground,
            'heights': out_heights
        }
        data_path = paths.processed_datasets / f"{ds_name}.pt"
        torch.save(data, data_path)
        print(f"Synthetic AMASS dataset is saved at: {data_path}")

def process_amass(dataset=None):
    def _foot_ground_probs(joint):
        """Compute foot-ground contact probabilities."""
        dist_lfeet = torch.norm(joint[1:, 10] - joint[:-1, 10], dim=1)
        dist_rfeet = torch.norm(joint[1:, 11] - joint[:-1, 11], dim=1)
        lfoot_contact = (dist_lfeet < 0.008).int()
        rfoot_contact = (dist_rfeet < 0.008).int()
        lfoot_contact = torch.cat((torch.zeros(1, dtype=torch.int), lfoot_contact))
        rfoot_contact = torch.cat((torch.zeros(1, dtype=torch.int), rfoot_contact))
        return torch.stack((lfoot_contact, rfoot_contact), dim=1)
    
    def _get_pocket_height(vert):
        rp_height = vert[vi_mask[3], 1]
        min_height = torch.min(vert[:, 1], dim=0).values
        
        return rp_height - min_height
    
    def _get_scale(shape):
        shape = torch.tensor(shape, dtype=torch.float32).unsqueeze(0)
        
        zero_pose = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(1, 24, 1, 1)
        zero_shape = torch.zeros(10).unsqueeze(0)
        
        _, _, vert_zero = body_model.forward_kinematics(zero_pose, zero_shape, calc_mesh=True)
        _, _, vert_shape = body_model.forward_kinematics(zero_pose, shape, calc_mesh=True)
        
        height_zero = _get_pocket_height(vert_zero.squeeze(0))
        height_shape = _get_pocket_height(vert_shape.squeeze(0))
        
        scale = height_shape / height_zero
        
        return scale
    
    # enable skipping processed files
    try:
        processed = [fpath.name for fpath in (paths.processed_datasets).iterdir()]
    except FileNotFoundError:
        processed = []

    if dataset is not None:
        processed = []
    
    for ds_name in datasets.amass_datasets:
        
        if dataset:
            if ds_name != dataset:
                continue
        
        # skip processed 
        if f"{ds_name}.pt" in processed:
            continue

        data_pose, data_trans, data_beta, length = [], [], [], []
        scale_data = []
        
        print("\rReading", ds_name)

        for npz_fname in tqdm(sorted(glob.glob(os.path.join(paths.raw_amass, ds_name, "*/*_poses.npz")))):
            
            try: cdata = np.load(npz_fname)
            except: continue

            framerate = int(cdata['mocap_framerate'])
            if framerate not in [120, 60, 59]:
                continue

            # enable downsampling
            step = max(1, round(framerate / TARGET_FPS))

            data_pose.extend(cdata['poses'][::step].astype(np.float32))
            data_trans.extend(cdata['trans'][::step].astype(np.float32))
            data_beta.append(cdata['betas'][:10])
            
            shape = cdata['betas'][:10]
            scale_data.append(_get_scale(shape))
            
            length.append(cdata['poses'][::step].shape[0])

        if len(data_pose) == 0:
            print(f"AMASS dataset, {ds_name} not supported")
            continue

        length = torch.tensor(length, dtype=torch.int)
        shape = torch.tensor(np.asarray(data_beta, np.float32))
        tran = torch.tensor(np.asarray(data_trans, np.float32))
        pose = torch.tensor(np.asarray(data_pose, np.float32)).view(-1, 52, 3)
        scale = torch.tensor(np.asarray(scale_data, np.float32))

        # include the left and right index fingers in the pose
        pose[:, 23] = pose[:, 37]     # right hand 
        pose = pose[:, :24].clone()   # only use body + right and left fingers

        # align AMASS global frame with DIP
        amass_rot = torch.tensor([[[1, 0, 0], [0, 0, 1], [0, -1, 0.]]])
        tran = amass_rot.matmul(tran.unsqueeze(-1)).view_as(tran)
        pose[:, 0] = math.rotation_matrix_to_axis_angle(
            amass_rot.matmul(math.axis_angle_to_rotation_matrix(pose[:, 0])))

        print("Synthesizing IMU accelerations and orientations")
        b = 0
        out_pose, out_shape, out_tran, out_joint, out_vrot, out_vacc, out_contact = [], [], [], [], [], [], []
        out_rheight, out_scale = [], []
        for i, l in tqdm(list(enumerate(length))):
            if l <= 12: b += l; print("\tdiscard one sequence with length", l); continue
            p = math.axis_angle_to_rotation_matrix(pose[b:b + l]).view(-1, 24, 3, 3)
            grot, joint, vert = body_model.forward_kinematics(p, shape[i], tran[b:b + l], calc_mesh=True)

            out_pose.append(p.clone())  # N, 24, 3, 3
            out_tran.append(tran[b:b + l].clone())  # N, 3
            out_shape.append(shape[i].clone())  # 10
            out_joint.append(joint[:, :24].contiguous().clone())  # N, 24, 3
            out_vacc.append(_syn_acc(vert[:, vi_mask]))  # N, 6, 3
            out_contact.append(_foot_ground_probs(joint).clone()) # N, 2
            out_vrot.append(grot[:, ji_mask])  # N, 6, 3, 3
            out_rheight.append(_relative_height(vert))  # N
            out_scale.append(scale[i].clone())  # N
            b += l

        print("Saving...")
        # print(out_vacc.shape, out_pose.shape)
        data = {
            'joint': out_joint,
            'pose': out_pose,
            'shape': out_shape,
            'tran': out_tran,
            'acc': out_vacc,
            'ori': out_vrot,
            'contact': out_contact,
            'rheight': out_rheight,
            'scale': out_scale
        }
        if dataset:
            data_path = paths.processed_datasets / "debug" / f"{ds_name}.pt"
        else:
            data_path = paths.processed_datasets / f"{ds_name}.pt"
        torch.save(data, data_path)
        print(f"Synthetic AMASS dataset is saved at: {data_path}")

def process_totalcapture():
    """Preprocess TotalCapture dataset for testing."""

    inches_to_meters = 0.0254
    pos_file = 'gt_skel_gbl_pos.txt'
    ori_file = 'gt_skel_gbl_ori.txt'

    subjects = ['S1', 'S2', 'S3', 'S4', 'S5']

    # # Load poses from processed AMASS dataset
    # amass_tc = torch.load(os.path.join(paths.processed_datasets, "AMASS", "TotalCapture", "pose.pt"))
    # tc_poses = {pose.shape[0]: pose for pose in amass_tc}

    processed, failed_to_process = [], []
    accs, oris, poses, trans = [], [], [], []
    heights = []
    rheights = []
    
    # downsampling
    step = max(1, round(60 / TARGET_FPS))
    
    for file in sorted(os.listdir(paths.calibrated_totalcapture)):
        if not file.endswith(".pkl") or ('s5' in file and 'acting3' in file) or not any(file.startswith(s.lower()) for s in subjects):
            continue

        data = pickle.load(open(os.path.join(paths.calibrated_totalcapture, file), 'rb'), encoding='latin1')
        ori = torch.from_numpy(data['ori']).float() # [N, 6, 3, 3]
        acc = torch.from_numpy(data['acc']).float() # [N, 6, 3]

        # load pose from dip calibration
        pose = torch.from_numpy(data['gt']).float().view(-1, 24, 3)
        
        # downsample
        acc = acc[::step].contiguous()
        ori = ori[::step].contiguous()
        pose = pose[::step].contiguous()
        
        # # Load pose data from AMASS
        # try: 
        #     name_split = file.split("_")
        #     subject, activity = name_split[0], name_split[1].split(".")[0]
        #     pose_npz = np.load(os.path.join(paths.raw_amass, "TotalCapture", subject, f"{activity}_poses.npz"))
        #     pose = torch.from_numpy(pose_npz['poses']).float().view(-1, 52, 3)
            
        #     # include the left and right index fingers in the pose
        #     pose[:, 23] = pose[:, 37]     # right hand
        #     pose = pose[:, :24].clone()   # only use body + right and left fingers
            
        #     # align AMASS global frame with DIP
        #     amass_rot = torch.tensor([[[1, 0, 0], [0, 0, 1], [0, -1, 0.]]])
        #     pose[:, 0] = math.rotation_matrix_to_axis_angle(
        #         amass_rot.matmul(math.axis_angle_to_rotation_matrix(pose[:, 0])))
            
        # except:
        #     failed_to_process.append(f"{subject}_{activity}")
        #     print(f"Failed to Process: {file}")
        #     continue

        # pose = tc_poses[pose.shape[0]]
    
        # acc/ori and gt pose do not match in the dataset
        if acc.shape[0] < pose.shape[0]:
            pose = pose[:acc.shape[0]]
        elif acc.shape[0] > pose.shape[0]:
            acc = acc[:pose.shape[0]]
            ori = ori[:pose.shape[0]]
        
        # convert axis-angle to rotation matrix
        pose = math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)

        assert acc.shape[0] == ori.shape[0] and ori.shape[0] == pose.shape[0]
        accs.append(acc)    # N, 6, 3
        oris.append(ori)    # N, 6, 3, 3
        poses.append(pose)  # N, 24, 3, 3

        processed.append(file)
    
    for subject_name in subjects:
        for motion_name in sorted(os.listdir(os.path.join(paths.raw_totalcapture_official, subject_name))):
            if (subject_name == 'S5' and motion_name == 'acting3') or motion_name.startswith(".") or (f"{subject_name.lower()}_{motion_name}" in failed_to_process):
                continue   # no SMPL poses

            f = open(os.path.join(paths.raw_totalcapture_official, subject_name, motion_name, pos_file))
            line = f.readline().split('\t')
            index = torch.tensor([line.index(_) for _ in ['LeftFoot', 'RightFoot', 'Spine']])
            pos = []
            while line:
                line = f.readline()
                pos.append(torch.tensor([[float(_) for _ in p.split(' ')] for p in line.split('\t')[:-1]]))
            pos = torch.stack(pos[:-1])[:, index] * inches_to_meters
            pos[:, :, 0].neg_()
            pos[:, :, 2].neg_()
            trans.append(pos[:, 2] - pos[:1, 2])   # N, 3

    # downsample
    trans = [t[::step] for t in trans]
    
    # match trans with poses
    for i in range(len(accs)):
        if accs[i].shape[0] < trans[i].shape[0]:
            trans[i] = trans[i][:accs[i].shape[0]]
        assert trans[i].shape[0] == accs[i].shape[0]

    # remove acceleration bias and add relative height
    v_accs, v_oris = [], []
    for iacc, pose, tran in zip(accs, poses, trans):
        pose = pose.view(-1, 24, 3, 3)
        grot, joint, vert = body_model.forward_kinematics(pose, tran=tran, calc_mesh=True)
        vacc = _syn_acc(vert[:, vi_mask])
        rheights.append(_relative_height(vert))
        
        ground = _foot_min(joint, fix=False)
        
        heights.append(_get_heights(vert, ground).squeeze())
        
        v_accs.append(vacc)
        v_oris.append(grot[:, ji_mask])
        
        for imu_id in range(6):
            for i in range(3):
                d = -iacc[:, imu_id, i].mean() + vacc[:, imu_id, i].mean()
                iacc[:, imu_id, i] += d

    data = {
        'acc': accs,
        'ori': oris,
        'pose': poses,
        'tran': trans,
        'rheight': rheights,
        'heights': heights
    }
    
    data_path = paths.eval_dir / "totalcapture.pt"
    torch.save(data, data_path)
    print("Preprocessed TotalCapture dataset is saved at:", data_path)

def process_dipimu(split="test"):
    """Preprocess DIP for finetuning and evaluation."""
    imu_mask = [7, 8, 9, 10, 0, 2]

    test_split = ['s_09', 's_10']
    train_split = ['s_01', 's_02', 's_03', 's_04', 's_05', 's_06', 's_07', 's_08']
    subjects = train_split if split == "train" else test_split
     
    # left wrist, right wrist, left thigh, right thigh, head, pelvis
    vi_mask = torch.tensor([1961, 5424, 876, 4362, 411, 3021])
    ji_mask = torch.tensor([18, 19, 1, 2, 15, 0])

    # enable downsampling
    step = max(1, round(60 / TARGET_FPS))

    accs, oris, poses, trans, shapes, joints = [], [], [], [], [], []
    rheights = []
    heights = []

    for subject_name in subjects:
        for motion_name in os.listdir(os.path.join(paths.raw_dip, subject_name)):
            try:
                path = os.path.join(paths.raw_dip, subject_name, motion_name)
                print(f"Processing: {subject_name}/{motion_name}")
                data = pickle.load(open(path, 'rb'), encoding='latin1')
                acc = torch.from_numpy(data['imu_acc'][:, imu_mask]).float()
                ori = torch.from_numpy(data['imu_ori'][:, imu_mask]).float()
                pose = torch.from_numpy(data['gt']).float()

                # fill nan with nearest neighbors
                for _ in range(4):
                    acc[1:].masked_scatter_(torch.isnan(acc[1:]), acc[:-1][torch.isnan(acc[1:])])
                    ori[1:].masked_scatter_(torch.isnan(ori[1:]), ori[:-1][torch.isnan(ori[1:])])
                    acc[:-1].masked_scatter_(torch.isnan(acc[:-1]), acc[1:][torch.isnan(acc[:-1])])
                    ori[:-1].masked_scatter_(torch.isnan(ori[:-1]), ori[1:][torch.isnan(ori[:-1])])

                # enable downsampling
                acc = acc[6:-6:step].contiguous()
                ori = ori[6:-6:step].contiguous()
                pose = pose[6:-6:step].contiguous()

                shape = torch.ones((10))
                tran = torch.zeros(pose.shape[0], 3) # dip-imu does not contain translations
                if torch.isnan(acc).sum() == 0 and torch.isnan(ori).sum() == 0 and torch.isnan(pose).sum() == 0:
                    accs.append(acc.clone())
                    oris.append(ori.clone())
                    trans.append(tran.clone())  
                    shapes.append(shape.clone()) # default shape
                    
                    # forward kinematics to get the joint position
                    p = math.axis_angle_to_rotation_matrix(pose).reshape(-1, 24, 3, 3)
                    grot, joint, vert = body_model.forward_kinematics(p, shape, tran, calc_mesh=True)
                    poses.append(p.clone())
                    joints.append(joint)
                    rheights.append(_relative_height(vert))
                    
                    # contact_num = 5
                    # fc_probs = _foot_ground_probs(joint).clone()
                    # f_min = _foot_min(joint, fix=False)
                    # cur_ground = f_min[0].item()
                    
                    # ground = torch.full_like(f_min, cur_ground)
                    
                    # for frame in range(fc_probs.shape[0]):
                    #     g = f_min[frame].item()
                        
                    #     if frame >= contact_num:
                    #         fp_last_n = fc_probs[frame - contact_num:frame]
                    #     else:
                    #         fp_last_n = fc_probs[:frame]
                            
                    #     ground[frame] = cur_ground
                        
                    #     contact = _foot_contact(fp_last_n)
                    #     residual = abs(g - cur_ground)
                    #     if contact and residual > 1e-3:
                    #         ground[frame] = g
                    #         cur_ground = g
                    #         if residual > 0.3:
                    #             print(f"Warning: {subject_name}/{motion_name} has a large residual: {residual}")
                            
                    ground = _foot_min(joint, fix=False)
                    heights.append(_get_heights(vert, ground))
                    
                else:
                    print(f"DIP-IMU: {subject_name}/{motion_name} has too much nan! Discard!")
            except Exception as e:
                print(f"Error processing the file: {path}.", e)


    print("Saving...")
    data = {
        'joint': joints,
        'pose': poses,
        'shape': shapes,
        'tran': trans,
        'acc': accs,
        'ori': oris,
        'rheight': rheights,
        'heights': heights
    }
    data_path = paths.eval_dir / f"dip_{split}.pt"
    torch.save(data, data_path)
    print(f"Preprocessed DIP-IMU dataset is saved at: {data_path}")

def process_imuposer(split: str="train"):
    """Preprocess the IMUPoser dataset"""

    train_split = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8']
    test_split = ['P9', 'P10']
    subjects = train_split if split == "train" else test_split

    accs, oris, poses, trans = [], [], [], []
    rheights = []
    heights = []
    
    for pid_path in sorted(paths.raw_imuposer.iterdir()):
        if pid_path.name not in subjects:
            continue

        print(f"Processing: {pid_path.name}")
        for fpath in sorted(pid_path.iterdir()):
            with open(fpath, "rb") as f: 
                fdata = pickle.load(f)
                
                acc = fdata['imu'][:, :5*3].view(-1, 5, 3)
                ori = fdata['imu'][:, 5*3:].view(-1, 5, 3, 3)
                pose = math.axis_angle_to_rotation_matrix(fdata['pose']).view(-1, 24, 3, 3)
                tran = fdata['trans'].to(torch.float32)
                
                 # align IMUPoser global fame with DIP
                rot = torch.tensor([[[-1, 0, 0], [0, 0, 1], [0, 1, 0.]]])
                pose[:, 0] = rot.matmul(pose[:, 0])
                tran = tran.matmul(rot.squeeze())

                # ensure sizes are consistent
                assert tran.shape[0] == pose.shape[0]
                
                grot, joint, vert = body_model.forward_kinematics(pose, tran=tran, calc_mesh=True)
                
                # acc = _syn_acc(vert[:, vi_mask])
                # ori = grot[:, ji_mask]

                accs.append(acc)    # N, 5, 3
                oris.append(ori)    # N, 5, 3, 3
                poses.append(pose)  # N, 24, 3, 3
                trans.append(tran)  # N, 3
                rheights.append(_relative_height(vert))  # N
                
                ground = _foot_min(joint, fix=False)
                heights.append(_get_heights(vert, ground))

    print(f"# Data Processed: {len(accs)}")
    data = {
        'acc': accs,
        'ori': oris,
        'pose': poses,
        'tran': trans,
        'rheight': rheights,
        'heights': heights
    }
    data_path = paths.eval_dir / f"imuposer_{split}.pt"
    torch.save(data, data_path)

def create_directories():
    paths.processed_datasets.mkdir(exist_ok=True, parents=True)
    paths.eval_dir.mkdir(exist_ok=True, parents=True)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="amass")
    args = parser.parse_args()

    # create dataset directories
    create_directories()

    # process datasets
    if args.dataset == "floor":
        gen_amass_floor()
    elif args.dataset == "amass":
        process_amass()
    elif args.dataset == "totalcapture":
        process_totalcapture()
    elif args.dataset == "imuposer":
        process_imuposer(split="train")
        process_imuposer(split="test")
    elif args.dataset == "dip":
        process_dipimu(split="train")
        process_dipimu(split="test")
    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")
