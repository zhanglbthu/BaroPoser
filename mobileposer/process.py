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
import articulate as art
from utils.data_utils import _foot_contact, _get_heights, _foot_min, _get_ground
import sys

# left wrist, right wrist, left thigh, right thigh, head, pelvis, left shank, right shank
vi_mask = torch.tensor([1961, 5424, 876, 4362, 411, 3021, 1176, 4662])
ji_mask = torch.tensor([18, 19, 1, 2, 15, 0, 4, 5])
body_model = ParametricModel(paths.smpl_file)

def _syn_acc(v, smooth_n=4, fps=60):
    """Synthesize accelerations from vertex positions."""
    mid = smooth_n // 2
    acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * (fps ** 2) for i in range(0, v.shape[0] - 2)])
    acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
    if mid != 0:
        acc[smooth_n:-smooth_n] = torch.stack(
            [(v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n]) * (fps ** 2) / smooth_n ** 2
             for i in range(0, v.shape[0] - smooth_n * 2)])
    return acc

def _foot_ground_probs(joint):
    """Compute foot-ground contact probabilities."""
    dist_lfeet = torch.norm(joint[1:, 10] - joint[:-1, 10], dim=1)
    dist_rfeet = torch.norm(joint[1:, 11] - joint[:-1, 11], dim=1)
    lfoot_contact = (dist_lfeet < 0.008).int()
    rfoot_contact = (dist_rfeet < 0.008).int()
    lfoot_contact = torch.cat((torch.zeros(1, dtype=torch.int), lfoot_contact))
    rfoot_contact = torch.cat((torch.zeros(1, dtype=torch.int), rfoot_contact))
    return torch.stack((lfoot_contact, rfoot_contact), dim=1)

def process_amass(dataset=None, heights: bool=False):
    # enable skipping processed files
    try:
        processed = [fpath.name for fpath in (paths.processed_datasets).iterdir()]
    except FileNotFoundError:
        processed = []

    if dataset is not None:
        processed = []
    
    for ds_name in datasets.amass_datasets:
        # skip processed 
        if dataset is not None and dataset != ds_name:
            continue
        
        if f"{ds_name}.pt" in processed:
            continue

        data_pose, data_trans, data_beta, length = [], [], [], []
        
        print("\rReading", ds_name)

        for npz_fname in tqdm(sorted(glob.glob(os.path.join(paths.raw_amass, ds_name, "*/*_poses.npz")))):

            try: cdata = np.load(npz_fname)
            except: continue

            framerate = int(cdata['mocap_framerate'])

            # enable downsampling
            if framerate == 120: step = 2
            elif framerate == 60 or framerate == 59: step = 1
            else: continue

            data_pose.extend(cdata['poses'][::step].astype(np.float32))
            data_trans.extend(cdata['trans'][::step].astype(np.float32))
            data_beta.append(cdata['betas'][:10])
            length.append(cdata['poses'][::step].shape[0])
        
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
        out_pose, out_shape, out_tran, out_joint, out_vrot, out_vacc, out_contact = [], [], [], [], [], [], []
        out_ground, out_heights = [], []
        
        for i, l in tqdm(list(enumerate(length))):
            if l <= 12: b += l; print("\tdiscard one sequence with length", l); continue
            p = math.axis_angle_to_rotation_matrix(pose[b:b + l]).view(-1, 24, 3, 3)
            
            if heights:
                _, joint = body_model.forward_kinematics(p, shape[i], tran[b:b + l], calc_mesh=False)
                fc_probs = _foot_ground_probs(joint).clone()
                ground = _get_ground(joint, fc_probs, l)
            else:
                ground = torch.zeros((l, 1))
            
            grot, joint, vert = body_model.forward_kinematics(p, shape[i], tran[b:b + l], calc_mesh=True)
            
            out_pose.append(p.clone())  # N, 24, 3, 3
            out_tran.append(tran[b:b + l].clone())  # N, 3
            out_shape.append(shape[i].clone())  # 10
            out_joint.append(joint[:, :24].contiguous().clone())  # N, 24, 3
            out_vacc.append(_syn_acc(vert[:, vi_mask]))  # N, 6, 3
            out_contact.append(_foot_ground_probs(joint).clone()) # N, 2
            out_vrot.append(grot[:, ji_mask])  # N, 6, 3, 3
            out_ground.append(ground.clone())  # N, 1
            out_heights.append(_get_heights(vert, ground, vi_mask))  # N, 2
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
            'ground': out_ground,
            'heights': out_heights
        }
        if dataset:
            data_path = paths.processed_datasets / "eval" / f"{ds_name}.pt"
            os.makedirs(data_path.parent, exist_ok=True)
            
        else:
            data_path = paths.processed_datasets / f"{ds_name}.pt"
        torch.save(data, data_path)
        print(f"Synthetic AMASS dataset is saved at: {data_path}")

def process_totalcapture_raw(debug=False):
    
    # if paths.eval_dir / 'totalcapture_raw.pt' exists, skip processing
    if (paths.eval_dir / 'totalcapture_raw.pt').exists():
        return
    
    print('======================== Processing TotalCapture Dataset ========================')
    joint_names = ['L_LowArm', 'R_LowArm', 'L_UpLeg', 'R_UpLeg', 'Head', 'Pelvis', 'L_LowLeg', 'R_LowLeg']
    
    vicon_gt_dir = paths.vicon_gt_dir
    imu_dir = paths.imu_dir
    calib_dir = paths.calib_dir
    AMASS_smpl_dir = paths.AMASS_smpl_dir
    DIP_smpl_dir = paths.DIP_smpl_dir
    
    data = {'name': [], 'RIM': [], 'RSB': [], 'RIS': [], 'aS': [], 'wS': [], 'mS': [], 'tran': [], 'AMASS_pose': [], 'DIP_pose': []}
    n_extracted_imus = len(joint_names)

    for subject_name in ['s1', 's2', 's3', 's4', 's5']:
        for action_name in sorted(os.listdir(os.path.join(imu_dir, subject_name))):
            # read imu file
            f = open(os.path.join(imu_dir, subject_name, action_name), 'r')
            line = f.readline().split('\t')
            n_sensors, n_frames = int(line[0]), int(line[1])
            R = torch.zeros(n_frames, n_extracted_imus, 4)
            a = torch.zeros(n_frames, n_extracted_imus, 3)
            w = torch.zeros(n_frames, n_extracted_imus, 3)
            m = torch.zeros(n_frames, n_extracted_imus, 3)
            for i in range(n_frames):
                assert int(f.readline()) == i + 1, 'parse imu file error'
                for _ in range(n_sensors):
                    line = f.readline().split('\t')
                    if line[0] in joint_names:
                        j = joint_names.index(line[0])
                        R[i, j] = torch.tensor([float(_) for _ in line[1:5]])  # wxyz
                        a[i, j] = torch.tensor([float(_) for _ in line[5:8]])
                        w[i, j] = torch.tensor([float(_) for _ in line[8:11]])
                        m[i, j] = torch.tensor([float(_) for _ in line[11:14]])
            R = art.math.quaternion_to_rotation_matrix(R).view(-1, n_extracted_imus, 3, 3)

            # read calibration file
            name = subject_name + '_' + action_name.split('_')[0].lower()
            RSB = torch.zeros(n_extracted_imus, 3, 3)
            RIM = torch.zeros(n_extracted_imus, 3, 3)
            with open(os.path.join(calib_dir, subject_name, name + '_calib_imu_bone.txt'), 'r') as f:
                n_sensors = int(f.readline())
                for _ in range(n_sensors):
                    line = f.readline().split()
                    if line[0] in joint_names:
                        j = joint_names.index(line[0])
                        q = torch.tensor([float(line[4]), float(line[1]), float(line[2]), float(line[3])])  # wxyz
                        RSB[j] = art.math.quaternion_to_rotation_matrix(q)[0].t()
            with open(os.path.join(calib_dir, subject_name, name + '_calib_imu_ref.txt'), 'r') as f:
                n_sensors = int(f.readline())
                for _ in range(n_sensors):
                    line = f.readline().split()
                    if line[0] in joint_names:
                        j = joint_names.index(line[0])
                        q = torch.tensor([float(line[4]), float(line[1]), float(line[2]), float(line[3])])  # wxyz
                        RIM[j] = art.math.quaternion_to_rotation_matrix(q)[0].t()
            RSB = RSB.matmul(torch.tensor([[-1, 0, 0], [0, 0, -1], [0, -1, 0.]]))  # change bone frame to SMPL
            RIM = RIM.matmul(torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1.]]))   # change global frame to SMPL

            # read root translation
            tran = []
            with open(os.path.join(vicon_gt_dir, subject_name.upper(), action_name.split('_')[0].lower(), 'gt_skel_gbl_pos.txt')) as f:
                idx = f.readline().split('\t').index('Hips')
                while True:
                    line = f.readline()
                    if line == '':
                        break
                    t = [float(_) * 0.0254 for _ in line.split('\t')[idx].split(' ')]   # inches_to_meters
                    tran.append([-t[0], t[1], -t[2]])
            tran = torch.tensor(tran)

            # read SMPL pose parameters calculated by AMASS
            f = os.path.join(AMASS_smpl_dir, subject_name, action_name.split('_')[0].lower() + '_poses.npz')
            AMASS_pose = None
            if os.path.exists(f):
                d = np.load(f)
                AMASS_pose = torch.from_numpy(d['poses'])[:, :72].float()
                root_rot = art.math.axis_angle_to_rotation_matrix(AMASS_pose[:, :3])
                root_rot = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0.]]).matmul(root_rot)  # align global frame
                root_rot = art.math.rotation_matrix_to_axis_angle(root_rot)
                AMASS_pose[:, :3] = root_rot
                AMASS_pose[:, 66:] = 0  # hand

            # read SMPL pose parameters calculated by DIP
            f = os.path.join(DIP_smpl_dir, name + '.pkl')
            DIP_pose = None
            if os.path.exists(f):
                d = pickle.load(open(f, 'rb'), encoding='latin1')
                DIP_pose = torch.from_numpy(d['gt']).float()

            # align data
            n_aligned_frames = min(n_frames, tran.shape[0], AMASS_pose.shape[0] if AMASS_pose is not None else 1e8, DIP_pose.shape[0] if DIP_pose is not None else 1e8)
            if AMASS_pose is not None:
                AMASS_pose = AMASS_pose[-n_aligned_frames:]
            if DIP_pose is not None:
                DIP_pose = DIP_pose[-n_aligned_frames:]
            tran = tran[-n_aligned_frames:] - tran[-n_aligned_frames]
            R = R[-n_aligned_frames:]
            a = a[-n_aligned_frames:]
            w = w[-n_aligned_frames:]
            m = m[-n_aligned_frames:]

            # validate data (for debug purpose)
            if debug and DIP_pose is not None:
                model = art.ParametricModel(paths.smpl_file)
                DIP_pose = art.math.axis_angle_to_rotation_matrix(DIP_pose).view(-1, 24, 3, 3)
                syn_RMB = model.forward_kinematics_R(DIP_pose)[:, [18, 19, 1, 2, 15, 0, 4, 5]]
                real_RMB = RIM.transpose(1, 2).matmul(R).matmul(RSB)
                real_aM = RIM.transpose(1, 2).matmul(R).matmul(a.unsqueeze(-1)).squeeze(-1)
                print('real-syn imu ori err:', art.math.radian_to_degree(art.math.angle_between(real_RMB, syn_RMB).mean()))
                print('mean acc in M:', real_aM.mean(dim=(0, 1)))   # (0, +g, 0)

            # save results
            data['name'].append(name)
            data['RIM'].append(RIM)
            data['RSB'].append(RSB)
            data['RIS'].append(R)
            data['aS'].append(a)
            data['wS'].append(w)
            data['mS'].append(m)
            data['tran'].append(tran)
            data['AMASS_pose'].append(AMASS_pose)
            data['DIP_pose'].append(DIP_pose)
            print('Finish Processing %s' % name, '(no AMASS pose)' if AMASS_pose is None else '', '(no DIP pose)' if DIP_pose is None else '')

    os.makedirs(paths.eval_dir, exist_ok=True)
    torch.save(data, paths.eval_dir / 'totalcapture_raw.pt')

def process_totalcapture_from_raw(heights: bool=False):
    """Preprocess TotalCapture dataset for testing."""
    r"""
        totalcapture data.pt
            data['name'].append(name)
            data['RIM'].append(RIM)
            data['RSB'].append(RSB)
            data['RIS'].append(R)
            data['aS'].append(a)
            data['wS'].append(w)
            data['mS'].append(m)
            data['tran'].append(tran)
            data['AMASS_pose'].append(AMASS_pose)
            data['DIP_pose'].append(DIP_pose)
    """
    
    accs, oris, poses, trans, joints = [], [], [], [], []
    raw_data_path = paths.eval_dir / 'totalcapture_raw.pt'
    data = torch.load(raw_data_path)
    
    RIM = data['RIM']
    RSB = data['RSB']
    oris_raw = data['RIS'] # list of tensor in shape [-1,3,3]
    accs_raw = data['aS'] # list of tensor in shape [-1,3]
    real_RMB = [RIM[i].transpose(1, 2).matmul(oris_raw[i]).matmul(RSB[i]) for i in range(len(oris_raw))]
    real_aM = [RIM[i].transpose(1, 2).matmul(oris_raw[i]).matmul((accs_raw[i]).unsqueeze(-1)).squeeze(-1)+torch.tensor([0,-9.8,0]) for i in range(len(accs_raw))]
    poses_raw = data['DIP_pose']  # list of tensor in shape (-1, 24, 3, 3)
    trans_raw = data['tran']
    
    # calculate joint positions
    idx = 0
    for pose, tran in zip(poses_raw, trans_raw):
        if pose is not None and tran is not None:
            _, joint = body_model.forward_kinematics(pose, tran=tran)
            joints.append(joint)
            accs.append(real_aM[idx])
            oris.append(real_RMB[idx])

            poses.append(pose.view(-1, 24, 3, 3))
            trans.append(tran)
        idx += 1
    
    # print totalcapture length
    print(f"TotalCapture dataset length: {len(poses)}")
    
    # remove acceleration bias
    print("Removing acceleration bias and add height to the dataset")
    out_ground, out_heights = [], []
    vori = []
    for iacc, pose, tran in tqdm(zip(accs, poses, trans)):
        grot, joint, vert = body_model.forward_kinematics(pose, tran=tran, calc_mesh=True)
        
        if heights:
            fc_probs = _foot_ground_probs(joint).clone()
            ground = _get_ground(joint, fc_probs, pose.shape[0])
        else:
            ground = torch.zeros((pose.shape[0], 1))
            
        out_ground.append(ground)
        out_heights.append(_get_heights(vert, ground, vi_mask))
        vori.append(grot[:, ji_mask])
        vacc = _syn_acc(vert[:, vi_mask])
        for imu_id in range(6):
            for i in range(3):
                d = -iacc[:, imu_id, i].mean() + vacc[:, imu_id, i].mean()
                iacc[:, imu_id, i] += d
    
    data = {
        'joint': joints,
        'pose': poses,
        'tran': trans,
        'acc': accs,
        'ori': oris,
        # 'ori': vori,
        'ground': out_ground,
        'heights': out_heights
    }
    data_path = paths.eval_dir / 'totalcapture.pt'
    torch.save(data, data_path)
    print(f"Preprocessed TotalCapture dataset is saved at: {data_path}")

def process_dipimu(split="test", heights: bool=False):
    """Preprocess DIP for finetuning and evaluation."""
    imu_mask = [7, 8, 9, 10, 0, 2, 11, 12]

    test_split = ['s_09', 's_10']
    train_split = ['s_01', 's_02', 's_03', 's_04', 's_05', 's_06', 's_07', 's_08']
    subjects = train_split if split == "train" else test_split

    accs, oris, poses, trans, shapes, joints = [], [], [], [], [], []
    out_ground, out_heights = [], []

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

                shape = torch.ones((10))
                tran = torch.zeros(pose.shape[0], 3) # dip-imu does not contain translations
                if torch.isnan(acc).sum() == 0 and torch.isnan(ori).sum() == 0 and torch.isnan(pose).sum() == 0:
                    accs.append(acc.clone())
                    oris.append(ori.clone())
                    trans.append(tran.clone())  
                    shapes.append(shape.clone()) # default shape
                    
                    # forward kinematics to get the joint position
                    p = math.axis_angle_to_rotation_matrix(pose).reshape(-1, 24, 3, 3)
                    _, joint, vert = body_model.forward_kinematics(p, shape, tran, calc_mesh=True)
                    poses.append(p.clone())
                    joints.append(joint)
                    
                    if heights:
                        ground = _foot_min(joint)
                    else:
                        ground = torch.zeros((p.shape[0], 1))
                        
                    out_ground.append(ground)
                    out_heights.append(_get_heights(vert, ground, vi_mask))
                    
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
        'ground': out_ground,
        'heights': out_heights
    }
    data_path = paths.processed_datasets / 'eval' / f"dip_{split}.pt"
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
    grounds = []
    
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
                
                print(f"frames: {pose.shape[0]}")
                
                grot, joint, vert = body_model.forward_kinematics(pose, tran=tran, calc_mesh=True)
                
                ground = _foot_min(joint)

                accs.append(acc)    # N, 5, 3
                # accs.append(_syn_acc(vert[:, vi_mask], fps=25))  # N, 5, 3
                oris.append(ori)    # N, 5, 3, 3
                poses.append(pose)  # N, 24, 3, 3
                trans.append(tran)  # N, 3
                grounds.append(ground) # N, 1
                heights.append(_get_heights(vert, ground, vi_mask))  # N, 2

    print(f"# Data Processed: {len(accs)}")
    data = {
        'acc': accs,
        'ori': oris,
        'pose': poses,
        'tran': trans,
        'ground': grounds,
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
    parser.add_argument("--heights", action="store_true")
    args = parser.parse_args()

    # create dataset directories
    create_directories()
    
    if args.heights:
        print("Including heights in the dataset.")
    
    if args.dataset == "amass":
        process_amass(heights=args.heights)
    elif args.dataset == "totalcapture":
        # process_totalcapture_raw(debug=True)
        process_totalcapture_from_raw(heights=args.heights)
    elif args.dataset == "imuposer":
        process_imuposer(split="train")
        process_imuposer(split="test")
    elif args.dataset == "dip":
        process_dipimu(split="train", heights=args.heights)
        process_dipimu(split="test", heights=args.heights)
    elif args.dataset in datasets.amass_datasets:
        process_amass(dataset=args.dataset, heights=args.heights)
    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")
