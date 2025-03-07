import os
import torch
import torch.nn as nn
import lightning as L
from torch.nn import functional as F
import numpy as np
import math
from mobileposer.config import *
from mobileposer.config import amass
from mobileposer.utils.model_utils import reduced_pose_to_full
import mobileposer.articulate as art
from mobileposer.models.rnn import RNN
from model.base_model.rnn import RNNWithInit
from utils.data_utils import _foot_min, _get_heights

vi_mask = torch.tensor([1961, 5424, 876, 4362, 411, 3021, 1176, 4662])
ji_mask = torch.tensor([18, 19, 1, 2, 15, 0, 4, 5])

class Poser(L.LightningModule):
    """
    Inputs: N IMUs.
    Outputs: SMPL Pose Parameters (as 6D Rotations).
    """
    def __init__(self, finetune: bool=False, combo_id: str="lw_rp_h", height: bool=False, winit=False,
                 device='cuda'):
        super().__init__()
        
        # constants
        self.C = model_config
        self.finetune = finetune
        self.hypers = finetune_hypers if finetune else train_hypers

        # input dimensions
        imu_set = amass.combos_mine[combo_id]
        imu_num = len(imu_set)
        imu_input_dim = imu_num * 12
        
        if model_config.poser_wh:
            self.input_dim = imu_input_dim + 2
        else:
            self.input_dim = imu_input_dim
            
        # model definitions
        self.bodymodel = art.model.ParametricModel(paths.smpl_file, device=device)
        self.pose = RNNWithInit(n_input=self.input_dim, 
                                n_output=joint_set.n_reduced*6, 
                                n_hidden=512, 
                                n_rnn_layer=2, 
                                dropout=0.4,
                                init_size=24 * 3,
                                bidirectional=False) # pose estimation model
        
        # log input and output dimensions
        if torch.cuda.current_device() == 0:
            print(f"Input dimensions: {self.input_dim}")
            print(f"Output dimensions: {joint_set.n_reduced*6}")
        
        # loss function
        self.loss = nn.MSELoss()
        self.t_weight = 1e-5 
        self.use_pos_loss = True

        # track stats
        self.validation_step_loss = []
        self.training_step_loss = []
        self.save_hyperparameters()

    @classmethod
    def from_pretrained(cls, model_path):
        # init pretrained-model
        model = Poser.load_from_checkpoint(model_path)
        model.hypers = finetune_hypers
        model.finetune = True
        return model

    def _reduced_global_to_full(self, reduced_pose):
        pose = art.math.r6d_to_rotation_matrix(reduced_pose).view(-1, joint_set.n_reduced, 3, 3)
        pose = reduced_pose_to_full(pose.unsqueeze(0)).squeeze(0).view(-1, 24, 3, 3)
        pred_pose = self.global_to_local_pose(pose)
        for ignore in joint_set.ignored: pred_pose[:, ignore] = torch.eye(3, device=self.device)
        pred_pose[:, 0] = pose[:, 0]
        return pred_pose

    def predict_RNN(self, input, init_pose):
        input_lengths = input.shape[0]
        
        init_pose = init_pose.view(1, 24, 3, 3)
        init_joint = self.bodymodel.forward_kinematics(init_pose)[1].view(-1, 72)
        
        input = (input.unsqueeze(0), init_joint)
        
        pred_pose = self.forward(input, [input_lengths])
        
        return pred_pose.squeeze(0)

    def forward(self, batch, input_lengths=None):
        # forward the pose prediction model
        pred_pose, _, _ = self.pose(batch, input_lengths)
        return pred_pose

    def hat(self, v):
        """
        将旋转向量 v (shape: [..., 3]) 转换为对应的 3x3 反对称矩阵.
        """
        zero = torch.zeros_like(v[..., 0])
        M = torch.stack([
            torch.stack([zero, -v[..., 2], v[..., 1]], dim=-1),
            torch.stack([v[..., 2], zero, -v[..., 0]], dim=-1),
            torch.stack([-v[..., 1], v[..., 0], zero], dim=-1)
        ], dim=-2)
        return M

    def rodrigues(self, omega):
        """
        将旋转向量 omega (shape: [..., 3]) 转换为旋转矩阵 (shape: [..., 3, 3])
        采用 Rodrigues 公式.
        """
        theta = torch.norm(omega, dim=-1, keepdim=True)  # shape: [..., 1]
        I = torch.eye(3, device=omega.device).expand(omega.shape[:-1] + (3, 3))
        # 避免除零，利用 theta+1e-8
        u = omega / (theta + 1e-8)
        u_hat = self.hat(u)
        cos_theta = torch.cos(theta).unsqueeze(-1)
        sin_theta = torch.sin(theta).unsqueeze(-1)

        R = I + sin_theta * u_hat + (1 - cos_theta) * (u_hat @ u_hat)
        return R

    def euler2rot(self, euler_angles):
        theta_z = euler_angles[..., 0]
        theta_y = euler_angles[..., 1]
        theta_x = euler_angles[..., 2]

        # 计算对应的余弦和正弦值
        cosz, sinz = torch.cos(theta_z), torch.sin(theta_z)
        cosy, siny = torch.cos(theta_y), torch.sin(theta_y)
        cosx, sinx = torch.cos(theta_x), torch.sin(theta_x)

        # 构造绕 Z 轴旋转矩阵 Rz，shape: [B, S, 3, 3]
        Rz = torch.stack([
            torch.stack([cosz, -sinz, torch.zeros_like(cosz)], dim=-1),
            torch.stack([sinz,  cosz, torch.zeros_like(cosz)], dim=-1),
            torch.stack([torch.zeros_like(cosz), torch.zeros_like(cosz), torch.ones_like(cosz)], dim=-1)
        ], dim=-2)

        # 构造绕 Y 轴旋转矩阵 Ry，shape: [B, S, 3, 3]
        Ry = torch.stack([
            torch.stack([cosy, torch.zeros_like(cosy), siny], dim=-1),
            torch.stack([torch.zeros_like(cosy), torch.ones_like(cosy), torch.zeros_like(cosy)], dim=-1),
            torch.stack([-siny, torch.zeros_like(cosy), cosy], dim=-1)
        ], dim=-2)

        # 构造绕 X 轴旋转矩阵 Rx，shape: [B, S, 3, 3]
        Rx = torch.stack([
            torch.stack([torch.ones_like(cosx), torch.zeros_like(cosx), torch.zeros_like(cosx)], dim=-1),
            torch.stack([torch.zeros_like(cosx), cosx, -sinx], dim=-1),
            torch.stack([torch.zeros_like(cosx), sinx, cosx], dim=-1)
        ], dim=-2)

        # 按ZYX顺序计算最终旋转矩阵： R = Rz @ Ry @ Rx
        R = torch.matmul(torch.matmul(Rz, Ry), Rx)  # 结果 shape: [B, S, 3, 3]
        return R

    def shared_step(self, batch):
        # unpack data
        inputs, outputs = batch
        imu_inputs, input_lengths = inputs
        outputs, output_lengths = outputs

        # target pose
        target_pose = outputs['poses'] # [batch_size, window_length, 144] 
        B, S, _ = target_pose.shape

        # target joints
        joints = outputs['joints']
        target_joints = joints.view(B, S, -1)

        # predict pose
        pose_input = imu_inputs

        pose_input_noisy = pose_input.clone()
        rot = pose_input_noisy[..., 15:24].view(B, S, 1, 3, 3)   # shape: [B, S, 18]
        
        # TODO: add rotation noise
        gt_pose_6d = target_pose.view(-1, 24, 6).clone()
        gt_pose = art.math.r6d_to_rotation_matrix(gt_pose_6d).view(-1, 24, 3, 3)

        gt_pose_local = self.global_to_local_pose(gt_pose)
        gt_pose_local_thigh = gt_pose_local[:, 2]
        gt_pose_local_euler = art.math.rotation_matrix_to_euler_angle(gt_pose_local_thigh).view(-1, 3)

        noise_euler = gt_pose_local_euler.clone()
        # TODO: tune noise level
        noise_euler[:, 0] = - noise_euler[:, 0] * 0.5
        noise_euler[:, 1] = 0.0
        noise_euler[:, 2] = - noise_euler[:, 2] * 0.5
        R_noise = art.math.euler_angle_to_rotation_matrix(noise_euler).view(B, S, 1, 3, 3)
        
        rot_noisy = torch.matmul(rot, R_noise).view(B, S, 9)

        pose_input_noisy[..., 15:24] = rot_noisy      
        
        pose_input = (pose_input_noisy, target_joints[:, 0])
        
        pose_p = self(pose_input, input_lengths)

        # compute pose loss
        pose_t = target_pose.view(B, S, 24, 6)[:, :, joint_set.reduced].view(B, S, -1)
        loss = self.loss(pose_p, pose_t)
        
        if self.C.jerk_loss:
            loss += self.t_weight*self.compute_jerk_loss(pose_p)

        # joint position loss
        if self.use_pos_loss:
            full_pose_p = self._reduced_global_to_full(pose_p)
            joints_p = self.bodymodel.forward_kinematics(pose=full_pose_p.view(-1, 216))[1].view(B, S, -1)
            loss += self.loss(joints_p, target_joints)

        return loss

    def compute_jerk_loss(self, pred_pose):
        jerk = pred_pose[:, 3:, :] - 3*pred_pose[:, 2:-1, :] + 3*pred_pose[:, 1:-2, :] - pred_pose[:, :-3, :]
        l1_norm = torch.norm(jerk, p=1, dim=2)
        return l1_norm.sum(dim=1).mean()

    def compute_temporal_loss(self, pred_pose):
        acc = pred_pose[:, 2:, :] + pred_pose[:, :-2, :] - 2*pred_pose[:, 1:-1, :]
        l1_norm = torch.norm(acc, p=1, dim=2)
        return l1_norm.sum(dim=1).mean() 

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("training_step_loss", loss.item(), batch_size=self.hypers.batch_size)
        self.training_step_loss.append(loss.item())
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("validation_step_loss", loss.item(), batch_size=self.hypers.batch_size)
        self.validation_step_loss.append(loss.item())
        return {"loss": loss}
    
    def predict_step(self, batch, batch_idx):
        inputs, target = batch
        imu_inputs, input_lengths = inputs
        return self(imu_inputs, input_lengths)

    # train epoch start
    def on_fit_start(self):
        self.bodymodel = art.model.ParametricModel(paths.smpl_file, device=self.device)
        self.global_to_local_pose = self.bodymodel.inverse_kinematics_R
    
    def on_train_epoch_end(self):
        self.epoch_end_callback(self.training_step_loss, loop_type="train")
        self.training_step_loss.clear()    # free memory

    def on_validation_epoch_end(self):
        self.epoch_end_callback(self.validation_step_loss, loop_type="val")
        self.validation_step_loss.clear()  # free memory

    def on_test_epoch_end(self, outputs):
        self.epoch_end_callback(outputs, loop_type="test")

    def epoch_end_callback(self, outputs, loop_type):
        # log average loss
        average_loss = torch.mean(torch.Tensor(outputs))
        self.log(f"{loop_type}_loss", average_loss, prog_bar=True, batch_size=self.hypers.batch_size)
        # log learning late
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("learning_rate", lr, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hypers.lr) 
        return optimizer