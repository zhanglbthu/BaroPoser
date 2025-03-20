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
        imu_input_dim = imu_num * 12 + 1 if model_config.global_coord else 22
        
        self.input_dim = imu_input_dim
        self.output_dim = (joint_set.n_reduced + 1) * 6 if model_config.global_coord else joint_set.n_reduced * 6
        self.init_size = self.output_dim
            
        # model definitions
        self.bodymodel = art.model.ParametricModel(paths.smpl_file, device=device)
        self.pose = RNNWithInit(n_input=self.input_dim, 
                                n_output=self.output_dim, 
                                n_hidden=512, 
                                n_rnn_layer=2, 
                                dropout=0.4,
                                init_size=self.init_size,
                                bidirectional=False) # pose estimation model
        
        # log input and output dimensions
        if torch.cuda.current_device() == 0:
            print(f"Input dimensions: {self.input_dim}")
            print(f"Output dimensions: {self.output_dim}")
        
        # loss function
        self.loss = nn.MSELoss()

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
        
        glb_rot, _ = self.bodymodel.forward_kinematics(init_pose.view(-1, 24, 3, 3))
        glb_rot_6d = art.math.rotation_matrix_to_r6d(glb_rot).view(-1, 24, 6)
        
        if self.C.global_coord:
            input = self.input_normalize(input, angular_vel=True, global_coord=True)
            init_p = glb_rot_6d.view(-1, 24, 6)[:, joint_set.reduced_old].view(1, -1)
            
        else:
            input = self.input_normalize(input, angular_vel=True)
            init_p = self.target_pose_normalize(glb_rot_6d).view(-1, 24, 6)[:, joint_set.reduced].view(1, -1)
        
        input = (input.unsqueeze(0), init_p)
        
        pred_pose = self.forward(input, [input_lengths])
        
        return pred_pose.squeeze(0)

    def forward(self, batch, input_lengths=None):
        # forward the pose prediction model
        pred_pose, _, _ = self.pose(batch, input_lengths)
        return pred_pose

    def input_normalize(self, pose_input, angular_vel=False, add_noise=False, global_coord=False):
        if len(pose_input.shape) == 3:
            B, S, _ = pose_input.shape
        pose_input = pose_input.view(-1, 28)
        glb_acc = pose_input[:, :6].view(-1, 2, 3)
        glb_rot = pose_input[:, 6:24].view(-1, 2, 3, 3)
        root_angular_vel = pose_input[:, 24:27].view(-1, 3)
        rel_height = pose_input[:, 27:28].view(-1, 1)
        
        if add_noise:
            # add noise to glb_rot
            glb_rot = glb_rot.view(B, S, 2, 3, 3)
            axis_angle_thigh = torch.randn(B, 1, 3).to(self.device) * self.C.noise_std
            axis_angle_wrist = torch.randn(B, 1, 3).to(self.device) * self.C.noise_std
            
            wrist_rot = glb_rot[:, :, 0].view(B, S, 3, 3)
            thigh_rot = glb_rot[:, :, 1].view(B, S, 3, 3)
            
            wrist_rot_noisy = torch.matmul(wrist_rot, art.math.axis_angle_to_rotation_matrix(axis_angle_wrist).view(B, 1, 3, 3)).view(B, S, 9)
            thigh_rot_noisy = torch.matmul(thigh_rot, art.math.axis_angle_to_rotation_matrix(axis_angle_thigh).view(B, 1, 3, 3)).view(B, S, 9)
            
            glb_rot = torch.cat([wrist_rot_noisy, thigh_rot_noisy], dim=-1).view(-1, 2, 3, 3)
            
            # add noise to angular velocity
            root_angular_vel_mat = art.math.axis_angle_to_rotation_matrix(root_angular_vel).view(B, S, 3, 3)
            # angular_vel_noise = R_noise ^ T * angular_vel_mat * R_noise
            root_angular_vel_noisy = torch.matmul(torch.matmul(art.math.axis_angle_to_rotation_matrix(axis_angle_thigh).view(B, 1, 3, 3).transpose(2, 3), 
                                                               root_angular_vel_mat), 
                                                  art.math.axis_angle_to_rotation_matrix(axis_angle_thigh).view(B, 1, 3, 3)).view(B, S, 9)
            root_angular_vel = art.math.rotation_matrix_to_axis_angle(root_angular_vel_noisy).view(-1, 3)
            
            # add noise to relative height
            rel_height = rel_height + torch.randn(B, S, 1).view(-1, 1).to(self.device) * self.C.h_noise
        
        if global_coord:
            input = torch.cat((glb_acc.flatten(1), glb_rot.flatten(1), rel_height), dim=1)
        else:
            g = torch.tensor([0, -1, 0], dtype=torch.float32, device=pose_input.device)
            root_rotation = glb_rot[:, 1] # [N, 3, 3]
            g_root = root_rotation.transpose(1, 2).matmul(g) # [N, 3]
            
            acc = torch.cat((glb_acc[:, :1] - glb_acc[:, 1:], glb_acc[:, 1:]), dim=1).bmm(glb_rot[:, 1])
            ori = torch.cat((glb_rot[:, 1:].transpose(2, 3).matmul(glb_rot[:, :1]), glb_rot[:, 1:]), dim=1)
            
            if angular_vel:
                input = torch.cat((acc.flatten(1), ori[:, :1].flatten(1), root_angular_vel, rel_height, g_root), dim=1)
            else:
                input = torch.cat((acc.flatten(1), ori.flatten(1)), dim=1)
        return input
    
    def target_pose_normalize(self, target_pose):
        '''
        convert target pose [N, 24, 6] to relative pose
        '''
        
        target_pose = target_pose.view(-1, 24, 6)
        target_pose = art.math.r6d_to_rotation_matrix(target_pose).view(-1, 24, 3, 3)
        
        target_pose_rel = target_pose[:, 2:3].transpose(2, 3).matmul(target_pose[:, :24])
        
        target_pose_rel_r6d = art.math.rotation_matrix_to_r6d(target_pose_rel)
        return target_pose_rel_r6d 

    def joint_normalize(self, joints, root_rot):
        
        joints = joints.view(-1, 24, 3)
        j = (joints[:, :] - joints[:, 2:3]).bmm(root_rot)
        j_init = j[:, joint_set.joint_init]
        
        return j_init

    def shared_step(self, batch, add_noise=False):
        # unpack data
        inputs, outputs = batch
        imu_inputs, input_lengths = inputs
        outputs, output_lengths = outputs

        # target pose
        target_pose = outputs['poses'] # [batch_size, window_length, 144] 
        B, S, _ = target_pose.shape

        # predict pose
        pose_input = imu_inputs  
        # normalize input
        if model_config.global_coord:
            pose_input = self.input_normalize(pose_input, angular_vel=True, add_noise=add_noise, global_coord=True).view(B, S, -1)
            init_pose = target_pose[:, 0].view(-1, 24, 6)[:, joint_set.reduced_old].view(B, -1)
            pose_input = (pose_input, init_pose)
            pose_t = target_pose.view(B, S, 24, 6)[:, :, joint_set.reduced_old].view(B, S, -1)
        else:
            target_pose = self.target_pose_normalize(target_pose).view(B, S, -1)
            pose_input = self.input_normalize(pose_input, angular_vel=True, add_noise=add_noise).view(B, S, -1)
            init_pose = target_pose[:, 0].view(-1, 24, 6)[:, joint_set.reduced].view(B, -1)
            pose_input = (pose_input, init_pose)
            pose_t = target_pose.view(B, S, 24, 6)[:, :, joint_set.reduced].view(B, S, -1)
        
        pose_p = self(pose_input, input_lengths)

        # compute pose loss
        loss = self.loss(pose_p, pose_t)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, add_noise=True)
        self.log("training_step_loss", loss.item(), batch_size=self.hypers.batch_size)
        self.training_step_loss.append(loss.item())
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, add_noise=False)
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