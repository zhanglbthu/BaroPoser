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
        imu_input_dim = 22
        
        self.input_dim = imu_input_dim
            
        # model definitions
        self.bodymodel = art.model.ParametricModel(paths.smpl_file, device=device)
        self.pose = RNNWithInit(n_input=self.input_dim, 
                                n_output=joint_set.n_reduced*6, 
                                n_hidden=512, 
                                n_rnn_layer=2, 
                                dropout=0.4,
                                init_size=joint_set.n_pose_init * 3,
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
        input = self.input_normalize(input, angular_vel=True)
        
        init_pose = init_pose.view(1, 24, 3, 3)
        glb_rot, init_joint = self.bodymodel.forward_kinematics(init_pose)
        init_joint = init_joint.view(-1, 72)
        root_rot = glb_rot[:, 2]    
        init_joint = self.joint_normalize(init_joint, root_rot).view(1, -1)
        
        input = (input.unsqueeze(0), init_joint)
        
        pred_pose = self.forward(input, [input_lengths])
        
        return pred_pose.squeeze(0)

    def forward(self, batch, input_lengths=None):
        # forward the pose prediction model
        pred_pose, _, _ = self.pose(batch, input_lengths)
        return pred_pose

    def input_normalize(self, pose_input, angular_vel=False):
        pose_input = pose_input.view(-1, 28)
        glb_acc = pose_input[:, :6].view(-1, 2, 3)
        glb_rot = pose_input[:, 6:24].view(-1, 2, 3, 3)
        root_angular_vel = pose_input[:, 24:27].view(-1, 3)
        rel_height = pose_input[:, 27:28].view(-1, 1)
        
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
        
        # root rotation
        root_rot = imu_inputs[:, :, 15:24].view(-1, 3, 3)

        # target pose
        target_pose = outputs['poses'] # [batch_size, window_length, 144] 
        B, S, _ = target_pose.shape
        target_pose = self.target_pose_normalize(target_pose).view(B, S, -1)

        # target joints
        joints = outputs['joints']
        joints = self.joint_normalize(joints, root_rot)
        target_joints = joints.view(B, S, -1)

        # predict pose
        pose_input = imu_inputs  
        # normalize input
        pose_input = self.input_normalize(pose_input, angular_vel=True).view(B, S, -1)
        
        pose_input = (pose_input, target_joints[:, 0])
        
        pose_p = self(pose_input, input_lengths)

        # compute pose loss
        pose_t = target_pose.view(B, S, 24, 6)[:, :, joint_set.reduced].view(B, S, -1)
        loss = self.loss(pose_p, pose_t)

        # # joint position loss
        # if self.use_pos_loss:
        #     full_pose_p = self._reduced_global_to_full(pose_p)
        #     joints_p = self.bodymodel.forward_kinematics(pose=full_pose_p.view(-1, 216))[1].view(B, S, -1)
        #     loss += self.loss(joints_p, target_joints)

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