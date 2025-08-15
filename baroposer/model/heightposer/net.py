import os
import numpy as np
import torch
torch.set_printoptions(sci_mode=False)
import torch.nn as nn
from torch.nn import functional as F
import lightning as L
from torch.optim.lr_scheduler import StepLR 
from tqdm import tqdm
import time

from mobileposer.config import *
from mobileposer.utils.model_utils import reduced_pose_to_full
from mobileposer.helpers import *
import mobileposer.articulate as art

from model.heightposer.poser import Poser
from model.heightposer.velocity import Velocity

class HeightPoserNet(L.LightningModule):
    """
    Inputs: N IMUs.
    Outputs: SMPL Pose Parameters (as 6D Rotations) and Translation. 
    """

    def __init__(self, 
                 poser: Poser=None,
                 velocity: Velocity=None,
                 combo_id: str="lw_rp_h"):
        super().__init__()

        # constants
        self.C = model_config
        self.hypers = train_hypers 

        # body model
        self.bodymodel = art.model.ParametricModel(paths.smpl_file, device=self.C.device)
        self.global_to_local_pose = self.bodymodel.inverse_kinematics_R

        # model definitions
        self.pose = poser if poser else Poser(combo_id=combo_id)                   # pose estimation model
        self.velocity = velocity if velocity else Velocity(combo_id=combo_id)       # velocity estimation model

        # base joints
        self.j, _ = self.bodymodel.get_zero_pose_joint_and_vertex() # [24, 3]
        self.feet_pos = self.j[10:12].clone() # [2, 3]
        self.floor_y = self.j[10:12, 1].min().item() # [1]

        # variables
        self.last_root_pos = torch.zeros(3).to(self.C.device)
        self.last_joints = torch.zeros(24, 3).to(self.C.device)
        self.current_root_y = 0
        self.imu = None
        self.rnn_state = None

        if getenv("PHYSICS"):
            from dynamics import PhysicsOptimizer
            self.dynamics_optimizer = PhysicsOptimizer(debug=False)
            self.dynamics_optimizer.reset_states()

        # track stats
        self.validation_step_loss = []
        self.training_step_loss = []
        self.save_hyperparameters(ignore=['poser', 'joints', 'foot_contact', 'velocity', 'joint_all'])

    @classmethod
    def from_pretrained(cls, model_path):
        # init pretrained-model
        model = HeightPoserNet.load_from_checkpoint(model_path)
        model.hypers = finetune_hypers
        model.finetune = True
        return model

    def reset(self):
        self.rnn_state = None
        self.imu = None
        self.current_root_y = 0
        self.last_root_pos = torch.zeros(3).to(self.C.device)
    
    def _reduced_global_to_full(self, reduced_pose):
        pose = art.math.r6d_to_rotation_matrix(reduced_pose).view(-1, 16, 3, 3)
        pose = reduced_pose_to_full(pose.unsqueeze(0)).squeeze(0).view(-1, 24, 3, 3)
        pred_pose = self.global_to_local_pose(pose)
        pred_pose[:, joint_set.leaf_joint] = torch.eye(3, device=self.C.device)
        pred_pose[:, 0] = pose[:, 0]
        return pred_pose

    def _reduced_glb_6d_to_full_local_mat(self, root_rotation, glb_reduced_pose):
        glb_reduced_pose = art.math.r6d_to_rotation_matrix(glb_reduced_pose).view(-1, joint_set.n_reduced_local, 3, 3)
        glb_reduced_pose = root_rotation.unsqueeze(1).matmul(glb_reduced_pose)
        
        global_full_pose = torch.eye(3, device=glb_reduced_pose.device).repeat(glb_reduced_pose.shape[0], 24, 1, 1)
        global_full_pose[:, joint_set.reduced_local] = glb_reduced_pose
        global_full_pose[:, 2] = root_rotation
        pose = self.global_to_local_pose(global_full_pose).view(-1, 24, 3, 3)
        
        pose[:, joint_set.leaf_joint] = torch.eye(3, device=pose.device)
        
        return pose

    def input_process(self, inputs):
        # process input
        input_dim = inputs.shape[-1]
        inputs = inputs.view(-1, input_dim)
        
        imu_inputs = inputs[:, :12 * 2]
        h_inputs = inputs[:, -1:].view(-1, 1)
        
        inputs = torch.cat([imu_inputs, h_inputs], dim=-1)
        
        return inputs

    def predict_full(self, input, init_pose, vel_input=None):
        N, _ = input.shape
        
        pose_input = input
        
        pred_pose = self.pose.predict_RNN(pose_input, init_pose)
        if self.C.global_coord:
            pred_pose = self._reduced_global_to_full(pred_pose).view(-1, 24, 3, 3)
        else:
            root_rotation = input[:, 15:24].view(-1, 3, 3)
            pred_pose = self._reduced_glb_6d_to_full_local_mat(root_rotation=root_rotation, glb_reduced_pose=pred_pose)
        
        # predict velocity
        # input = torch.cat((input[:, :24], input[:, -1:], vel_input), dim=1)

        pred_vel = self.velocity.predict_RNN(input)
        
        pred_vel = pred_vel / (datasets.fps/amass.vel_scale)
        
        translation = self.velocity_to_root_position(pred_vel)

        return pred_pose, translation
    
    def predict(self, input, init_pose, poser_only=False):
        pred_pose = self.pose.predict_RNN(input, init_pose)
            
        if self.C.local_coord:
            root_rotation = input[:, 15:24].view(-1, 3, 3)
            pred_pose = self._reduced_glb_6d_to_full_local_mat(root_rotation=root_rotation, glb_reduced_pose=pred_pose)
        else:
            pred_pose = self._reduced_global_to_full(pred_pose)
            
        if poser_only:    
            return pred_pose
        
        # predict velocity
        pred_vel = self.velocity.predict_RNN(input)
        pred_vel = pred_vel / (datasets.fps/amass.vel_scale)
        
        translation = self.velocity_to_root_position(pred_vel)
        return pred_pose, translation
    
    @staticmethod
    def velocity_to_root_position(velocity):
        r"""
        Change velocity to root position. (not optimized)

        :param velocity: Velocity tensor in shape [num_frame, 3].
        :return: Translation tensor in shape [num_frame, 3] for root positions.
        """
        return torch.stack([velocity[:i+1].sum(dim=0) for i in range(velocity.shape[0])])