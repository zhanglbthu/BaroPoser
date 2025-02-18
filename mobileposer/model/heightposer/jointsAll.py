import os
import torch
import torch.nn as nn
import lightning as L
from torch.nn import functional as F
import numpy as np

from mobileposer.config import *
import mobileposer.articulate as art
from model.base_model.rnn import RNN, RNNWithInit

class JointsAll(L.LightningModule):
    """
    Inputs: N IMUs.
    Outputs: 24 Joint positions. 
    """
    def __init__(self, finetune: bool=False, combo_id: str="lw_rp_h", height: bool=False, winit: bool=True, device='cuda'):
        super().__init__()
        # constants
        self.C = model_config
        self.finetune = finetune
        self.hypers = finetune_hypers if finetune else train_hypers
        
        # input dimensions
        imu_set = amass.combos_mine[combo_id]
        imu_num = len(imu_set)
        imu_input_dim = len(joint_set.pred) * 3 + imu_num * 12
        self.input_dim = imu_input_dim + 2 if model_config.ja_wheights else imu_input_dim
        
        # output dimensions
        pred_joint = joint_set.full
        self.output_dim = len(pred_joint) * 3

        # model definitions
        self.bodymodel = art.model.ParametricModel(paths.smpl_file, device=device)
        
        self.winit = winit
        self.imu = None

        if self.winit:
            print("Using RNN model winit")
            self.joints = RNN(n_input=self.input_dim, n_output = self.output_dim, n_hidden=512,
                                    n_rnn_layer=2, dropout=0.4, bidirectional=False) # joint estimation model with init
        else:
            print("Using biRNN model")
            self.joints = RNN(self.input_dim, self.output_dim, 256)
        
        # log input and output dimensions
        print(f"Input dimensions: {self.input_dim}")
        print(f"Output dimensions: {self.output_dim}")
        
        # loss function 
        self.loss = nn.MSELoss()
        self.t_weight = 1e-5

        # track stats
        self.validation_step_loss = []
        self.training_step_loss = []
        self.save_hyperparameters()

    @classmethod
    def from_pretrained(cls, model_path):
        # init pretrained-model
        model = JointsAll.load_from_checkpoint(model_path)
        model.hypers = finetune_hypers
        model.finetune = True
        return model

    def forward(self, batch, input_lengths=None):
        # forward joint model
        joints, _, _ = self.joints(batch, input_lengths)

        return joints

    def predict_RNN(self, input):
        input_lengths = input.shape[0]
        
        # init_pose = init_pose.view(1, 24, 3, 3)
        
        # init_joint = self.bodymodel.forward_kinematics(init_pose)[1].view(-1, 72)
        
        # input = (input.unsqueeze(0), init_joint)
        input = input.unsqueeze(0)
        
        pred_joints = self.forward(input, [input_lengths])
        
        return pred_joints.squeeze(0)
        
    def predict_biRNN(self, data):
        imu = data.repeat(self.num_total_frames, 1) if self.imu is None else torch.cat((self.imu[1:], data.view(1, -1))) 
        
        pred_joints = self.forward(imu.unsqueeze(0), [self.num_total_frames])
        
        self.imu = imu.squeeze(0)
        
        return pred_joints.squeeze(0)
        
    def reset(self):
        self.imu = None

    def shared_step(self, batch):
        '''
        inputs:
            0: [batch_size, seq_len, input_dim] 
            1: [batch_size] (input lengths)
        outputs:
            0: 'poses', 'joints', 'trans', 'foot_contacts', 'vels'
            1: lengths of each output 
        '''
        inputs, outputs = batch
        
        imu_inputs, input_lengths = inputs
        
        if not model_config.ja_wheights and model_config.wheights:
            imu_inputs = imu_inputs[:, :, :-2]
        
        outputs, _ = outputs
        
        # target joints
        joints = outputs['joints'] # [batch_size, seq_len, 24, 3]

        target_joints = joints.view(joints.shape[0], joints.shape[1], -1)
        B, S, _ = target_joints.shape
        
        # concatenate input
        pred_joints = joints[:, :, joint_set.pred].view(B, S, -1)
        imu_inputs = torch.cat((pred_joints, imu_inputs), dim=-1)
        
        # predicted joints
        pred_joints = self(imu_inputs, input_lengths)

        # compute loss
        loss = self.loss(pred_joints, target_joints)
        
        if self.C.symmetry_loss:
            loss += self.C.sym_loss_weight*self.compute_symmetry_loss(pred_joints, target_joints)
        
        if self.C.jerk_loss:
            loss += self.C.jerk_loss_weight*self.compute_temporal_loss(pred_joints)
            
        return loss

    def compute_temporal_loss(self, pred_pose):
        acc = pred_pose[:, 2:, :] + pred_pose[:, :-2, :] - 2*pred_pose[:, 1:-1, :]
        l1_norm = torch.norm(acc, p=1, dim=2)
        return l1_norm.sum(dim=1).mean() 

    def compute_symmetry_loss(self, pred_joint, target_joint):
        B, S = pred_joint.shape[:2]
        
        joint_p, joint_t = pred_joint.view(B, S, -1, 3), target_joint.view(B, S, -1, 3)
        joint_rarm_p = joint_p[:, :, joint_set.rarm]        
        joint_larm_t = joint_t[:, :, joint_set.larm]
        
        # extract y values from arm joints: [B, S, joint_num, 3] -> [B, S, joint_num]
        y_rarm_p = joint_rarm_p[:, :, :, 1].view(B, S, -1)
        y_larm_t = joint_larm_t[:, :, :, 1].view(B, S, -1)
        
        sym_loss = torch.abs(y_rarm_p - y_larm_t)
        
        return sym_loss.sum(dim=1).mean()
        
        
    
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

    def on_train_epoch_end(self):
        self.epoch_end_callback(self.training_step_loss, loop_type="train")
        self.training_step_loss.clear()

    def on_validation_epoch_end(self):
        self.epoch_end_callback(self.validation_step_loss, loop_type="val")
        self.validation_step_loss.clear()

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
        return torch.optim.AdamW(self.parameters(), lr=self.hypers.lr)
