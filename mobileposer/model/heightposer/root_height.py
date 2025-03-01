import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import lightning as L
import numpy as np

from mobileposer.config import *
import mobileposer.articulate as art
from mobileposer.models.rnn import RNN


class RootHeight(L.LightningModule):
    """
    Inputs: N IMUs.
    Outputs: Root Height.
    """
    def __init__(self, finetune: bool=False, combo_id = 'lw_rp_h', device='cuda'):
        super().__init__()
        
        # constants
        self.C = model_config
        self.hypers = train_hypers

        # input dimensions
        imu_set = amass.combos_mine[combo_id]
        imu_num = len(imu_set)
        imu_input_dim = imu_num * 12
        self.input_dim = imu_input_dim + 2 if self.C.rooth_wh else imu_input_dim 

        # model definitions
        self.root_height = RNN(self.input_dim, 2, 64, bidirectional=False)  # foot-ground probability model

        # log input and output dimensions
        print(f"Input dimensions: {self.input_dim}")
        print(f"Output dimensions: 1")
        
        # loss function
        self.loss = nn.MSELoss()

        # track stats
        self.validation_step_loss = []
        self.training_step_loss = []
        self.save_hyperparameters()

    def forward(self, batch, input_lengths=None):
        # forward foot contact model
        root_height, _, _ = self.root_height(batch, input_lengths)
        return root_height
    
    def shared_step(self, batch):
        # unpack data
        inputs, outputs = batch
        imu_inputs, input_lengths = inputs
        outputs, _ = outputs

        pass

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

    def on_fit_start(self):
        self.bodymodel = art.model.ParametricModel(paths.smpl_file, device=self.device)
    
    def on_train_epoch_end(self):
        self.epoch_end_callback(self.training_step_loss, loop_type="train")
        self.training_step_loss.clear()    # free memory

    def on_validation_epoch_end(self):
        self.epoch_end_callback(self.validation_step_loss, loop_type="val")
        self.validation_step_loss.clear()  # free memory

    def on_test_epoch_end(self, outputs):
        self.epoch_end_callback(outputs, loop_type="test")

    def epoch_end_callback(self, outputs, loop_type):
        average_loss = torch.mean(torch.Tensor(outputs))
        self.log(f"{loop_type}_loss", average_loss, prog_bar=True, batch_size=self.hypers.batch_size)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hypers.lr) 