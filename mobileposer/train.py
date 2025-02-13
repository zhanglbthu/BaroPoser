import os
import math
import numpy as np
import torch
torch.set_printoptions(sci_mode=False)
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import lightning as L
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch import seed_everything
from argparse import ArgumentParser
from pathlib import Path
from typing import List
from tqdm import tqdm 
import wandb
import sys
from mobileposer.config import *

from mobileposer.constants import IMUPOSER, MOBILEPOSER, HEIGHTPOSER
from mobileposer.data import PoseDataModule
from mobileposer.utils.file_utils import (
    get_datestring, 
)
from mobileposer.config import paths, train_hypers, finetune_hypers, model_config

# set precision for Tensor cores
torch.set_float32_matmul_precision('medium')

class TrainingManager:
    """Manage training of MobilePoser modules."""
    def __init__(self, finetune: str=None, fast_dev_run: bool=False):
        self.finetune = finetune
        self.fast_dev_run = fast_dev_run
        self.hypers = finetune_hypers if finetune else train_hypers

    def _setup_tensorboard_logger(self, save_path: Path):
        tensorboard_logger = TensorBoardLogger(
            save_path, 
            name=get_datestring()
        )
        return tensorboard_logger
    
    def _setup_callbacks(self, save_path):
        checkpoint_callback = ModelCheckpoint(
                monitor="validation_step_loss",
                save_top_k=3,
                mode="min",
                verbose=False,
                dirpath=save_path, 
                save_weights_only=True,
                filename="{epoch}-{validation_step_loss:.4f}"
                )
        return checkpoint_callback

    def _setup_trainer(self, module_path: Path):
        print("Module Path: ", module_path.name, module_path)
        logger = self._setup_tensorboard_logger(module_path)
        checkpoint_callback = self._setup_callbacks(module_path)
        print(self.hypers.device)
        trainer = L.Trainer(
                fast_dev_run=self.fast_dev_run,
                min_epochs=self.hypers.num_epochs,
                max_epochs=self.hypers.num_epochs,
                devices=self.hypers.device, 
                accelerator=self.hypers.accelerator,
                logger=logger,
                callbacks=[checkpoint_callback],
                deterministic=True,
                )
        return trainer

    def train_module(self, model: L.LightningModule, module_name: str, checkpoint_path: Path, combo_id: str=None):
        # set the appropriate hyperparameters
        model.hypers = self.hypers 

        # create directory for module
        module_path = checkpoint_path / combo_id / module_name
        os.makedirs(module_path, exist_ok=True)
        
        # save config.py to model_path
        os.system(f"cp config.py {module_path}")
            
        datamodule = PoseDataModule(finetune=self.finetune, combo_id=combo_id, 
                                    wheights=model_config.wheights,
                                    winit=False)
        
        trainer = self._setup_trainer(module_path)

        print()
        print("-" * 50)
        print(f"Training Module: {module_name}")
        print("-" * 50)
        print()

        try:
            trainer.fit(model, datamodule=datamodule)
        finally:
            wandb.finish()
            del model
            torch.cuda.empty_cache()

def get_checkpoint_path(finetune: str, init_from: str, name: str=None):
    if finetune:
        # finetune from a checkpoint
        parts = init_from.split(os.path.sep)
        checkpoint_path = Path(os.path.join(parts[0], parts[1], parts[2]))
        finetune_dir = f"finetuned_{finetune}"
        checkpoint_path = checkpoint_path / finetune_dir

    else:
        # make directory for trained models
        if init_from is not None:
            checkpoint_path = Path(init_from)
        else:
            checkpoint_path = paths.checkpoint / name

    os.makedirs(checkpoint_path, exist_ok=True)
    return Path(checkpoint_path)

def get_model(model_name: str, module_name: str):
    # 取model_name以_分割，取第一个元素
    model = model_name.split("_")[0]
    if model == "imuposer":
        module = IMUPOSER[module_name]
        model = module(combo_id=model_config.combo_id)
    
    elif model == "mobileposer":
        module = MOBILEPOSER[module_name]
        model = module(combo_id=model_config.combo_id)
        
    elif model == "heightposer":
        module = HEIGHTPOSER[module_name]
        model = module(combo_id=model_config.combo_id)    
    
    return model

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--module", default=None)
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--finetune", type=str, default=None)
    parser.add_argument("--init-from", nargs="?", default=None, type=str)
    args = parser.parse_args()

    # set seed for reproducible results
    seed_everything(42, workers=True)

    # create checkpoint directory, if missing
    paths.checkpoint.mkdir(exist_ok=True)

    # initialize training manager
    checkpoint_path = get_checkpoint_path(args.finetune, args.init_from, model_config.name)
    
    training_manager = TrainingManager(
        finetune=args.finetune,
        fast_dev_run=args.fast_dev_run
    )

    # set imu set combo
    imu_set = amass.combos_mine[model_config.combo_id]
    imu_num = len(imu_set)
    
    model = get_model(model_config.name, args.module)
    training_manager.train_module(model, args.module, checkpoint_path, combo_id=model_config.combo_id)