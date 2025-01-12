import torch
import os
import numpy as np
from mobileposer.config import paths

def process_dataset():
    pass

if __name__ == '__main__':
    data_dir = paths.processed_datasets / 'eval'
    # 列举data_dir下所有以.pt结尾的文件
    files = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
    for f in files:
        print(f)
        data = torch.load(data_dir / f)
        print(data.keys())