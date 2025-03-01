import torch
from argparse import ArgumentParser
from mobileposer.config import paths, datasets
import mobileposer.articulate as art
from utils.data_utils import _get_heights
from tqdm import tqdm

bodymodel = art.model.ParametricModel(paths.smpl_file)
vi_mask = torch.tensor([1961, 5424, 876, 4362, 411, 3021])

def process_data(data_dir, out_dir):
    # 遍历data_dir下的所有.pt文件
    for data_file in data_dir.iterdir():
        if data_file.suffix != ".pt":
            continue
        
        file_data = torch.load(data_file)
        
        poses, trans = file_data["pose"], file_data["tran"]
        shapes = file_data.get("shape", [None] * len(poses))
        
        # print file_data name and if shape is None
        print("Processing", data_file.name, "shape is None:", all([s is None for s in shapes]))
        
        root_heights, pocket_heights, wrist_heights = [], [], []
        
        for pose, tran, shape in tqdm(zip(poses, trans, shapes)):
            _, _, glb_v = bodymodel.forward_kinematics(pose=pose, tran=tran, shape=shape, calc_mesh=True)
            
            root_h, pocket_h, wrist_h = _get_heights(glb_v, vi_mask)
            root_heights.append(root_h.clone())
            pocket_heights.append(pocket_h.clone())
            wrist_heights.append(wrist_h.clone())
            
        data = {key: value for key, value in file_data.items() if key not in ["ground", "heights"]}
        data["rootH"] = root_heights
        data["pocketH"] = pocket_heights
        data["wristH"] = wrist_heights
        
        torch.save(data, out_dir / data_file.name)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    
    if args.train:
        print("Process training data...")
        data_dir = paths.processed_datasets
        out_dir = paths.out_datasets
    else:
        print("Process testing data...")
        data_dir = paths.processed_datasets / "eval"
        out_dir = paths.out_datasets / "eval"
    
    
    # create out_dir if it doesn't exist
    out_dir.mkdir(exist_ok=True)
    process_data(data_dir, out_dir)
