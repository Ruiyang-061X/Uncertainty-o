import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1')
import os
import numpy as np
import json
from torch.utils.data import Dataset
from util.misc import *


class Pix3D(Dataset):


    def __init__(self):
        super().__init__()
        self.data_root = '/data/lab/yan/huzhang/huzhang1/data/Pix3D/'
        self.json_path = '/data/lab/yan/huzhang/huzhang1/data/Pix3D/pix3d.json'
        with open(self.json_path, 'r') as f:
            self.data = json.load(f)


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        info = self.data[idx]
        img_path = os.path.join(self.data_root, info['img'])
        model_path = os.path.join(self.data_root, info['model'])
        point_cloud = read_obj_file(model_path)
        point_cloud_path = f'/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/benchmark/Pix3D/{idx}.npy'
        np.save(point_cloud_path, point_cloud)
        res = {
            'idx': idx,
            'data': {
                'image': img_path,
                'point_cloud': point_cloud_path,
            }
        }
        return res


if __name__ == "__main__":
    benchmark = Pix3D()
    print(len(benchmark))
    print(benchmark[0])