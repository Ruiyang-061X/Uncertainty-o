import numpy as np
from torch.utils.data import Dataset
import os
import json


label_dict = {
    '02691156': 'Airplane',
    '02773838': 'Bag',
    '02954340': 'Cap',
    '02958343': 'Car',
    '03001627': 'Chair',
    '03261776': 'Earphone',
    '03467517': 'Guitar',
    '03624134': 'Knife',
    '03636649': 'Lamp',
    '03642806': 'Laptop',
    '03790512': 'Motorbike',
    '03797390': 'Mug',
    '03948459': 'Pistol',
    '04099429': 'Rocket',
    '04225987': 'Skateboard',
    '04379243': 'Table'
}


class ShapeNet(Dataset):


    def __init__(self):
        super().__init__()
        self.data_dir = '/data/lab/yan/huzhang/huzhang1/data/ShapeNet'
        self.train_file_list_path = '/data/lab/yan/huzhang/huzhang1/data/ShapeNet/train_test_split/shuffled_train_file_list.json'
        with open(self.train_file_list_path, 'r') as f:
            self.file_list = json.load(f)


    def __len__(self):
        return len(self.file_list)


    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        category_id = file_path.split('/')[1]
        file_path = file_path.replace('shape_data/', '')
        point_cloud_file = os.path.join(self.data_dir, file_path + '.txt')
        points = np.loadtxt(point_cloud_file)[:, :3]
        rgb = np.zeros((points.shape[0], 3))
        point_cloud = np.hstack((points, rgb))
        category_name = label_dict[category_id]
        res = {
            'idx': idx,
            'data': {
                'point_cloud': point_cloud,
                'text': {
                    'question': 'What is it? Use one word to describe.',
                    'answer': category_name,
                }
            }
        }
        return res


if __name__ == "__main__":
    benchmark = ShapeNet()
    print(len(benchmark))
    print(benchmark[0])
    np.savetxt('/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/benchmark/ShapeNet/0.xyz', benchmark[0]['data']['point_cloud'], fmt='%.6f', delimiter=' ')