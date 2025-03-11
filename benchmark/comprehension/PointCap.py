import os
import json
from torch.utils.data import Dataset


class PointCap(Dataset):


    def __init__(self):
        super().__init__()
        self.data_root = '/data/lab/yan/huzhang/huzhang1/data/PointCap/point_cloud'
        self.annotation_file_path = '/data/lab/yan/huzhang/huzhang1/data/PointCap/annotation.json'
        self.annotation_list = json.load(open(self.annotation_file_path))
        self.pc_id_list = list(self.annotation_list.keys())


    def __len__(self):
        return len(self.annotation_list)


    def __getitem__(self, idx):
        res = {
            'idx': idx,
            'data': {
                'point_cloud': os.path.join(self.data_root, f'{self.pc_id_list[idx]}_8192.npy'),
                'text': {
                    'question': 'Provide a one-sentence caption for the provided point cloud.',
                    'answer': self.annotation_list[self.pc_id_list[idx]],
                }
            }
        }
        return res


if __name__ == "__main__":
    benchmark = PointCap()
    print(len(benchmark))
    print(benchmark[0])