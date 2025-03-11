import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset


label_name_map = {
    0: 'airplane',
    1: 'bathtub',
    2: 'bed',
    3: 'bench',
    4: 'bookshelf',
    5: 'bottle',
    6: 'bowl',
    7: 'car',
    8: 'chair',
    9: 'cone',
    10: 'cup',
    11: 'curtain',
    12: 'desk',
    13: 'door',
    14: 'dresser',
    15: 'flower_pot',
    16: 'glass_box',
    17: 'guitar',
    18: 'keyboard',
    19: 'lamp',
    20: 'laptop',
    21: 'mantel',
    22: 'monitor',
    23: 'night_stand',
    24: 'person',
    25: 'piano',
    26: 'plant',
    27: 'radio',
    28: 'range_hood',
    29: 'sink',
    30: 'sofa',
    31: 'stairs',
    32: 'stool',
    33: 'table',
    34: 'tent',
    35: 'toilet',
    36: 'tv_stand',
    37: 'vase',
    38: 'wardrobe',
    39: 'xbox'
}


class ModelNet(Dataset):


    def __init__(self):
        super().__init__()
        self.ds = load_dataset("jxie/modelnet40")


    def __len__(self):
        return len(self.ds['test'])


    def __getitem__(self, idx):
        row = self.ds['test'][idx]
        inputs = np.array(row['inputs'])
        black_color = np.zeros_like(inputs)
        expanded_inputs = np.hstack((inputs, black_color))
        res = {
            'idx': idx,
            'data': {
                'point_cloud': expanded_inputs,
                'text': {
                    'question': 'What is it? Use one word to describe.',
                    'answer': f'{label_name_map[row["label"]]}.',
                }
            }
        }
        return res


if __name__ == "__main__":
    benchmark = ModelNet()
    print(len(benchmark))
    print(benchmark[0])
    np.savetxt('/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/benchmark/ModelNet/0.xyz', benchmark[0]['data']['point_cloud'], fmt='%.6f', delimiter=' ')