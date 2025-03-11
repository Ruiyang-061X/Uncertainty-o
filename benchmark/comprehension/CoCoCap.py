import os
import json
from torch.utils.data import Dataset


class CoCoCap(Dataset):


    def __init__(self):
        super().__init__()
        self.data_root = '/data/lab/yan/huzhang/huzhang1/data/CoCoCap/image/'
        self.annotation_file_path = '/data/lab/yan/huzhang/huzhang1/data/CoCoCap/annotation.json'
        self.annotation_list = json.load(open(self.annotation_file_path))


    def __len__(self):
        return len(self.annotation_list)


    def __getitem__(self, idx):
        res = {
            'idx': idx,
            'data': {
                'image': os.path.join(self.data_root, self.annotation_list[idx]['image']),
                'text': {
                    'question': 'Provide a one-sentence caption for the provided image.',
                    'answer': self.annotation_list[idx]['caption'][0],
                }
            }
        }
        return res


if __name__ == "__main__":
    benchmark = CoCoCap()
    print(len(benchmark))
    print(benchmark[0])