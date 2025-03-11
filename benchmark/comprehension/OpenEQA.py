import os
import json
from torch.utils.data import Dataset


class OpenEQA(Dataset):


    def __init__(self):
        super().__init__()
        self.data_root = '/data/lab/yan/huzhang/huzhang1/data/OpenEQA/'
        self.annotation_file_path = '/data/lab/yan/huzhang/huzhang1/data/OpenEQA/open-eqa-v0.json'
        self.annotation_list = json.load(open(self.annotation_file_path))


    def __len__(self):
        return len(self.annotation_list)


    def __getitem__(self, idx):
        res = {
            'idx': idx,
            'data': {
                'video': os.path.join(self.data_root, str(self.annotation_list[idx]['episode_history'].split('/')[1]) + '.mp4'),
                'text': {
                    'question': self.annotation_list[idx]['question'] + ' Answer the question using a single word or phrase.',
                    'answer': self.annotation_list[idx]['answer'],
                }
            }
        }
        return res


if __name__ == "__main__":
    benchmark = OpenEQA()
    print(len(benchmark))
    print(benchmark[0])