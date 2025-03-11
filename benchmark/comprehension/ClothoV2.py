import json
import os
from torch.utils.data import Dataset


class ClothoV2(Dataset):


    def __init__(self):
        super().__init__()
        self.data_root = '/data/lab/yan/huzhang/huzhang1/data/ClothoV2/audio'
        self.annotation_file_path = '/data/lab/yan/huzhang/huzhang1/data/ClothoV2/annotation.json'
        self.annotation_list = json.load(open(self.annotation_file_path))
        self.file_path_list = [os.path.join(self.data_root, i['file_name']) for i in self.annotation_list['images']]
        self.caption_list = []
        m = {}
        for i in self.annotation_list['annotations']:
            if i['image_id'] not in m:
                m[i['image_id']] = i['caption']
        for i in range(len(m)):
            self.caption_list.append(m[i])


    def __len__(self):
        return len(self.file_path_list)


    def __getitem__(self, idx):
        res = {
            'idx': idx,
            'data': {
                'audio': self.file_path_list[idx],
                'text': {
                    'question': 'Provide a one-sentence caption for the provided audio.',
                    'answer': self.caption_list[idx],
                }
            }
        }
        return res


if __name__ == "__main__":
    benchmark = ClothoV2()
    print(len(benchmark))
    print(benchmark[0])