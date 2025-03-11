import os
import json
from torch.utils.data import Dataset


class MSVDQA(Dataset):


    def __init__(self):
        super().__init__()
        self.data_root = '/data/lab/yan/huzhang/huzhang1/data/MSVDQA/video'
        self.annotation_file_path = '/data/lab/yan/huzhang/huzhang1/data/MSVDQA/annotation.json'
        self.mapping = '/data/lab/yan/huzhang/huzhang1/data/MSVDQA/youtube_mapping.txt'
        self.annotation_list = json.load(open(self.annotation_file_path))
        self.id_to_name_map = {i.strip().split(' ')[1] : i.strip().split(' ')[0] for i in open(self.mapping).readlines()}


    def __len__(self):
        return len(self.annotation_list)


    def __getitem__(self, idx):
        res = {
            'idx': idx,
            'data': {
                'video': os.path.join(self.data_root, self.id_to_name_map['vid' + str(self.annotation_list[idx]['video_id'])] + '.avi'),
                'text': {
                    'question': self.annotation_list[idx]['question'] + ' Answer the question using a single word or phrase.',
                    'answer': self.annotation_list[idx]['answer'],
                }
            }
        }
        return res


if __name__ == "__main__":
    benchmark = MSVDQA()
    print(len(benchmark))
    print(benchmark[0])