import os
import json
from torch.utils.data import Dataset


class MSRVTTQA(Dataset):


    def __init__(self):
        super().__init__()
        self.data_root = '/data/lab/yan/huzhang/huzhang1/data/MSRVTTQA/video'
        self.annotation_file_path = '/data/lab/yan/huzhang/huzhang1/data/MSRVTTQA/annotation.json'
        self.annotation_list = json.load(open(self.annotation_file_path))


    def __len__(self):
        return len(self.annotation_list)


    def __getitem__(self, idx):
        res = {
            'idx': idx,
            'data': {
                'video': os.path.join(self.data_root, 'video' + str(self.annotation_list[idx]['video_id']) + '.mp4'),
                'text': {
                    'question': self.annotation_list[idx]['question'] + ' Answer the question using a single word or phrase.',
                    'answer': self.annotation_list[idx]['answer'],
                }
            }
        }
        return res


if __name__ == "__main__":
    benchmark = MSRVTTQA()
    print(len(benchmark))
    print(benchmark[0])