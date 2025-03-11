import os
import csv
from torch.utils.data import Dataset


class AudioCaps(Dataset):


    def __init__(self):
        super().__init__()
        self.data_root = '/data/lab/yan/huzhang/huzhang1/data/AudioCaps/test/'
        self.annotation_file_path = '/data/lab/yan/huzhang/huzhang1/data/AudioCaps/test.csv'
        self.annotation_list = []
        with open(self.annotation_file_path, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.annotation_list.append({
                    'audiocap_id': row['audiocap_id'],
                    'caption': row['caption'],
                })


    def __len__(self):
        return len(self.annotation_list)
    

    def __getitem__(self, idx):
        res = {
            'idx': idx,
            'data': {
                'audio': os.path.join(self.data_root, f"{self.annotation_list[idx]['audiocap_id']}.wav"),
                'text': {
                    'question': 'Provide a one-sentence caption for the provided audio.',
                    'answer': self.annotation_list[idx]['caption']
                }
            }
        }
        return res


if __name__ == "__main__":
    benchmark = AudioCaps()
    print(len(benchmark))
    print(benchmark[0])