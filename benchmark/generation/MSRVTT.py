from datasets import load_dataset
from torch.utils.data import Dataset


class MSRVTT(Dataset):


    def __init__(self):
        super().__init__()
        self.ds = load_dataset("AlexZigma/msr-vtt")


    def __len__(self):
        return len(self.ds['val'])


    def __getitem__(self, idx):
        row = self.ds['val'][idx]
        res = {
            'idx': idx,
            'data': {
                'video': '/data/lab/yan/huzhang/huzhang1/data/MSRVTT/' + row['video_id'] + '.mp4',
                'text': {
                    'question': 'Provide a one-sentence caption for the provided video.',
                    'answer': row['caption'],
                }
            }
        }
        return res


if __name__ == "__main__":
    benchmark = MSRVTT()
    print(len(benchmark))
    print(benchmark[0])