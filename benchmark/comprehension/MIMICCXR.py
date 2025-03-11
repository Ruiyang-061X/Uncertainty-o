from datasets import load_dataset
from torch.utils.data import Dataset


class MIMICCXR(Dataset):


    def __init__(self):
        super().__init__()
        self.ds = load_dataset("itsanmolgupta/mimic-cxr-dataset-10k")


    def __len__(self):
        return len(self.ds['train'])


    def __getitem__(self, idx):
        row = self.ds['train'][idx]
        res = {
            'idx': idx,
            'data': {
                'image': row['image'],
                'text': {
                    'question': 'Provide one-sentence impression for the chest X-ray.',
                    'answer': row['impression'],
                }
            }
        }
        return res


if __name__ == "__main__":
    benchmark = MIMICCXR()
    print(len(benchmark))
    print(benchmark[0])