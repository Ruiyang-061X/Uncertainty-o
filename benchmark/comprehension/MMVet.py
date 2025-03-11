from datasets import load_dataset
from torch.utils.data import Dataset


class MMVet(Dataset):


    def __init__(self):
        super().__init__()
        self.ds = load_dataset("whyu/mm-vet")


    def __len__(self):
        return len(self.ds['test'])


    def __getitem__(self, idx):
        row = self.ds['test'][idx]
        res = {
            'idx': idx,
            'data': {
                'image': row['image'],
                'text': {
                    'question': row['question'],
                    'answer': row['answer'],
                }
            }
        }
        return res


if __name__ == "__main__":
    benchmark = MMVet()
    print(len(benchmark))
    print(benchmark[0])