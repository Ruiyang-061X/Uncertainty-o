from datasets import load_dataset
from torch.utils.data import Dataset


class LLaVABench(Dataset):


    def __init__(self):
        super().__init__()
        self.ds = load_dataset("lmms-lab/llava-bench-in-the-wild")


    def __len__(self):
        return len(self.ds['train'])


    def __getitem__(self, idx):
        row = self.ds['train'][idx]
        res = {
            'idx': idx,
            'data': {
                'image': row['image'],
                'text': {
                    'question': row['question'],
                    'answer': row['gpt_answer'],
                }
            }
        }
        return res


if __name__ == "__main__":
    benchmark = LLaVABench()
    print(len(benchmark))
    print(benchmark[0])