from datasets import load_dataset
from torch.utils.data import Dataset
import os
import soundfile as sf


class ClothoAQA(Dataset):


    def __init__(self):
        super().__init__()
        self.ds = load_dataset("TwinkStart/ClothoAQA")


    def __len__(self):
        return len(self.ds['test'])


    def __getitem__(self, idx):
        row = self.ds['test'][idx]
        output_path = os.path.join('/data/lab/yan/huzhang/huzhang1/data/ClothoAQA', row["file_name"])
        sf.write(output_path, row['audio']["array"], row['audio']["sampling_rate"])
        res = {
            'idx': idx,
            'data': {
                'audio': output_path,
                'text': {
                    'question': row['QuestionText'],
                    'answer': row['answer'],
                }
            }
        }
        return res


if __name__ == "__main__":
    benchmark = ClothoAQA()
    print(len(benchmark))
    print(benchmark[0])