import os
import csv
from torch.utils.data import Dataset


class NextQA(Dataset):


    def __init__(self):
        super().__init__()
        self.data_root = '/data/lab/yan/huzhang/huzhang1/data/NextQA/video/'
        self.annotation_file_path = '/data/lab/yan/huzhang/huzhang1/data/NextQA/annotation/val.csv'
        self.annotation_list = []
        with open(self.annotation_file_path, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.annotation_list.append({
                    'video': row['video'],
                    'question': row['question'],
                    'answer': row['answer'],
                    'choices': [row['a0'], row['a1'], row['a2'], row['a3'], row['a4']],
                })


    def __len__(self):
        return len(self.annotation_list)
    

    def __getitem__(self, idx):
        question = self.annotation_list[idx]['question']
        choices = self.annotation_list[idx]['choices']
        formatted_choices = '\n'.join([f"({i}): {choices[i]}" for i in range(len(choices))])
        final_question = f'{question}\n{formatted_choices}\nThis is single-choice question, answer with one choice number in 0, 1, 2, 3, 4.'
        res = {
            'idx': idx,
            'data': {
                'video': os.path.join(self.data_root, f"{self.annotation_list[idx]['video']}.mp4"),
                'text': {
                    'question': final_question,
                    'answer': self.annotation_list[idx]['answer']
                }
            }
        }
        return res


if __name__ == "__main__":
    benchmark = NextQA()
    print(len(benchmark))
    print(benchmark[0])