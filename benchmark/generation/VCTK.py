import os
from torch.utils.data import Dataset


class VCTK(Dataset):


    def __init__(self):
        super().__init__()
        self.data_root = '/data/lab/yan/huzhang/huzhang1/data/VCTK/'
        self.text_root = os.path.join(self.data_root, 'VCTK-Corpus/txt')
        self.wav_root = os.path.join(self.data_root, 'VCTK-Corpus/wav48')
        self.text_files = []
        self.wav_files = []
        for speaker in os.listdir(self.text_root):
            speaker_text_dir = os.path.join(self.text_root, speaker)
            speaker_wav_dir = os.path.join(self.wav_root, speaker)
            for text_file in os.listdir(speaker_text_dir):
                if text_file.endswith('.txt'):
                    wav_file = text_file.replace('.txt', '.wav')
                    self.text_files.append(os.path.join(speaker_text_dir, text_file))
                    self.wav_files.append(os.path.join(speaker_wav_dir, wav_file))


    def __len__(self):
        return len(self.text_files)


    def __getitem__(self, idx):
        with open(self.text_files[idx], 'r') as f:
            text = f.read().strip()
        res = {
            'idx': idx,
            'data': {
                'audio': self.wav_files[idx],
                'text': {
                    'question': 'Provide text content for the provided speech.',
                    'answer': text,
                }
            }
        }
        return res


if __name__ == "__main__":
    benchmark = VCTK()
    print(len(benchmark))
    print(benchmark[0])