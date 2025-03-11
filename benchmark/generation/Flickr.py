from datasets import load_dataset
from torch.utils.data import Dataset


class Flickr(Dataset):


    def __init__(self):
        super().__init__()
        self.ds = load_dataset("nlphuji/flickr30k")


    def __len__(self):
        return len(self.ds['test'])


    def save_image(self, image, path):
        image.save(path)
        

    def __getitem__(self, idx):
        row = self.ds['test'][idx]
        path = f'/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/benchmark/Flickr/{idx}.png'
        self.save_image(row['image'], path)
        res = {
            'idx': idx,
            'data': {
                'image': path,
                'text': {
                    'question': 'Provide a one-sentence caption for the provided image.',
                    'answer': row['caption'][0],
                }
            }
        }
        return res


if __name__ == "__main__":
    benchmark = Flickr()
    print(len(benchmark))
    print(benchmark[0])