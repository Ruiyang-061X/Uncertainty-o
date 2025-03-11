import torch
from diffusers import StableDiffusion3Pipeline
from datetime import datetime


def get_cur_time():
    return datetime.today().strftime('%Y_%m_%d_%H_%M_%S')


class StableDiffusion:


    def __init__(self):
        self.build_model()


    def build_model(self):
        self.pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16)
        self.pipe = self.pipe.to("cuda")


    def generate(self, in_modal, out_modal, data, temp):
        image = self.pipe(
            data['text']['answer'],
            num_inference_steps=28,
            guidance_scale=5 * (0.8 - temp),
        ).images[0]
        image_path = f'/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/generation/image/{get_cur_time()}.jpg'
        image.save(image_path)
        ans = {
            'image': image_path
        }
        return ans


if __name__ == "__main__":
    mllm = StableDiffusion()
    data = {
        'text': {
            'answer': "A person wearing a blueshirt and hat is situated on some stairs, leaning on a windowsill.",
        }
    }
    print(data)
    ans = mllm.generate(
        in_modal=['text'],
        out_modal=['image'],
        data=data,
        temp=0.1,
    )
    print(ans)