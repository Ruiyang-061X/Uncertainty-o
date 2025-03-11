import warnings
warnings.filterwarnings("ignore")
from openai import OpenAI
from moviepy.editor import VideoFileClip
import os
import numpy as np
from PIL import Image
import base64


def convert_to_image_list(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = f'/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/other/'
    os.makedirs(output_dir, exist_ok=True)
    clip = VideoFileClip(video_path)
    duration = clip.duration
    if duration * 24 < 5:
        raise ValueError("Video is too short to sample 5 frames evenly.")
    timestamps = np.linspace(0, duration, 5)
    image_paths = []
    for i, t in enumerate(timestamps):
        frame = clip.get_frame(t)
        img_path = os.path.join(output_dir, f"{video_name}_{i + 1}.jpg")
        Image.fromarray(frame).save(img_path, "JPEG")
        image_paths.append(img_path)
    clip.close()
    return image_paths


def encode_image(image_path):
    if not isinstance(image_path, str):
        image_path.save('/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/1.png')
        image_path = '/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/1.png'
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class QwenVLMax:

    def __init__(self):
        self.build_model()

    def build_model(self):
        self.model = OpenAI(
            api_key="sk-xxxxxxxxxx",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def generate(self, in_modal, out_modal, data, temp):
        try:
            image_list = convert_to_image_list(data['video'])
            for i in range(len(image_list)):
                base64_image = encode_image(image_list[i])
                image_list[i] = f"data:image/jpeg;base64,{base64_image}"
            response = self.model.chat.completions.create(
                model="qwen-vl-max-latest",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": data['text']['question']
                            },
                            {
                                "type": "video",
                                "video": image_list
                            },
                        ],
                    }
                ],
                temperature=temp
            )
            ans = response.choices[0].message.content
            res = {
                'text': ans
            }
            return res
        except Exception as e:
            print(e)
            res = {
                'text': 'Error.'
            }
            return res
    

if __name__ == "__main__":
    mllm = QwenVLMax()
    data = {
        'video': '/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/.asset/horse_running.mp4',
        'text': {
            'question': 'What is it?',
        }
    }
    print(data)
    ans = mllm.generate(
        in_modal=['video', 'text'],
        out_modal=['text'],
        data=data,
        temp=0.1,
    )
    print(ans)