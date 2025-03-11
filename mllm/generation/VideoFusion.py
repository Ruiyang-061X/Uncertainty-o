import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from datetime import datetime


def get_cur_time():
    return datetime.today().strftime('%Y_%m_%d_%H_%M_%S')


class VideoFusion:


    def __init__(self):
        self.build_model()


    def build_model(self):
        self.pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_model_cpu_offload()


    def generate(self, in_modal, out_modal, data, temp):
        video_frames = self.pipe(
            data['text']['answer'],
            num_inference_steps=25
        ).frames
        video_path = f'/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/generation/video/{get_cur_time()}.mp4'
        export_to_video(video_frames.squeeze(), video_path)
        ans = {
            'video': video_path
        }
        return ans


if __name__ == "__main__":
    mllm = VideoFusion()
    data = {
        'text': {
            'answer': "Spiderman is surfing",
        }
    }
    print(data)
    ans = mllm.generate(
        in_modal=['text'],
        out_modal=['video'],
        data=data,
        temp=0.1,
    )
    print(ans)