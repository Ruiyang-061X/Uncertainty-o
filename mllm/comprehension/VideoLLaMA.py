import warnings
warnings.filterwarnings("ignore")
import torch
from transformers import AutoModelForCausalLM, AutoProcessor


class VideoLLaMA:


    def __init__(self):
        self.build_model()


    def build_model(self):
        self.device = "cuda:0"
        self.model_path = "DAMO-NLP-SG/VideoLLaMA3-7B"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            device_map={"": self.device},
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)


    def generate(self, in_modal, out_modal, data, temp):
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": {"video_path": data['video'], "fps": 1, "max_frames": 128}},
                    {"type": "text", "text": data['text']['question']},
                ]
            },
        ]
        inputs = self.processor(
            conversation=conversation,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        output_ids = self.model.generate(**inputs, max_new_tokens=128)
        response = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        res = {
            'text': response
        }
        return res


if __name__ == "__main__":
    mllm = VideoLLaMA()
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