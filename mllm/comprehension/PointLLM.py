import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/dependency/PointLLM')
from transformers import AutoTokenizer
import torch
import numpy as np
from pointllm.conversation import conv_templates, SeparatorStyle
from pointllm.utils import disable_torch_init
from pointllm.model import *
from pointllm.model.utils import KeywordsStoppingCriteria


def load_point_cloud(pc):
    if isinstance(pc, str):
        pc = np.load(pc)
    pc = torch.tensor(pc)
    pc = pc.unsqueeze_(0).to(torch.float32)
    return pc


class PointLLM:


    def __init__(self):
        self.build_model()


    def build_model(self):
        disable_torch_init()
        self.model_path = 'RunsenXu/PointLLM_7B_v1.2' 
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = PointLLMLlamaForCausalLM.from_pretrained(self.model_path, low_cpu_mem_usage=False, use_cache=True, torch_dtype=torch.float16).cuda()
        self.model.initialize_tokenizer_point_backbone_config_wo_embedding(self.tokenizer)
        self.model.eval()
        self.mm_use_point_start_end = getattr(self.model.config, "mm_use_point_start_end", False)
        self.point_backbone_config = self.model.get_model().point_backbone_config
        if self.mm_use_point_start_end:
            if "v1" in self.model_path.lower():
                conv_mode = "vicuna_v1_1"
            else:
                raise NotImplementedError
            self.conv = conv_templates[conv_mode].copy()
        stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2
        self.keywords = [stop_str]


    def generate(self, in_modal, out_modal, data, temp):
        point_token_len = self.point_backbone_config['point_token_len']
        default_point_patch_token = self.point_backbone_config['default_point_patch_token']
        default_point_start_token = self.point_backbone_config['default_point_start_token']
        default_point_end_token = self.point_backbone_config['default_point_end_token']
        point_clouds = load_point_cloud(data['point_cloud'])
        point_clouds = point_clouds.cuda().to(torch.float16)
        self.conv.reset()
        qs = data['text']['question']
        if self.mm_use_point_start_end:
            qs = default_point_start_token + default_point_patch_token * point_token_len + default_point_end_token + '\n' + qs
        else:
            qs = default_point_patch_token * point_token_len + '\n' + qs
        self.conv.append_message(self.conv.roles[0], qs)
        self.conv.append_message(self.conv.roles[1], None)
        prompt = self.conv.get_prompt()
        inputs = self.tokenizer([prompt])
        input_ids = torch.as_tensor(inputs.input_ids).cuda()
        stopping_criteria = KeywordsStoppingCriteria(self.keywords, self.tokenizer, input_ids)
        stop_str = self.keywords[0]
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                point_clouds=point_clouds,
                do_sample=True,
                temperature=temp,
                top_k=50,
                max_length=2048,
                top_p=0.95,
                stopping_criteria=[stopping_criteria])
            input_token_len = input_ids.shape[1]
            outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
            res = {
                'text': outputs
            }
            return res


if __name__ == "__main__":
    mllm = PointLLM()
    data = {
        'point_cloud': '/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/.asset/horse.npy',
        'text': {
            'question': 'What is it?',
        }
    }
    print(data)
    ans = mllm.generate(
        in_modal=['point_cloud', 'text'],
        out_modal=['text'],
        data=data,
        temp=0.1,
    )
    print(ans)