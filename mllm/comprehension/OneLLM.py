import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/dependency/OneLLM')
sys.path.append('/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1')
from dependency.OneLLM.model.meta import MetaModel
from dependency.OneLLM.data.conversation_lib import conv_templates
from dependency.OneLLM.util.misc import default_tensor_type
from dependency.OneLLM.util.misc import setup_for_distributed
from dependency.OneLLM.data import video_utils
from dependency.OneLLM.data.data_utils import make_audio_features
import numpy as np
from PIL import Image
import multiprocessing as mp
from fairscale.nn.model_parallel import initialize as fs_init
import torch
import torch.distributed as dist
import torchvision.transforms as transforms


def load_image(image):
    if isinstance(image, dict):
        image = image['image']
    if isinstance(image, str):
        image = Image.open(image)
    T_resized_center_crop = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])
    return T_resized_center_crop(image)


def load_audio(audio):
    fbank = make_audio_features(audio, mel_bins=128)
    fbank = fbank.transpose(0, 1)[None]
    return fbank


def load_video(video_path):
    video_feats = video_utils.load_and_transform_video_data(video_path, video_path, clip_duration=1, clips_per_video=5)
    return video_feats[:, :, 0]


def to_8192(point_cloud):
    try:
        num_points = point_cloud.shape[0]
        if num_points < 8192:
            padding = np.zeros((8192 - num_points, 6))
            point_cloud_8192 = np.vstack((point_cloud, padding))
        elif num_points > 8192:
            indices = np.random.choice(num_points, 8192, replace=False)
            point_cloud_8192 = point_cloud[indices]
        else:
            point_cloud_8192 = point_cloud
        return point_cloud_8192
    except:
        return point_cloud
    

def load_point_cloud(pc):
    if isinstance(pc, dict):
        pc = pc['point_cloud']
    if isinstance(pc, str):
        pc = np.load(pc)
    pc = to_8192(pc)
    if pc.shape[1] == 3:  # If the point cloud only has XYZ coordinates
        zeros = np.zeros((pc.shape[0], 3))  # Create an array of black colors (RGB = [0,0,0])
        pc = np.hstack((pc, zeros))  # Concatenate to make it N,6
    pc = torch.tensor(pc)
    return pc


class OneLLM:


    def __init__(self):
        self.build_model()


    def build_model(self):
        if mp.get_start_method(allow_none=True) != "spawn":
            mp.set_start_method("spawn", force=True)
        dist.init_process_group(backend="nccl", rank=0, world_size=1, init_method=f"tcp://127.0.0.1:23181")
        fs_init.initialize_model_parallel(1)
        setup_for_distributed(True)
        with default_tensor_type(dtype=torch.float16, device="cuda"):
            self.model = MetaModel(
                "onellm",
                "/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/dependency/OneLLM/config/llama2/7B.json",
                None,
                "/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/dependency/OneLLM/config/llama2/tokenizer.model"
            )
        pretrained_path = "/data/lab/yan/huzhang/huzhang1/hub/models--csuhan--OneLLM-7B/snapshots/e8bd281d09b64620759ba6c5273684817dc9a331/consolidated.00-of-01.pth"
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.half().cuda().eval()


    def modal_convert(self, modal):
        if modal == 'point_cloud':
            return 'point'
        return modal

    
    def build_prompt(self, data):
        conv = conv_templates["v1"].copy()
        conv.append_message(conv.roles[0], data['text']['question'])
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        return prompt


    def generate(self, in_modal, out_modal, data, temp):
        if 'image' in in_modal:
            m = 'image'
            d = load_image(data['image'])
        elif 'audio' in in_modal:
            m = 'audio'
            d = load_audio(data['audio'])
        elif 'video' in in_modal:
            m = 'video'
            d = load_video(data['video'])
        elif 'point_cloud' in in_modal:
            m = 'point_cloud'
            d = load_point_cloud(data['point_cloud'])
        d = torch.tensor([d.numpy()]).cuda().to(torch.float16)
        prompt = self.build_prompt(data)
        # print('-'*100)
        # print(d.shape)
        # print('-'*100)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            responses = self.model.generate(
                prompts=[prompt],
                images=d,
                max_gen_len=128,
                temperature=temp,
                top_p=0.75,
                modal=self.modal_convert(m))
            response = responses[0][len(prompt):].split('###')[0].strip()
        ans = {
            'text': response
        }
        return ans


if __name__ == "__main__":
    mllm = OneLLM()
    data = {
        'image': '/data/lab/yan/huzhang/huzhang1/data/CoCoCap/image/val2014/COCO_val2014_000000000042.jpg',
        'text': {
            'question': 'Provide a one-sentence caption for the provided image.',
        }
    }
    print(data)
    ans = mllm.generate(
        in_modal=['image', 'text'],
        out_modal=['text'],
        data=data,
        temp=0.1,
    )
    print(ans)

    data = {
        'audio': '/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/dependency/OneLLM/.asset/data/_Stream 2 at Krka falls.wav',
        'text': {
            'question': 'Provide a one-sentence caption for the provided audio.',
        }
    }
    print(data)
    ans = mllm.generate(
        in_modal=['audio', 'text'],
        out_modal=['text'],
        data=data,
        temp=0.1,
    )
    print(ans)

    data = {
        'video': '/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/dependency/OneLLM/.asset/data/_0nX-El-ySo_83_93.avi',
        'text': {
            'question': 'Provide a one-sentence caption for the provided video.',
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

    data = {
        'point_cloud': '/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/dependency/OneLLM/.asset/data/0a9893005a7a4985ba1ee8625d0f859f_8192.npy',
        'text': {
            'question': 'Provide a one-sentence caption for the provided point cloud.',
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