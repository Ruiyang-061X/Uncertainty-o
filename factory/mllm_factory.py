from mllm.comprehension.InternVL import InternVL
from mllm.comprehension.OneLLM import OneLLM
from mllm.comprehension.PointLLM import PointLLM
from mllm.comprehension.VideoLLaMA import VideoLLaMA
from mllm.comprehension.GPT4o import GPT4o
from mllm.comprehension.QwenVLMax import QwenVLMax
from mllm.generation.AnyGPT import AnyGPT
from mllm.generation.HunyuanVideo import HunyuanVideo
from mllm.generation.RGB2point import RGB2point
from mllm.generation.StableDiffusion import StableDiffusion
from mllm.generation.VideoFusion import VideoFusion


MLLM_MAP = {
    'InternVL': InternVL,
    'OneLLM': OneLLM,
    'PointLLM': PointLLM,
    'VideoLLaMA': VideoLLaMA,
    'AnyGPT': AnyGPT,
    'HunyuanVideo': HunyuanVideo,
    'RGB2point': RGB2point,
    'StableDiffusion': StableDiffusion,
    'VideoFusion': VideoFusion,
    'GPT4o': GPT4o,
    'QwenVLMax': QwenVLMax,
}


def obtain_mllm(name):
    mllm_class = MLLM_MAP.get(name)
    if not mllm_class:
        raise ValueError(f"Unsupported MLLM: {name}")
    return mllm_class()