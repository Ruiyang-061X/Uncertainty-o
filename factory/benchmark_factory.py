from benchmark.comprehension.AudioCaps import AudioCaps
from benchmark.comprehension.ClothoV2 import ClothoV2
from benchmark.comprehension.CoCoCap import CoCoCap
from benchmark.comprehension.LLaVABench import LLaVABench
from benchmark.comprehension.MMVet import MMVet
from benchmark.comprehension.ModelNet import ModelNet
from benchmark.comprehension.MSRVTTQA import MSRVTTQA
from benchmark.comprehension.MSVDQA import MSVDQA
from benchmark.comprehension.NextQA import NextQA
from benchmark.comprehension.PointCap import PointCap
from benchmark.comprehension.ShapeNet import ShapeNet
from benchmark.comprehension.ClothoAQA import ClothoAQA
from benchmark.comprehension.MIMICCXR import MIMICCXR
from benchmark.comprehension.OpenEQA import OpenEQA
from benchmark.generation.Flickr import Flickr
from benchmark.generation.MSRVTT import MSRVTT
from benchmark.generation.Pix3D import Pix3D
from benchmark.generation.VCTK import VCTK


BENCHMARK_MAP = {
    'AudioCaps': AudioCaps,
    'ClothoV2': ClothoV2,
    'CoCoCap': CoCoCap,
    'LLaVABench': LLaVABench,
    'MMVet': MMVet,
    'ModelNet': ModelNet,
    'MSRVTTQA': MSRVTTQA,
    'MSVDQA': MSVDQA,
    'NextQA': NextQA,
    'PointCap': PointCap,
    'Flickr': Flickr,
    'MSRVTT': MSRVTT,
    'Pix3D': Pix3D,
    'VCTK': VCTK,
    'ShapeNet': ShapeNet,
    'ClothoAQA': ClothoAQA,
    'MIMICCXR': MIMICCXR,
    'OpenEQA': OpenEQA,
}


def obtain_benchmark(name):
    benchmark_class = BENCHMARK_MAP.get(name)
    if not benchmark_class:
        raise ValueError(f"Unsupported benchmark: {name}")
    return benchmark_class()