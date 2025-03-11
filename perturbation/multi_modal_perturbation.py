import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/perturbation')
from text_perturbation import perturbation_of_text_prompt
from image_perturbation import perturbation_of_image_prompt
from audio_perturbation import perturbation_of_audio_prompt
from video_perturbation import perturbation_of_video_prompt
from point_cloud_perturbation import perturbation_of_point_cloud_prompt


def multi_modal_prompt_perturbation(args, idx, x, modal, llm):
    if modal == 'text':
        return perturbation_of_text_prompt(args, x, llm)
    elif modal == 'image':
        return perturbation_of_image_prompt(args, idx, x)
    elif modal == 'audio':
        return perturbation_of_audio_prompt(args, idx, x)
    elif modal == 'video':
        return perturbation_of_video_prompt(args, idx, x)
    elif modal == 'point_cloud':
        return perturbation_of_point_cloud_prompt(args, idx, x)
    else:
        raise ValueError(f"Unknown modality: {modal}")