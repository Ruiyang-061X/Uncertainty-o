import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1')
sys.path.append('/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/uncertainty')
from mllm.comprehension.OneLLM import OneLLM
from llm.Qwen import Qwen
from text_semantic_uncertainty import text_semantic_checking, text_semantic_uncertainty


def multi_modal_captioning(x, modal, mllm):
    if isinstance(x, dict):
        x = x[modal]
    caption_prompt = f"Please generate a concise caption that describes the key elements of the given {modal}."
    data = {
        modal: x,
        'text': {
            'question': caption_prompt
        }
    }
    cap = mllm.generate(
        in_modal=[modal, 'text'],
        out_modal=['text'],
        data=data,
        temp=0.1
    )
    return cap


def batch_multi_modal_captioning(x_list, modal, mllm):
    cap_list = []
    for x in x_list:
        cap_list.append(multi_modal_captioning(x, modal, mllm))
    return cap_list


def multi_modal_semantic_checking(x1, x2, modal, mllm, llm):
    cap1 = multi_modal_captioning(x1, modal, mllm)
    cap2 = multi_modal_captioning(x2, modal, mllm)
    return text_semantic_checking(cap1, cap2, llm)


def multi_modal_semantic_uncertainty(x_list, modal, mllm, llm, idx, log_dict):
    cap_list = batch_multi_modal_captioning(x_list, modal, mllm)
    log_dict[idx]['cap_list'] = cap_list
    return text_semantic_uncertainty(cap_list, llm, idx, log_dict)


if __name__ == "__main__":
    mllm = OneLLM()
    llm = Qwen('Qwen2.5-7B-Instruct')

    ###########
    #  image
    ###########
    x1 = "/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/.asset/image_car.jpg"
    x2 = "/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/.asset/image_dog.jpg"
    res = multi_modal_semantic_checking(x1, x2, 'image', mllm, llm)
    print(res)

    x_list = [
        "/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/.asset/image_car.jpg",
        "/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/.asset/image_car.jpg",
        "/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/.asset/image_car.jpg",
        "/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/.asset/image_dog.jpg",
        "/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/.asset/image_dog.jpg",
    ]
    log_dict = {0: {}}
    uncertainty = multi_modal_semantic_uncertainty(x_list, 'image', mllm, llm, 0, log_dict)
    print(log_dict)
    print(uncertainty)

    ###########
    #  audio
    ###########
    x1 = "/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/.asset/audio_dog.wav"
    x2 = "/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/.asset/audio_car.wav"
    res1 = multi_modal_semantic_checking(x1, x2, 'audio', mllm, llm)
    print(res1)

    x_list = [
        "/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/.asset/audio_dog.wav",
        "/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/.asset/audio_dog.wav",
        "/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/.asset/audio_dog.wav",
        "/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/.asset/audio_car.wav",
        "/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/.asset/audio_car.wav",
    ]
    log_dict = {0: {}}
    uncertainty = multi_modal_semantic_uncertainty(x_list, 'audio', mllm, llm, 0, log_dict)
    print(log_dict)
    print(uncertainty)

    ###########
    #  video
    ###########
    x1 = "/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/.asset/video_autos.mp4"
    x2 = "/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/.asset/video_education.mp4"
    res1 = multi_modal_semantic_checking(x1, x2, 'video', mllm, llm)
    print(res1)

    x_list = [
        "/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/.asset/video_autos.mp4",
        "/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/.asset/video_autos.mp4",
        "/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/.asset/video_autos.mp4",
        "/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/.asset/video_education.mp4",
        "/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/.asset/video_education.mp4",
    ]
    log_dict = {0: {}}
    uncertainty = multi_modal_semantic_uncertainty(x_list, 'video', mllm, llm, 0, log_dict)
    print(log_dict)
    print(uncertainty)

    #################
    #  point_cloud
    #################
    x1 = "/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/.asset/point_cloud_duck.npy"
    x2 = "/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/.asset/point_cloud_lemon.npy"
    res1 = multi_modal_semantic_checking(x1, x2, 'point_cloud', mllm, llm)
    print(res1)

    x_list = [
        "/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/.asset/point_cloud_duck.npy",
        "/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/.asset/point_cloud_duck.npy",
        "/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/.asset/point_cloud_duck.npy",
        "/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/.asset/point_cloud_lemon.npy",
        "/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/.asset/point_cloud_lemon.npy",
    ]
    log_dict = {0: {}}
    uncertainty = multi_modal_semantic_uncertainty(x_list, 'point_cloud', mllm, llm, 0, log_dict)
    print(log_dict)
    print(uncertainty)