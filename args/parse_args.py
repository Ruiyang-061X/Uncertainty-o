import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='comprehension')
    parser.add_argument('--mllm', type=str, default='AnyGPT')
    parser.add_argument('--benchmark', type=str, default='ClothoV2')
    parser.add_argument('--mllm_captioner', type=str, default='OneLLM')
    parser.add_argument('--llm', type=str, default='Qwen2.5-7B-Instruct')
    parser.add_argument('--in_modal', type=str, nargs='+', default=['image'])
    parser.add_argument('--out_modal', type=str, nargs='+', default=['text'])
    parser.add_argument('--text_perturbation', type=str, default='llm_rephrasing')
    parser.add_argument('--image_perturbation', type=str, default='blurring')
    parser.add_argument('--audio_perturbation', type=str, default='adjust_volume')
    parser.add_argument('--video_perturbation', type=str, default='play_speed')
    parser.add_argument('--point_cloud_perturbation', type=str, default='random_sampling')
    parser.add_argument('--inference_temp', type=float, default=0.1)
    parser.add_argument('--sampling_temp', type=float, default=1.0)
    parser.add_argument('--sampling_time', type=int, default=5)
    parser.add_argument('--exp_dir', type=str, default='/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/exp')
    parser.add_argument('--output_dir', type=str, default='/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--mllm_answer_json', type=str, default='')
    parser.add_argument('--subset', type=int, nargs='+', default=[0, 127])
    parser.add_argument('--step_cnt_thres', default=8)
    args = parser.parse_args()
    return args