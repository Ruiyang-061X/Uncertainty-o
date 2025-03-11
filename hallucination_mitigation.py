import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1')
from factory.mllm_factory import obtain_mllm
from factory.benchmark_factory import obtain_benchmark
from factory.llm_factory import obtain_llm
from perturbation.image_perturbation import *
from perturbation.audio_perturbation import *
from perturbation.video_perturbation import *
from perturbation.point_cloud_perturbation import *
from perturbation.text_perturbation import *
from perturbation.multi_modal_perturbation import *
from uncertainty.text_semantic_uncertainty import *
from uncertainty.multi_modal_semantic_uncertainty import *
from metric.AUROC import *
from metric.AURAC import *
from metric.ECE import *
from util.misc import *
from args.parse_args import parse_args
from tqdm import tqdm
import json 
import os
import copy
import torch


def inference(args, data, mllm):
    ans = mllm.generate(
        in_modal=args.in_modal,
        out_modal=args.out_modal,
        data=data,
        temp=args.inference_temp
    )
    return ans


def answer_checking(args, ans, data, mllm_captioner, llm):
    if 'text' in args.out_modal:
        return text_semantic_checking(ans['text'], data['text']['answer'], llm)
    return multi_modal_semantic_checking(ans[args.out_modal[0]], data[args.out_modal[0]], args.out_modal[0], mllm_captioner, llm)


def perturbation(args, idx, data, llm):
    prompt_list = [copy.deepcopy(data) for _ in range(args.sampling_time)]
    for modal in args.in_modal:
        if modal == 'text':
            text_key = 'question' if args.type == 'comprehension' else 'answer'
            x = data[modal][text_key]
        else:
            x = data[modal]
        perturbed_list = multi_modal_prompt_perturbation(args, idx, x, modal, llm)
        for i in range(args.sampling_time):
            if modal == 'text':
                prompt_list[i][modal][text_key] = perturbed_list[i]
            else:
                prompt_list[i][modal] = perturbed_list[i]
    return prompt_list


def sampling(args, prompt_list, mllm):
    ans_sampling_list = []
    for i in range(args.sampling_time):
        ans = mllm.generate(
            in_modal=args.in_modal,
            out_modal=args.out_modal,
            data=prompt_list[i],
            temp=args.sampling_temp
        )
        ans_sampling_list.append(ans)
    return ans_sampling_list


def uncertainty_estimation(args, idx, ans_sampling_list, mllm_captioner, llm, log_dict):
    if 'text' in args.out_modal:
        return text_semantic_uncertainty(ans_sampling_list, llm, idx, log_dict)
    return multi_modal_semantic_uncertainty(ans_sampling_list, args.out_modal[0], mllm_captioner, llm, idx, log_dict)


def mitigation(args, log_dict, idx, data, mllm, ans, flag_ans_correct, mllm_captioner, uncertainty, llm):
    if not flag_ans_correct:
        revise_prompt = f"Question: {data['text']['question']} Answer: {ans['text']} We find this answer have high uncertainty score of {abs(uncertainty)} (0 - 1.0), improve your answer:"
        data['text']['initial_question'] = data['text']['question']
        data['text']['question'] = revise_prompt
        revised_ans = inference(args, data, mllm)
        flag_revised_ans_correct = answer_checking(args, revised_ans, data, mllm_captioner, llm)
        if flag_revised_ans_correct:
            log_dict[idx]['flag_ans_correct'] = flag_revised_ans_correct


def process(args, idx, mllm, benchmark, mllm_captioner, llm, log_dict):
    data = benchmark[idx]['data']
    log_dict[idx]['data'] = data
    ans = inference(args, data, mllm)
    log_dict[idx]['ans'] = ans
    flag_ans_correct = answer_checking(args, ans, data, mllm_captioner, llm)
    log_dict[idx]['flag_ans_correct'] = flag_ans_correct
    prompt_list = perturbation(args, idx, data, llm)
    log_dict[idx]['prompt_list'] = prompt_list
    ans_sampling_list = sampling(args, prompt_list, mllm)
    log_dict[idx]['ans_sampling_list'] = ans_sampling_list
    uncertainty = uncertainty_estimation(args, idx, ans_sampling_list, mllm_captioner, llm, log_dict)
    log_dict[idx]['uncertainty'] = uncertainty
    mitigation(args, log_dict, idx, data, mllm, ans, flag_ans_correct, mllm_captioner, uncertainty, llm)


def batch_process(args, mllm, benchmark, mllm_captioner, llm):
    log_dict = {}
    log_dict['args'] = str(args)
    begin_time_str = get_cur_time()
    log_dict['begin_time_str'] = begin_time_str
    benchmark_size = len(benchmark)
    if args.debug:
        benchmark_size = 1
        args.subset = []
    log_dict['benchmark_size'] = benchmark_size
    ground_truth_list = []
    uncertainty_list = []
    for idx in tqdm(range(benchmark_size)):
        if len(args.subset) > 0 and (idx < args.subset[0] or idx > args.subset[1]):
            continue
        log_dict[idx] = {}
        process(args, idx, mllm, benchmark, mllm_captioner, llm, log_dict)
        ground_truth_list.append(0 if log_dict[idx]['flag_ans_correct'] else 1)
        uncertainty_list.append(log_dict[idx]['uncertainty'])
    total_cnt = benchmark_size if len(args.subset) == 0 else args.subset[1] - args.subset[0] + 1
    log_dict['ACC'] = round(ground_truth_list.count(0) / total_cnt * 100, 1)
    log_dict['end_time_str'] = get_cur_time()
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
    log_path = f'{args.exp_dir}/log_{begin_time_str}.json'
    if args.debug:
        log_path = log_path.replace('.json', '_deubg.json')
    with open(log_path, "w") as f:
        json.dump(log_dict, f, default=custom_serializer, indent=4)
    print(f"- Full log is saved at {log_path}.")


def fix_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    fix_seed(0)
    args = parse_args()
    mllm = obtain_mllm(args.mllm)
    benchmark = obtain_benchmark(args.benchmark)
    llm = obtain_llm(args.llm)
    if args.mllm_captioner != args.mllm:
        mllm_captioner = obtain_mllm(args.mllm_captioner)
    else:
        mllm_captioner = mllm
    batch_process(args, mllm, benchmark, mllm_captioner, llm)


if __name__ == "__main__":
    main()