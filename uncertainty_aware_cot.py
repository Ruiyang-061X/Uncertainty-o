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


def inference(args, data, mllm):
    ans = mllm.generate(
        in_modal=args.in_modal,
        out_modal=args.out_modal,
        data=data,
        temp=args.inference_temp
    )
    return ans


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
        return text_semantic_uncertainty(ans_sampling_list, llm, idx, None)
    return multi_modal_semantic_uncertainty(ans_sampling_list, args.out_modal[0], mllm_captioner, llm, idx, None)


def check_finish(ans):
    return 'Finish' in ans


def obtain_cot_prompt(original_question, step_cnt, ans_list, uncertainty_list):
    if step_cnt == 0:
        return f"Question: {original_question}. Let think step-by-step. Provide your first step of thinking: "
    cot_prompt = f"Question: {original_question}"
    for i in range(step_cnt):
        cot_prompt += f" Step {i + 1} Answer {ans_list[i]['text']} Uncertainty Score {abs(uncertainty_list[i])} Range (0.0 - 1.0);"
    cot_prompt += " Let think step-by-step. Answer 'Finish' when you have finish thinking. Provide your next step of thinking: "
    return cot_prompt


def cot(args, idx, mllm, benchmark, mllm_captioner, llm, log_dict):
    data = benchmark[idx]['data']
    log_dict[idx]['data'] = data
    step_cnt = 0
    cot_prompt_list = []
    ans_list = []
    uncertainty_list = []
    original_question = data['text']['question']
    data['text']['original_question'] = original_question
    while True:
        cot_prompt = obtain_cot_prompt(original_question, step_cnt, ans_list, uncertainty_list)
        cot_prompt_list.append(cot_prompt)
        data['text']['question'] = cot_prompt
        ans = inference(args, data, mllm)
        ans_list.append(ans)
        prompt_list = perturbation(args, idx, data, llm)
        ans_sampling_list = sampling(args, prompt_list, mllm)
        uncertainty = uncertainty_estimation(args, idx, ans_sampling_list, mllm_captioner, llm, log_dict)
        uncertainty_list.append(uncertainty)
        step_cnt += 1
        if check_finish(ans['text']) or step_cnt >= args.step_cnt_thres:
            break
    log_dict[idx]['step_cnt'] = step_cnt
    log_dict[idx]['cot_prompt_list'] = cot_prompt_list
    log_dict[idx]['ans_list'] = ans_list
    log_dict[idx]['uncertainty_list'] = uncertainty_list


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
    step_cnt_list = []
    for idx in tqdm(range(benchmark_size)):
        if len(args.subset) > 0 and (idx < args.subset[0] or idx > args.subset[1]):
            continue
        log_dict[idx] = {}
        cot(args, idx, mllm, benchmark, mllm_captioner, llm, log_dict)
        step_cnt_list.append(log_dict[idx]['step_cnt'])
    total_cnt = benchmark_size if len(args.subset) == 0 else args.subset[1] - args.subset[0] + 1
    log_dict['# Step'] = round(sum(step_cnt_list) / total_cnt, 1)
    log_dict['end_time_str'] = get_cur_time()
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
    log_path = f'{args.exp_dir}/log_{begin_time_str}.json'
    if args.debug:
        log_path = log_path.replace('.json', '_deubg.json')
    with open(log_path, "w") as f:
        json.dump(log_dict, f, default=custom_serializer, indent=4)
    print(f"- Full log is saved at {log_path}.")


def main():
    args = parse_args()
    mllm = obtain_mllm(args.mllm)
    benchmark = obtain_benchmark(args.benchmark)
    llm = obtain_llm(args.llm)
    if args.out_modal != ['text'] and args.mllm_captioner != args.mllm:
        mllm_captioner = obtain_mllm(args.mllm_captioner)
    else:
        mllm_captioner = mllm
    batch_process(args, mllm, benchmark, mllm_captioner, llm)


if __name__ == "__main__":
    main()