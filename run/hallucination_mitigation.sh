# Hallucination Mitigation

python /data/lab/yan/huzhang/huzhang1/code/Uncertainty-o/hallucination_mitigation.py --type comprehension --mllm VideoLLaMA --benchmark MSRVTTQA --llm Qwen2.5-7B-Instruct --in_modal 'video' 'text' --out_modal 'text' --video_perturbation play_speed --text_perturbation llm_rephrasing --inference_temp 0.1 --sampling_temp 0.5 --sampling_time 5 --subset 55 85;