# Uncertainty-Aware Chain-of-Thought

python /data/lab/yan/huzhang/huzhang1/code/Uncertainty-o/uncertainty_aware_cot.py --type comprehension --mllm InternVL --benchmark MMVet --llm Qwen2.5-7B-Instruct --in_modal 'image' 'text' --out_modal 'text' --image_perturbation image_blurring --text_perturbation llm_rephrasing --inference_temp 0.1 --sampling_temp 0.5 --sampling_time 3  --subset 0 13;