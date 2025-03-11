# Comprehension Hallucination Detection

python /data/lab/yan/huzhang/huzhang1/code/Uncertainty-o/hallucination_detection.py --type comprehension --mllm InternVL --benchmark LLaVABench --llm Qwen2.5-7B-Instruct --in_modal 'image' 'text' --out_modal 'text' --image_perturbation image_blurring --text_perturbation llm_rephrasing --inference_temp 0.1 --sampling_temp 0.5 --sampling_time 5;

python /data/lab/yan/huzhang/huzhang1/code/Uncertainty-o/hallucination_detection.py --type comprehension --mllm InternVL --benchmark MMVet --llm Qwen2.5-7B-Instruct --in_modal 'image' 'text' --out_modal 'text' --image_perturbation image_blurring --text_perturbation llm_rephrasing --inference_temp 0.1 --sampling_temp 0.5 --sampling_time 5;

python /data/lab/yan/huzhang/huzhang1/code/Uncertainty-o/hallucination_detection.py --type comprehension --mllm InternVL --benchmark CoCoCap --llm Qwen2.5-7B-Instruct --in_modal 'image' 'text' --out_modal 'text' --image_perturbation image_blurring --text_perturbation llm_rephrasing --inference_temp 0.1 --sampling_temp 0.5 --sampling_time 5;

python /data/lab/yan/huzhang/huzhang1/code/Uncertainty-o/hallucination_detection.py --type comprehension --mllm VideoLLaMA --benchmark MSRVTTQA --llm Qwen2.5-7B-Instruct --in_modal 'video' 'text' --out_modal 'text' --video_perturbation play_speed --text_perturbation llm_rephrasing --inference_temp 0.1 --sampling_temp 0.5 --sampling_time 5;

python /data/lab/yan/huzhang/huzhang1/code/Uncertainty-o/hallucination_detection.py --type comprehension --mllm VideoLLaMA --benchmark MSVDQA --llm Qwen2.5-7B-Instruct --in_modal 'video' 'text' --out_modal 'text' --video_perturbation play_speed --text_perturbation llm_rephrasing --inference_temp 0.1 --sampling_temp 0.5 --sampling_time 5;

python /data/lab/yan/huzhang/huzhang1/code/Uncertainty-o/hallucination_detection.py --type comprehension --mllm VideoLLaMA --benchmark NextQA --llm Qwen2.5-7B-Instruct --in_modal 'video' 'text' --out_modal 'text' --video_perturbation play_speed --text_perturbation llm_rephrasing --inference_temp 0.1 --sampling_temp 0.5 --sampling_time 5;

python /data/lab/yan/huzhang/huzhang1/code/Uncertainty-o/hallucination_detection.py --type comprehension --mllm OneLLM --benchmark ClothoV2 --llm Qwen2.5-7B-Instruct --in_modal 'audio' 'text' --out_modal 'text' --audio_perturbation adjust_volume --text_perturbation llm_rephrasing --inference_temp 0.1 --sampling_temp 0.5 --sampling_time 5;

python /data/lab/yan/huzhang/huzhang1/code/Uncertainty-o/hallucination_detection.py --type comprehension --mllm OneLLM --benchmark ClothoAQA --llm Qwen2.5-7B-Instruct --in_modal 'audio' 'text' --out_modal 'text' --audio_perturbation adjust_volume --text_perturbation llm_rephrasing --inference_temp 0.1 --sampling_temp 0.5 --sampling_time 5;

python /data/lab/yan/huzhang/huzhang1/code/Uncertainty-o/hallucination_detection.py --type comprehension --mllm OneLLM --benchmark AudioCaps --llm Qwen2.5-7B-Instruct --in_modal 'audio' 'text' --out_modal 'text' --audio_perturbation adjust_volume --text_perturbation llm_rephrasing --inference_temp 0.1 --sampling_temp 0.5 --sampling_time 5;

python /data/lab/yan/huzhang/huzhang1/code/Uncertainty-o/hallucination_detection.py --type comprehension --mllm PointLLM --benchmark ModelNet --llm Qwen2.5-7B-Instruct --in_modal 'point_cloud' 'text' --out_modal 'text' --point_cloud_perturbation random_sampling --text_perturbation llm_rephrasing --inference_temp 0.1 --sampling_temp 0.5 --sampling_time 5;

python /data/lab/yan/huzhang/huzhang1/code/Uncertainty-o/hallucination_detection.py --type comprehension --mllm PointLLM --benchmark ShapeNet --llm Qwen2.5-7B-Instruct --in_modal 'point_cloud' 'text' --out_modal 'text' --point_cloud_perturbation random_sampling --text_perturbation llm_rephrasing --inference_temp 0.1 --sampling_temp 0.5 --sampling_time 5;

python /data/lab/yan/huzhang/huzhang1/code/Uncertainty-o/hallucination_detection.py --type comprehension --mllm PointLLM --benchmark PointCap --llm Qwen2.5-7B-Instruct --in_modal 'point_cloud' 'text' --out_modal 'text' --point_cloud_perturbation random_sampling --text_perturbation llm_rephrasing --inference_temp 0.1 --sampling_temp 0.5 --sampling_time 5;

# Hallucination Detection for Closed-Source LMMs

python /data/lab/yan/huzhang/huzhang1/code/Uncertainty-o/hallucination_detection.py --type comprehension --mllm GPT4o --benchmark MMVet --llm Qwen2.5-7B-Instruct --in_modal 'image' 'text' --out_modal 'text' --image_perturbation image_blurring --text_perturbation llm_rephrasing --inference_temp 0.1 --sampling_temp 0.5 --sampling_time 5;

python /data/lab/yan/huzhang/huzhang1/code/Uncertainty-o/hallucination_detection.py --type comprehension --mllm QwenVLMax --benchmark MSRVTTQA --llm Qwen2.5-7B-Instruct --in_modal 'video' 'text' --out_modal 'text' --video_perturbation play_speed --text_perturbation llm_rephrasing --inference_temp 0.1 --sampling_temp 0.5 --sampling_time 5;


# Hallucination Detection for Safety-Critic Tasks

python /data/lab/yan/huzhang/huzhang1/code/Uncertainty-o/hallucination_detection.py --type comprehension --mllm InternVL --benchmark MIMICCXR --llm Qwen2.5-7B-Instruct --in_modal 'image' 'text' --out_modal 'text' --image_perturbation image_blurring --text_perturbation llm_rephrasing --inference_temp 0.1 --sampling_temp 0.5 --sampling_time 5;

python /data/lab/yan/huzhang/huzhang1/code/Uncertainty-o/hallucination_detection.py --type comprehension --mllm VideoLLaMA --benchmark OpenEQA --llm Qwen2.5-7B-Instruct --in_modal 'video' 'text' --out_modal 'text' --video_perturbation play_speed --text_perturbation llm_rephrasing --inference_temp 0.1 --sampling_temp 0.5 --sampling_time 5;
