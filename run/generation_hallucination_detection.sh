# Generation Hallucination Detection

python /data/lab/yan/huzhang/huzhang1/code/Uncertainty-o/hallucination_detection.py --type generation --mllm StableDiffusion --benchmark Flickr --llm Qwen2.5-7B-Instruct --in_modal 'text' --out_modal 'image' --text_perturbation llm_rephrasing --inference_temp 0.1 --sampling_temp 0.5 --sampling_time 5;

python /data/lab/yan/huzhang/huzhang1/code/Uncertainty-o/hallucination_detection.py --type generation --mllm RGB2point --benchmark Pix3D --llm Qwen2.5-7B-Instruct --in_modal 'image' --out_modal 'point_cloud' --image_perturbation image_blurring --inference_temp 0.1 --sampling_temp 0.5 --sampling_time 5;

python /data/lab/yan/huzhang/huzhang1/code/Uncertainty-o/hallucination_detection.py --type generation --mllm AnyGPT --benchmark VCTK --llm Qwen2.5-7B-Instruct --mllm_captioner AnyGPT --in_modal 'text' --out_modal 'audio' --text_perturbation llm_rephrasing --inference_temp 0.1 --sampling_temp 0.5 --sampling_time 5;

python /data/lab/yan/huzhang/huzhang1/code/Uncertainty-o/hallucination_detection.py --type generation --mllm VideoFusion --benchmark MSRVTT --llm Qwen2.5-7B-Instruct --in_modal 'text' --out_modal 'video' --text_perturbation llm_rephrasing --inference_temp 0.1 --sampling_temp 0.5 --sampling_time 5;