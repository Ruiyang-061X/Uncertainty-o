import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append("/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1")
sys.path.append("/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/dependency/AnyGPT")
sys.path.append("/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/dependency/AnyGPT/anygpt/src")
from seed2.seed_llama_tokenizer import ImageTokenizer
from m_utils.prompter import *
from m_utils.anything2token import *
from m_utils.read_modality import encode_music_by_path
from infer.pre_post_process import extract_text_between_tags
from soundstorm_speechtokenizer import SoundStorm, ConformerWrapper
from transformers import  LlamaForCausalLM, LlamaTokenizer, GenerationConfig, EncodecModel, AutoProcessor
from speechtokenizer import SpeechTokenizer
import torch
import torchaudio
import os
import logging
import json
import re
import numpy as np
from PIL import Image
from datetime import datetime
from einops import rearrange
from util.misc import *


logging.basicConfig()
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_soundstorm(model_path):
    conformer = ConformerWrapper(codebook_size=1024, num_quantizers=7, conformer={'dim':1024, 'depth': 12, 'heads':8, 'dim_head': 128, 'attn_flash': False})
    soundstorm = SoundStorm(net=conformer, num_semantic_token_ids=1024, semantic_pad_id=1024, pad_id=1024, schedule = 'cosine')
    soundstorm.load(model_path)
    return soundstorm


def semantic2acoustic(semantic_tokens, prompt_tokens, soundstorm, tokenizer, steps=1, greedy=True):
    generated = soundstorm.generate(semantic_tokens=semantic_tokens, prompt_tokens=prompt_tokens, steps=steps, greedy=greedy)
    wavs = tokenizer.decode(rearrange(generated, 'b n q -> q b n', b=semantic_tokens.size(0)))
    return wavs


class AnyGPT:


    def __init__(self):
        self.build_model()


    def build_model(self):
        self.model_name_or_path='/data/lab/yan/huzhang/huzhang1/hub/models--fnlp--AnyGPT-base/snapshots/7f3d03542f4dfa7738978f5f07918490e0a932e5'
        self.image_tokenizer_path='/data/lab/yan/huzhang/huzhang1/hub/models--AILab-CVC--seed-tokenizer-2/snapshots/c6d957a9b280d9ed3879b1eb5d99036cf8390012/seed_quantizer.pt'
        self.speech_tokenizer_path='/data/lab/yan/huzhang/huzhang1/hub/models--fnlp--AnyGPT-speech-modules/snapshots/47320f954f02f108d0359fb80cd481038bba6be9/speechtokenizer/ckpt.dev'
        self.speech_tokenizer_config='/data/lab/yan/huzhang/huzhang1/hub/models--fnlp--AnyGPT-speech-modules/snapshots/47320f954f02f108d0359fb80cd481038bba6be9/speechtokenizer/config.json'
        self.soundstorm_path='/data/lab/yan/huzhang/huzhang1/hub/models--fnlp--AnyGPT-speech-modules/snapshots/47320f954f02f108d0359fb80cd481038bba6be9/soundstorm/speechtokenizer_soundstorm_mls.pt'

        self.generate_config = '/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/dependency/AnyGPT/config/generate_config.json'
        self.text_generate_config='/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/dependency/AnyGPT/config/text_generate_config.json'
        self.image_generate_config='/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/dependency/AnyGPT/config/image_generate_config.json'
        self.speech_generate_config='/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/dependency/AnyGPT/config/speech_generate_config.json'
        self.music_generate_config='/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/dependency/AnyGPT/config/music_generate_config.json'

        self.output_dir='/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/generation/audio'

        self.model = LlamaForCausalLM.from_pretrained(self.model_name_or_path, load_in_8bit=False, torch_dtype=torch.float16, device_map="auto",)
        self.model.half().eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)

        self.tokenizer = LlamaTokenizer.from_pretrained(self.model_name_or_path)
        self.tokenizer.pad_token_id = (0)
        self.tokenizer.padding_side = "left" 

        self.image_tokenizer = ImageTokenizer(model_path=self.image_tokenizer_path, load_diffusion=True, diffusion_model_path="stabilityai/stable-diffusion-2-1-unclip", device=torch.device('cuda'), image_size=224)

        self.speech_tokenizer = SpeechTokenizer.load_from_checkpoint(self.speech_tokenizer_config, self.speech_tokenizer_path)     
        self.speech_tokenizer.eval().cuda()

        self.soundstorm = load_soundstorm(self.soundstorm_path)
        self.soundstorm.eval().cuda()

        self.music_tokenizer = EncodecModel.from_pretrained("facebook/encodec_32khz")
        self.music_tokenizer.eval().cuda()

        self.music_processor = AutoProcessor.from_pretrained("facebook/encodec_32khz")
        self.music_sample_rate = 32000
        self.music_segment_duration = 20

        self.audio_tokenizer = EncodecModel.from_pretrained("facebook/encodec_24khz")
        self.audio_tokenizer.eval().cuda()

        self.audio_processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
        self.audio_sample_rate = 24000
        self.audio_segment_duration = 5

        self.prompter = Prompter()
        self.ans_file = ''

    def encode_image(self, image_path=None, image_pil=None, image_torch=None):
        assert (image_path is None) + (image_pil is None) + (image_torch is None) == 2
        if image_path is not None:
            image_pil = Image.open(image_path).convert('RGB')
        if image_pil is not None:
            image_torch = self.image_tokenizer.processor(image_pil)
            image_torch = image_torch.cuda()
        return self.image_tokenizer.encode(image_torch)
    
    
    def decode_image(self, content, negative_indices=None, guidance_scale=10):
        codes = [[int(num) for num in re.findall(r'\d+', content)]]
        indices = torch.Tensor(codes).int().cuda()
        if negative_indices is not None:
            negative_indices = negative_indices.cuda()
        image = self.image_tokenizer.decode(indices, negative_indices=negative_indices, guidance_scale=guidance_scale)[0]
        return image
     

    def encode_speech(self, audio_path):
        if '{' in audio_path:
            audio_path = audio_path.split("'")[-2]
        print(audio_path)
        wav, sr = torchaudio.load(audio_path)
        if wav.shape[0] > 1:
            wav = wav[:1, ]
        if sr != self.speech_tokenizer.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.speech_tokenizer.sample_rate)
        wav = wav.unsqueeze(0).cuda()
        with torch.no_grad():
            codes = self.speech_tokenizer.encode(wav)
        return codes[0, 0, :]
    

    def decode_speech(self, content, prompt_path=None):
        if prompt_path:
            prompt_wav, sr = torchaudio.load(prompt_path).cuda()
            if sr != self.speech_tokenizer.sample_rate:
                prompt_wav = torchaudio.functional.resample(prompt_wav, sr, self.speech_tokenizer.sample_rate)
            if prompt_wav.shape[0] == 2:
                prompt_wav = prompt_wav.mean(dim=0).unsqueeze(0)
            prompt_tokens = rearrange(self.speech_tokenizer.encode(prompt_wav.unsqueeze(0)), 'q b n -> b n q')
        else:
            prompt_tokens = None
        semantic_codes = [[int(num) for num in re.findall(r'\d+', content)]]
        config_dict = json.load(open(self.generate_config, 'r'))
        wav = semantic2acoustic(torch.Tensor(semantic_codes).int().cuda(), prompt_tokens, self.soundstorm, self.speech_tokenizer, steps=config_dict['vc_steps'])
        wav = wav.squeeze(0).detach().cpu()
        return wav
    
    
    def content2rvq_codes(self, content, codebook_size, codebook_num):
        codes = [int(code) for code in re.findall(r'\d+', content)]
        codes = np.array([code % codebook_size for code in codes])
        n = codes.shape[0] // codebook_num
        codes = codes[:n * codebook_num]
        codes = codes.reshape(n, codebook_num).T
        codes = np.expand_dims(codes, 0)
        codes = np.expand_dims(codes, 0)
        codes = torch.tensor(codes).long().cuda() 
        return codes
    
    
    def decode_music(self, content):
        codes = self.content2rvq_codes(content, music_codebook_size, music_codebook_num)
        music = self.music_tokenizer.decode(codes, [None])
        music = music[0].squeeze(0).detach().cpu()
        return music
    
    
    def decode_audio(self, content):
        codes = self.content2rvq_codes(content, audio_codebook_size, audio_codebook_num)
        audio = self.audio_tokenizer.decode(codes, [None])
        audio = audio[0].squeeze(0).detach().cpu()
        return audio
    

    def preprocess(self, input_data, modality, to_modality):
        if modality == "text":
            processed_inputs = input_data
        else:
            if modality == "image":
                tokens = self.encode_image(image_path=input_data.strip())[0]
            elif modality == "speech":
                tokens = self.encode_speech(input_data.strip())
            elif modality == "music":
                tokens = encode_music_by_path(input_data.strip(), self.music_sample_rate, self.music_tokenizer, self.music_processor, torch.device('cuda'), segment_duration=self.music_segment_duration, one_channel=True, start_from_begin=True)[0][0]
            else:
                raise TypeError("wrong modality")
            processed_inputs = modality_tokens_to_string(tokens=tokens, modality=modality)
        prompt_seq = self.prompter.generate_prompt_input(modality_str=processed_inputs, modality=modality, to_modality=to_modality)
        return prompt_seq


    def postprocess(self, response, modality, input_data, voice_prompt):
        special_dict = modal_special_str[modality]
        modality_content = extract_text_between_tags(response, tag1=special_dict['sos'], tag2=special_dict['eos'])
        if modality == "image":
            generated_image = self.decode_image(modality_content)
            self.ans_file = os.path.join(self.output_dir, get_cur_time() + '.jpg')
            generated_image.save(self.ans_file)
        elif modality == "speech":
            generated_wav = self.decode_speech(modality_content, voice_prompt)
            self.ans_file = os.path.join(self.output_dir, get_cur_time() + '.wav')
            torchaudio.save(self.ans_file, generated_wav, self.speech_tokenizer.sample_rate)
        elif modality == "music":
            generated_music = self.decode_music(modality_content)
            self.ans_file = os.path.join(self.output_dir, get_cur_time() + '.wav')
            torchaudio.save(self.ans_file, generated_music, self.music_sample_rate)
        return response
    

    def response(self, modality, to_modality, input_data, voice_prompt=None, temp=0.1):
        preprocessed_prompts = (self.preprocess(input_data=input_data, modality=modality, to_modality=to_modality))
        input_ids = self.tokenizer(preprocessed_prompts, return_tensors="pt", padding=True).input_ids
        input_ids = input_ids.cuda()
        config_path=self.text_generate_config
        if to_modality == "speech":
            config_path=self.speech_generate_config
        elif to_modality == "music":
            config_path=self.music_generate_config
        elif to_modality == "image":
            config_path=self.image_generate_config
        config_dict = json.load(open(config_path, 'r'))
        config_dict['temperature'] = temp
        generation_config = GenerationConfig(**config_dict)
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )
        generated_ids = generated_ids.sequences
        response = self.tokenizer.batch_decode(generated_ids.cpu(), skip_special_tokens=True)[0]
        if to_modality == "text":
            response = extract_text_between_tags(response, tag1=f"{chatbot_name} : ", tag2="<eos>").strip()
        else:
            self.postprocess(response, to_modality, input_data, voice_prompt)
        return response
    
    
    def forward(self, prompts, temp):
        inputs = prompts.split("|")
        modality = inputs[0]
        to_modality = inputs[1]
        input_data = inputs[2]
        voice_prompt = None
        if modality == "text" and to_modality == "speech":
            try:
                voice_prompt = inputs[3]
            except IndexError:
                voice_prompt = None
        response = self.response(modality, to_modality, input_data, voice_prompt, temp)
        return response
        

    def __call__(self, input):
        return self.forward(input)


    def generate(self, in_modal, out_modal, data, temp):
        if 'text' in in_modal and 'audio' in out_modal:
            # text to speech
            final_prompt = f"text|speech|{data['text']['answer']}"
        if 'audio' in in_modal and 'text' in out_modal:
            # speech to text
            final_prompt = f"speech|text|{data['audio']}"
        ans_text = self.forward(final_prompt, temp)
        ans = {}
        if 'text' in in_modal and 'audio' in out_modal:
            # text to speech
            ans['audio'] = self.ans_file
        if 'audio' in in_modal and 'text' in out_modal:
            # speech to text
            ans['text'] = ans_text
        return ans


if __name__=='__main__':
    mllm = AnyGPT()

    # text to image
    data = {
        'text': 'A bustling medieval market scene with vendors selling exotic goods under colorful tents',
    }
    print(data)
    ans = mllm.generate(
        in_modal=['text'],
        out_modal=['image'],
        data=data,
        temp=0.1,
    )
    print(ans)
    print('-'*100)

    # image to text
    data = {
        'image': '/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/dependency/AnyGPT/static/infer/image/cat.jpg',
    }
    print(data)
    ans = mllm.generate(
        in_modal=['image'],
        out_modal=['text'],
        data=data,
        temp=0.1,
    )
    print(ans)
    print('-'*100)
    
    # text to speech
    data = {
        'text': 'I could be bounded in a nutshell and count myself a king of infinite space.'
    }
    print(data)
    ans = mllm.generate(
        in_modal=['text'],
        out_modal=['speech'],
        data=data,
        temp=0.1,
    )
    print(ans)
    print('-'*100)

    # text to music
    data = {
        'text': 'features an indie rock sound with distinct elements that evoke a dreamy, soothing atmosphere'
    }
    print(data)
    ans = mllm.generate(
        in_modal=['text'],
        out_modal=['music'],
        data=data,
        temp=0.1,
    )
    print(ans)
    print('-'*100)

    # music to text
    data = {
        'music': '/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/dependency/AnyGPT/static/infer/music/features an indie rock sound with distinct element.wav'
    }
    print(data)
    ans = mllm.generate(
        in_modal=['music'],
        out_modal=['text'],
        data=data,
        temp=0.1,
    )
    print(ans)
    print('-'*100)