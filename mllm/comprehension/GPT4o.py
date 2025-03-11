import warnings
warnings.filterwarnings("ignore")
import base64
import requests
import json 


def encode_image(image_path):
    if not isinstance(image_path, str):
        image_path.save('/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/1.png')
        image_path = '/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/1.png'
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class GPT4o:


    def __init__(self):
        self.build_model()


    def build_model(self):
        self.url = "https://api.deerapi.com/v1/chat/completions"
        self.headers = {
            "Content-Type": 'application/json', 
            "Authorization": "Bearer " + "sk-xxxxxxxxxxxx"
        }


    def generate(self, in_modal, out_modal, data, temp):
        try:
            base64_image = encode_image(data['image'])
            payload = json.dumps(
                {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": data['text']['question']
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                                },
                            ],
                        }
                    ],
                    "stream": False,
                    "temperature": temp,
                    "max_new_tokens": 32,
                    "do_sample": True,

                }
            )
            response = requests.post(
                url=self.url,
                headers=self.headers,
                data=payload,
                timeout=300
            )
            ans = response.json()["choices"][0]["message"]["content"]
            res = {
                'text': ans
            }
            return res
        except Exception as e:
            print(e)
            res = {
                'text': 'Error.'
            }
            return res
    

if __name__ == "__main__":
    mllm = GPT4o()
    data = {
        'image': '/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/.asset/image_car.jpg',
        'text': {
            'question': 'What is it?',
        }
    }
    print(data)
    ans = mllm.generate(
        in_modal=['image', 'text'],
        out_modal=['text'],
        data=data,
        temp=0.1,
    )
    print(ans)