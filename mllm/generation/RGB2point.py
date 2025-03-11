import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append("/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/dependency/RGB2point")
sys.path.append("/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1")
from model_RGB2point import PointCloudNet
from utils_RGB2point import predict
import torch
import time
from util.misc import *


class RGB2point:


    def __init__(self):
        self.build_model()


    def build_model(self):
        model_save_name = "/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/dependency/RGB2point/pc1024_three.pth"
        self.model = PointCloudNet(num_views=1, point_cloud_size=1024, num_heads=4, dim_feedforward=2048)
        self.model.load_state_dict(torch.load(model_save_name)["model"])
        self.model.eval()


    def generate(self, in_modal, out_modal, data, temp):
        save_path = f'/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/generation/point_cloud/{get_cur_time()}.ply'
        predict(self.model, data[in_modal[0]], save_path)
        npy_save_path = ply_to_npy(save_path)
        ans = {
            out_modal[0]: npy_save_path
        }
        time.sleep(1)
        return ans


if __name__ == "__main__":
    mllm = RGB2point()
    data = {
        'image': '/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/dependency/RGB2point/img/1013.jpg'
    }
    print(data)
    ans = mllm.generate(
        in_modal=['image'],
        out_modal=['point_cloud'],
        data=data,
        temp=0.1,
    )
    print(ans)