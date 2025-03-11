import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1')
sys.path.append('/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/uncertainty')
from llm.Qwen import Qwen
from uncertainty_caculation import *


def text_semantic_checking(t1, t2, llm):
    prompt = f"Text 1: {t1} Text 2: {t2} Respond with 'different' if they are totally different things. Respond with 'similar' when they share similiar semantic."
    ans = llm.generate(
        prompt,
        temp=0.1
    )
    return "similar" in ans.lower()
    

def text_semantic_uncertainty(t_list, llm, idx, log_dict):
    ans_cluster_idx = []
    ans_cluster_idx = [-1] * len(t_list)
    cur_cluster_idx = 0
    for i in range(len(t_list)):
        if ans_cluster_idx[i] == -1:
            ans_cluster_idx[i] = cur_cluster_idx
            for j in range(i + 1, len(t_list)):
                if ans_cluster_idx[j] == -1:
                    flag = text_semantic_checking(t_list[i], t_list[j], llm)
                    if flag:
                        ans_cluster_idx[j] = cur_cluster_idx
            cur_cluster_idx += 1
    if log_dict is not None:
        log_dict[idx]['ans_cluster_idx'] = ans_cluster_idx
    return uncertainty_caculation(ans_cluster_idx)


if __name__ == "__main__":
    llm = Qwen('Qwen2.5-7B-Instruct')

    ###########
    #  text
    ###########
    t1 = "A child holding a flowered umbrella and petting a yak."
    t2 = "A young boy holding an umbrella near a cow."
    res = text_semantic_checking(t1, t2, llm)
    print(res)

    t_list = [
        "A young boy walking a cow while holding an umbrella.",
        "A young boy walking through a field with a cow.",
        "A young boy holding an umbrella next to a herd of cattle.",
        "A young boy is holding an umbrella while tending to a cow.",
        "A woman is holding an umbrella and standing in front of a herd of cows.",
    ]
    log_dict = {0:{}}
    uncertainty = text_semantic_uncertainty(t_list, llm, 0, log_dict)
    print(log_dict)
    print(uncertainty)