from llm.Qwen import Qwen


LLM_MAP = {
    'Qwen2.5-0.5B-Instruct': Qwen,
    'Qwen2.5-1.5B-Instruct': Qwen,
    'Qwen2.5-3B-Instruct': Qwen,
    'Qwen2.5-7B-Instruct': Qwen,
}


def obtain_llm(name):
    llm_class = LLM_MAP.get(name)
    if not llm_class:
        raise ValueError(f"Unsupported LLM: {name}")
    return llm_class(name)