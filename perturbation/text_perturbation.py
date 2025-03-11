import string
import random
import sys
sys.path.append('/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1')
from llm.Qwen import Qwen


def llm_rephrasing(text, llm, temperature, text_perturbation_instruction_template="Given the input text: '{text}', generate a semantically equivalent variation by changing the wording, structure, grammar, or narrative. Ensure the perturbed text maintains the same meaning as the original. Provide only the rephrased text as the output."):
    try:
        instruction = text_perturbation_instruction_template.replace("{text}", text)
        text_perturbed = llm.generate(
            instruction,
            temperature
        )
        return text_perturbed
    except Exception as e:
        print(e)
        return text


def word_swapping(text):
    try:
        words = text.split()
        if len(words) < 2:
            return text
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
        text_perturbed = ' '.join(words)
        return text_perturbed
    except:
        return text


def word_deleting(text):
    try:
        words = text.split()
        if len(words) <= 1:
            return text
        idx = random.randint(0, len(words) - 1)
        del words[idx]
        text_perturbed = ' '.join(words)
        return text_perturbed
    except:
        return text


def word_inserting(text):
    try:
        words = text.split()
        if not words:
            return text
        word_to_insert = random.choice(words)
        idx = random.randint(0, len(words))
        words.insert(idx, word_to_insert)
        text_perturbed = ' '.join(words)
        return text_perturbed
    except:
        return text


def word_replacing(text):
    try:
        words = text.split()
        if len(words) < 2:
            return text
        idx_to_replace = random.randint(0, len(words) - 1)
        replacement_word = random.choice([w for i, w in enumerate(words) if i != idx_to_replace])
        words[idx_to_replace] = replacement_word
        text_perturbed = ' '.join(words)
        return text_perturbed
    except:
        return text


def word_shuffle(text):
    try:
        words = text.split()
        random.shuffle(words)
        text_perturbed = ' '.join(words)
        return text_perturbed
    except:
        return text


def word_dropout(text, p):
    try:
        words = text.split()
        dropped_words = [word for word in words if random.random() > p]
        text_perturbed = ' '.join(dropped_words)
        return text_perturbed
    except:
        return text


def character_dropout(text, p):
    try:
        chars = [char for char in text if random.random() > p]
        text_perturbed = ''.join(chars)
        return text_perturbed
    except:
        return text


def noise_injection(text, noise_level):
    try:
        chars = list(text)
        num_noisy_chars = int(noise_level * len(chars))
        for _ in range(num_noisy_chars):
            idx = random.randint(0, len(chars) - 1)
            chars[idx] = random.choice(string.ascii_letters)
        text_perturbed = ''.join(chars)
        return text_perturbed
    except:
        return text


def perturbation_of_text_prompt(args, text, llm):
    perturbed_text_list = []
    if args.text_perturbation == 'llm_rephrasing':
        for i in [1.0, 1.5, 2.0, 2.5, 3.0]:
            perturbed_text = llm_rephrasing(text, llm, i)
            perturbed_text_list.append(perturbed_text)
    if args.text_perturbation == 'swapping':
        for _ in range(args.sampling_time):
            perturbed_text = word_swapping(text)
            perturbed_text_list.append(perturbed_text)
    elif args.text_perturbation == 'deleting':
        for _ in range(args.sampling_time):
            perturbed_text = word_deleting(text)
            perturbed_text_list.append(perturbed_text)
    elif args.text_perturbation == 'inserting':
        for _ in range(args.sampling_time):
            perturbed_text = word_inserting(text)
            perturbed_text_list.append(perturbed_text)
    elif args.text_perturbation == 'replacing':
        for _ in range(args.sampling_time):
            perturbed_text = word_replacing(text)
            perturbed_text_list.append(perturbed_text)
    elif args.text_perturbation == 'word_shuffle':
        for _ in range(args.sampling_time):
            perturbed_text = word_shuffle(text)
            perturbed_text_list.append(perturbed_text)
    elif args.text_perturbation == 'word_dropout':
        for dropout_rate in [0.05, 0.1, 0.15, 0.2, 0.25]:
            perturbed_text = word_dropout(text, dropout_rate)
            perturbed_text_list.append(perturbed_text)
    elif args.text_perturbation == 'character_dropout':
        for dropout_rate in [0.05, 0.1, 0.15, 0.2, 0.25]:
            perturbed_text = character_dropout(text, dropout_rate)
            perturbed_text_list.append(perturbed_text)
    elif args.text_perturbation == 'noise_injection':
        for noise_level in [0.05, 0.1, 0.15, 0.2, 0.25]:
            perturbed_text = noise_injection(text, noise_level)
            perturbed_text_list.append(perturbed_text)
    return perturbed_text_list


if __name__ == "__main__":
    # text = 'A child holding a flowered umbrella and petting a yak.'
    text = 'how do the two man play the instrument\n(0): roll the handle\n(1): tap their feet\n(2): strum the string\n(3): hit with sticks\n(4): pat with hand\nThis is single-choice question, answer with one choice number in 0, 1, 2, 3, 4.'
    llm = Qwen('Qwen2.5-7B-Instruct')

    for i in [1.0, 1.5, 2.0, 2.5, 3.0]:
        text_perturbed = llm_rephrasing(text, llm, i)
        print(f'llm_rephrasing {i}')
        print(text_perturbed)

    print('-' * 100)

    for i in range(5):
        text_perturbed = word_swapping(text)
        print(f'word_swapping')
        print(text_perturbed)

    print('-' * 100)

    for i in range(5):
        text_perturbed = word_deleting(text)
        print(f'word_deleting')
        print(text_perturbed)

    print('-' * 100)

    for i in range(5):
        text_perturbed = word_inserting(text)
        print(f'word_inserting')
        print(text_perturbed)

    print('-' * 100)

    for i in range(5):
        text_perturbed = word_replacing(text)
        print(f'word_replacing')
        print(text_perturbed)

    print('-' * 100)

    for i in range(5):
        text_perturbed = word_shuffle(text)
        print(f'word_shuffle')
        print(text_perturbed)

    print('-' * 100)

    for i in [0.05, 0.10, 0.15, 0.20, 0.25]:
        text_perturbed = word_dropout(text, i)
        print(f'word_dropout {i}')
        print(text_perturbed)

    print('-' * 100)

    for i in [0.05, 0.10, 0.15, 0.20, 0.25]:
        text_perturbed = character_dropout(text, i)
        print(f'character_dropout {i}')
        print(text_perturbed)

    print('-' * 100)

    for i in [0.05, 0.10, 0.15, 0.20, 0.25]:
        text_perturbed = noise_injection(text, i)
        print(f'noise_injection {i}')
        print(text_perturbed)