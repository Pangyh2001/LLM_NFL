# 设置随机种子
import random

from tensorboardX import SummaryWriter

random.seed(42)

import re
from datetime import datetime

from datasets import load_dataset

from func import *
from args import *
import openai
# 忽略警告
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

"""
只让黑盒LLM预测受保护的提示，拿到响应之后计算余弦相似度，在计算随机提示和原始提示的余弦相似度。后者减去前者，后者的意思是随机猜测的恢复程度，前者是黑盒LLM迭代攻击的恢复程度。
"""
# TODO 还需要改一下，原文的隐私泄露随着epsilon的增加而增加吗，再分析一下
#  语义相似度如何计算，当前差的很小
from openai import OpenAI

def text_generaton_with_black_box_LLMs(prompt, tem,openai1):


    response = openai1.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=tem
    )

    return response.choices[0].message.content


def compute_semantic_similarity(original_prompt, recovered_prompt, embeddings):
    """
    计算原始提示和恢复的提示之间的语义相似度

    Args:
        original_prompt: 原始提示字符串
        recovered_prompt: 恢复的提示字符串
        embeddings: token级别的嵌入向量字典
    """
    # 将字符串分割成tokens
    orig_tokens = original_prompt.split()
    recv_tokens = recovered_prompt.split()

    # 获取每个token的嵌入向量并计算平均值
    orig_embs = np.mean([embeddings[token] for token in orig_tokens if token in embeddings], axis=0)
    recv_embs = np.mean([embeddings[token] for token in recv_tokens if token in embeddings], axis=0)

    # 重塑向量维度
    orig_embs = orig_embs.reshape(1, -1)
    recv_embs = recv_embs.reshape(1, -1)

    # 计算余弦相似度
    similarity = cosine_similarity(orig_embs, recv_embs)[0][0]

    return similarity


# 直接定义停用词列表，而不是从nltk导入
STOP_WORDS = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
              'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
              'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
              'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were',
              'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
              'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
              'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
              'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once'}


def evaluate_similarity(sentence1, sentence2, token_to_vector_dict):
    def get_sentence_embedding(sentence):
        # 文本预处理
        if isinstance(sentence, list):  # 如果输入是token列表
            tokens = [token.lower() for token in sentence]
        else:  # 如果输入是字符串
            sentence = sentence.lower()
            sentence = re.sub(r'[^\w\s]', '', sentence)
            tokens = sentence.split()

        vectors = []
        weights = []

        for token in tokens:
            if token not in STOP_WORDS and token in token_to_vector_dict:
                vectors.append(token_to_vector_dict[token])
                weights.append(1.0)  # 使用统一权重

        if not vectors:
            return np.zeros(len(next(iter(token_to_vector_dict.values()))))

        # 使用加权平均
        vectors = np.array(vectors)
        weights = np.array(weights)
        weighted_vectors = vectors * weights[:, np.newaxis]
        avg_vector = np.sum(weighted_vectors, axis=0) / np.sum(weights)

        # L2归一化
        norm = np.linalg.norm(avg_vector)
        return avg_vector / norm if norm != 0 else avg_vector

    emb1 = get_sentence_embedding(sentence1)
    emb2 = get_sentence_embedding(sentence2)

    # 使用tanh进行非线性变换
    similarity = np.tanh(cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0])

    return similarity

# 根据嵌入向量随机生成提示
def generate_random_tokens_from_dict(token_to_vector_dict, length=50):
    # 从词典中随机选择50个token
    available_tokens = list(token_to_vector_dict.keys())

    # 如果词典中的token数量少于要求生成的数量，允许重复选择
    selected_tokens = random.choices(available_tokens, k=length)

    # 将tokens用空格连接成字符串
    return ' '.join(selected_tokens)


# 计算黑盒LLM的恢复程度
def compute_R2(raw_document):
    # 计算原始提示的恢复程度。
    raw_tokens = get_first_50_tokens(raw_document)
    # print(F"原始文本：\n{raw_tokens}")
    # print("扰动输入:")
    perturbed_tokens = perturb_sentence(raw_tokens, args.eps, args.model, token_to_vector_dict, sorted_cl100_emb,
                                        sen_emb)
    # TODO 看看不加扰动的输出和提取出的输出的差距。
    # print(perturbed_tokens)

    # prompt="""Your task is to extend Prefix Text.
    # - Prefix Text:"""+perturbed_tokens+"""
    # \n\n Provide only your Continuation.
    # - Continuation:"""
    prompt = """You are a text recovery expert. Your task is to recover the original tokens that were replaced by their semantically similar alternatives.
            there are some examples:
                Original: The company announced record profits in the third quarter.
                Perturbed: Business speak happy money three time part.
                Recovered: The business reported positive financial results in the third period.

                Original: Scientists discover new species in Amazon rainforest.
                Perturbed: People find thing green tree place.
                Recovered: Researchers identified a new organism in the tropical forest.
        Perturbed: """ + perturbed_tokens + """\n
       You should only respond the recovered tokens.
       Output format:
       [token1] [token2] [token3] ...
       """

    # print("获取扰动提示的响应（恢复的内容）:")

    openai1 = OpenAI(
        api_key="sk-rlPPceYVhfHRinBCHvjVslt7DzuXPET6twEwJJeDmjQprSnS",
        base_url="https://xiaoai.plus/v1"
    )
    # TODO 这里改成了0.5
    response = text_generaton_with_black_box_LLMs(prompt, 0,openai1)
    response_text = get_first_50_tokens(response)

    # print(response_text)

    P_per = evaluate_similarity(raw_tokens, response_text, token_to_vector_dict)
    # print(f"黑盒LLM的恢复能力R2: \n{P_per}")
    # 得到扰动提示的恢复程度
    return P_per


# 计算随机提示和原始提示的语义相似度，模拟黑盒LLM随机猜测。
def compute_R1(random_text, raw_document):
    raw_tokens = get_first_50_tokens(raw_document)
    random_tokens = get_first_50_tokens(random_text)
    P_per = evaluate_similarity(raw_tokens, random_tokens, token_to_vector_dict)
    # print(f"随机猜测的恢复能力R1：\n{P_per}")
    return P_per




parser = get_parser()
args = parser.parse_args()

writer = SummaryWriter('./tensorboard_privacy')

# ================Load token embeddings================
with open("./data/cl100_embeddings.json", 'r') as f:
    cl100_emb = json.load(f)
    # vector_data_json = {k: cl100_emb[k] for k in list(cl100_emb.keys())[:11000]}
    vector_data_json = {k: cl100_emb[k] for k in list(cl100_emb.keys())}
    cl100_emb = None
    token_to_vector_dict = {token: np.array(vector) for token, vector in vector_data_json.items()}
if not os.path.exists(f'./data/sorted_cl100_embeddings.json'):
    init_func(1.0, token_to_vector_dict)
with open('./data/sorted_cl100_embeddings.json', 'r') as f1:
    sorted_cl100_emb = json.load(f1)
with open('./data/sensitivity_of_embeddings.json', 'r') as f:
    sen_emb = np.array(json.load(f))

with open('epsilon_p_result.txt', 'a') as f:
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    f.write("\n")
    f.write(current_time + "\n")
    f.write("Epsilon\tPrivacy Loss\n")  # 写入表头
    f.write("-" * 50 + "\n")  # 添加分隔线
    ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train[:100]")
    raw_dataset = ds['highlights']
    eps = [1, 2, 4, 8, 16, 32]
    with tqdm.tqdm(eps) as pbar:
        for epsilon in pbar:
            epsilon_p = []
            R1_p=[]
            R2_p=[]
            # 内层循环也使用tqdm来显示每个数据集的进度
            for raw_document in tqdm.tqdm(raw_dataset, leave=False):
                # 这个操作是必要的，因为扰动提示的时候会用到args.eps
                args.eps = epsilon
                R2 = compute_R2(raw_document)
                random_text = generate_random_tokens_from_dict(token_to_vector_dict)
                R1 = compute_R1(random_text, raw_document)
                privacy_leak = R1 - R2

                # 更新进度条后缀信息
                pbar.set_postfix({
                    'R1': f'{R1:.4f}',
                    'R2': f'{R2:.4f}',
                    'Privacy_Leak': f'{privacy_leak:.4f}'
                })


                epsilon_p.append(privacy_leak)
                R1_p.append(R1)
                R2_p.append(R2)

            mean_epsilon_p = float(np.mean(epsilon_p))
            mean_R1_p = float(np.mean(R1_p))
            mean_R2_p = float(np.mean(R2_p))
            # 使用add_scalars同时显示多个指标
            writer.add_scalars('Recovery_Metrics', {
                'Random_Prompt(R1)': mean_R1_p,
                'Perturbed_Prompt(R2)': mean_R2_p,  # 使用mean_R2_p而不是R2
                'Privacy_Leakage': mean_epsilon_p
            }, epsilon)

            print(f"{args.eps}:{mean_epsilon_p}\n")
            f.write(f"{args.eps}\t{mean_epsilon_p:.4f}\n")
            args.eps -= 1

    f.close()

# python main.py --eps 6.0 --model gpt-4

