
import re
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
from func import *
from openai import OpenAI
from args import *
import warnings
from datetime import datetime
from tensorboardX import SummaryWriter  # 添加 TensorBoard 相关库
from bert_score import score as bert_score  # 添加 BERTScore 计算库
from nltk.translate.bleu_score import sentence_bleu  # 添加 BLEU 计算库
from rouge import Rouge  # 添加 ROUGE 计算库

# 设置随机种子
random.seed(42)
warnings.filterwarnings("ignore", category=FutureWarning)
nltk.data.path.append('/home/chen/nltk_data')
stop_words = set(stopwords.words('english'))
from nltk.translate.bleu_score import SmoothingFunction
import warnings
warnings.filterwarnings('ignore')

def get_local_llm_response(article: str, blackbox_continuation: str, openai) -> str:
    system_prompt = f"""Analyze the coherence between the "Prefix Text" and "Perturbed Generation", then follow these rules:

        1. If the "Perturbed Generation" meets ALL these criteria:
           - Continues the story/topic from the "Prefix Text" naturally
           - Maintains consistent facts and details
           - Uses similar tone and style
           → Output the "Perturbed Generation" exactly as is

        2. Otherwise, if ANY criteria are not met:
           - Write a new continuation that:
           - Directly follows from the last sentence of "Prefix Text"
           - Matches the style and tone of "Perturbed Generation"
           - Maintains factual consistency with "Prefix Text"

        OUTPUT ONLY the continuation text, with no explanations or additional text.

        ——"Prefix Text": {article}
        ——"Perturbed Generation": {blackbox_continuation}
        ——"Your Continuation":"""

    try:
        completion = openai.chat.completions.create(
            model="vicuna-7b-v1.5",
            messages=[
                {"role": "system", "content": system_prompt},
            ],
            max_tokens=150,
            temperature=0.2
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating local LLM response: {str(e)}")
        return ""


def text_generation_with_black_box_LLMs(article, tem, openai):
    system_prompt = """You are a creative writer skilled in continuing stories and texts in a natural and engaging way.
        Your task is to generate a coherent continuation that matches the style and context of the original text."""

    user_prompt = f"""Continue this text in a natural and engaging way, maintaining the same style and tone. 
    Write 2-3 sentences that flow naturally from the original text.

    Original Text: {article}

    [your continuation]"""

    try:
        response = openai.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            max_tokens=150,
            temperature=tem,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating continuation: {str(e)}")
        return None

def get_text_slice(article: str) -> str:
    # 将文本分割成单词列表
    words = article.split()

    # 确保文本长度足够
    if len(words) < 150:  # 50 + 100
        return " ".join(words[50:])  # 如果文本不够长，返回50词之后的所有内容

    # 返回第50个词后的100个词
    return " ".join(words[50:150])

# 计算语义相似度。
def compute_similarity(text1, text2, token_to_vector_dict):
    def text_to_vector(text):
        words = re.findall(r'\w+', text.lower())
        vectors = [token_to_vector_dict[word] for word in words if word in token_to_vector_dict]

        if vectors:
            avg_vector = np.mean(vectors, axis=0)
            return avg_vector / np.linalg.norm(avg_vector)
        else:
            return np.zeros(len(next(iter(token_to_vector_dict.values()))))

    vector1 = text_to_vector(text1)
    vector2 = text_to_vector(text2)
    return cosine_similarity([vector1], [vector2])[0][0]


def extract_keywords(text):
    # 调试信息
    print("Input text:", text[:100] + "..." if len(text) > 100 else text)

    if not text or text.strip() == '':
        print("Warning: Empty input text")
        return []

    # 分词和预处理
    words = word_tokenize(text.lower())
    print(f"Total words after tokenization: {len(words)}")

    # 更宽松的词过滤条件
    words = [word for word in words if any(c.isalnum() for c in word)]
    print(f"Words after filtering: {len(words)}")
    print("Sample words:", words[:10])

    if not words:
        print("Warning: No valid words after preprocessing")
        return []

    try:
        # 使用更宽松的TF-IDF设置
        vectorizer = TfidfVectorizer(
            stop_words='english',
            min_df=1,  # 允许只出现一次的词
            max_df=1.0,  # 允许在所有文档中都出现的词
            token_pattern=r'(?u)\b\w+\b'  # 更宽松的词模式
        )

        text_for_tfidf = ' '.join(words)
        print(f"Text for TF-IDF: {text_for_tfidf[:100]}...")

        tfidf_matrix = vectorizer.fit_transform([text_for_tfidf])

        # 获取特征名称和对应的TF-IDF分数
        feature_names = vectorizer.get_feature_names_out()
        print(f"Number of features: {len(feature_names)}")

        if len(feature_names) == 0:
            print("Warning: No features extracted by TF-IDF")
            return []

        scores = tfidf_matrix.toarray()[0]

        # 将词和分数组合并排序
        word_scores = list(zip(feature_names, scores))
        word_scores.sort(key=lambda x: x[1], reverse=True)

        # 返回前5个关键词
        keywords = [word for word, score in word_scores[:5]]
        print("Extracted keywords:", keywords)
        return keywords

    except ValueError as e:
        print(f"ValueError in TF-IDF processing: {str(e)}")
        return []
    except Exception as e:
        print(f"Unexpected error in extract_keywords: {str(e)}")
        return []


def evaluate_summary(reference_summary, generated_summary, token_to_vector_dict):
    print("\nEvaluating summaries...")
    print("Reference summary length:", len(reference_summary))
    print("Generated summary length:", len(generated_summary))

    if not reference_summary or not generated_summary:
        print("Warning: Empty summary detected")
        return 0.0, 0.0

    try:
        # 提取关键词
        print("\nExtracting reference keywords...")
        reference_keywords = extract_keywords(reference_summary)
        print("\nExtracting generated summary keywords...")
        summary_keywords = extract_keywords(generated_summary)

        # 如果任一关键词列表为空，返回0分
        if not reference_keywords or not summary_keywords:
            print("Warning: Empty keywords list detected")
            return 0.0, 0.0

        # 计算关键词覆盖率
        common_keywords = set(reference_keywords) & set(summary_keywords)
        keyword_coverage = len(common_keywords) / len(reference_keywords)
        print(f"Keyword coverage: {keyword_coverage}")

        # 计算语义相似度
        ref_vectors = [token_to_vector_dict[word] for word in reference_keywords if word in token_to_vector_dict]
        sum_vectors = [token_to_vector_dict[word] for word in summary_keywords if word in token_to_vector_dict]

        if not ref_vectors or not sum_vectors:
            print("Warning: No word vectors found")
            return keyword_coverage, 0.0

        ref_centroid = np.mean(ref_vectors, axis=0)
        sum_centroid = np.mean(sum_vectors, axis=0)

        semantic_similarity = cosine_similarity([ref_centroid], [sum_centroid])[0][0]
        print(f"Semantic similarity: {semantic_similarity}")

        return keyword_coverage, semantic_similarity

    except Exception as e:
        print(f"Error in evaluate_summary: {str(e)}")
        return 0.0, 0.0


# 计算多样性
def compute_ngram_coverage(reference_summary, generated_summary, n=2):
    reference_ngrams = set(zip(*[reference_summary.split()[i:] for i in range(n)]))
    generated_ngrams = set(zip(*[generated_summary.split()[i:] for i in range(n)]))
    coverage = len(generated_ngrams.intersection(reference_ngrams)) / len(reference_ngrams) if reference_ngrams else 0
    return coverage

# 计算连贯性
def compute_coherence(reference_summary, generated_summary):
    # 计算 Jaccard 相似度
    reference_words = set(reference_summary.split())
    generated_words = set(generated_summary.split())
    intersection = len(reference_words.intersection(generated_words))
    union = len(reference_words.union(generated_words))
    return intersection / union if union > 0 else 0


def log_response_counts(epsilon, local_count, blackbox_count, filename="response_counts.txt"):
    """记录每个epsilon下的响应数量"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    with open(filename, 'a', encoding='utf-8') as f:
        log_line = f"[{timestamp}] Epsilon: {epsilon:.4f}, Local LLM responses: {local_count}, Blackbox LLM responses: {blackbox_count}\n"
        f.write(log_line)


def main():
    # 创建 TensorBoard 记录器
    tb_blackbox = SummaryWriter(log_dir='./tensorboard_30_XUXIE19/blackbox')
    tb_local = SummaryWriter(log_dir='./tensorboard_30_XUXIE19/local')
    tb_diff = SummaryWriter(log_dir='./tensorboard_30_XUXIE19/diff')

    # 创建openai实例，用于黑盒LLM和本地LLM
    client_black_box = OpenAI(
        # api_key="sk-rlPPceYVhfHRinBCHvjVslt7DzuXPET6twEwJJeDmjQprSnS",
        api_key="sk-QdzBVx8Xd0iiDd1uURPx9EJJ9RHaS8aFDeSSm8J6SLAnxXK2",
        base_url="https://xiaoai.plus/v1"
    )
    client_local = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8001/v1"
    )

    parser = get_parser()
    args = parser.parse_args()

    # Load token embeddings
    with open("./data/cl100_embeddings.json", 'r') as f:
        cl100_emb = json.load(f)
        token_to_vector_dict = {token: np.array(vector) for token, vector in cl100_emb.items()}

    if not os.path.exists(f'./data/sorted_cl100_embeddings.json'):
        init_func(1.0, token_to_vector_dict)

    with open('./data/sorted_cl100_embeddings.json', 'r') as f1:
        sorted_cl100_emb = json.load(f1)

    with open('./data/sensitivity_of_embeddings.json', 'r') as f:
        sen_emb = np.array(json.load(f))

    ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train[:20]")

    eps = [1,2,4,6,8,12,26,30, 32]
    for epsilon in tqdm.tqdm(eps):
        epsilon_p = []
        local_responses = []
        F1_all=[]
        F1b_all=[]
        rouge_all=[]
        rougeb_all=[]
        bleu_all=[]
        bleub_all=[]
        diversity_all=[]
        diversityb_all=[]
        coherence_all=[]
        coherenceb_all=[]
        kc_all=[]
        kcb_all=[]
        ss_all=[]
        ssb_all=[]


        blackbox_responses = []
        for data in tqdm.tqdm(ds, leave=False):
            article = data['article']
            reference_summary = get_text_slice(article)


            # 1. 获取黑盒LLM对原始文章的摘要
            res1 = text_generation_with_black_box_LLMs(article, 0.5, client_black_box)
            summary1 = get_first_100_tokens(res1)
            if summary1 is not None:
                blackbox_responses.append(summary1)

            # 2. 扰动原始文章
            perturbed_article = perturb_sentence(article, epsilon, args.model,
                                                 token_to_vector_dict,
                                                 sorted_cl100_emb,
                                                 sen_emb)

            # 3. 获取黑盒LLM对扰动文章的摘要
            res2_blackLLM = text_generation_with_black_box_LLMs(perturbed_article, 0.5, client_black_box)
            blackbox_summary = get_first_100_tokens(res2_blackLLM)

            # 4. 让本地LLM基于原始文章和黑盒摘要生成新的摘要



            local_summary = get_local_llm_response(article, blackbox_summary, client_local)
            local_summary = get_first_100_tokens(local_summary)
            if local_summary is not None:
                local_responses.append(local_summary)

            # 5. 计算评估指标
            # 计算 BERTScore  本地LLM 服务器LLM
            P, R, F1 = bert_score([local_summary], [reference_summary], lang='en', return_hash=False)
            # print(F1)
            F1_all.append(F1)
            P, R, F1_b = bert_score( [summary1],[reference_summary], lang='en', return_hash=False)
            F1b_all.append(F1_b)

            # 计算 ROUGE
            rouge = Rouge()
            rouge_scores = rouge.get_scores(local_summary, reference_summary)[0]
            # print(rouge_scores)
            rouge_all.append({
                'rouge-1': rouge_scores['rouge-1']['f'],
                'rouge-2': rouge_scores['rouge-2']['f'],
                'rouge-l': rouge_scores['rouge-l']['f']
            })

            rouge_scores_b = rouge.get_scores(summary1, reference_summary)[0]
            rougeb_all.append({
                'rouge-1': rouge_scores_b['rouge-1']['f'],
                'rouge-2': rouge_scores_b['rouge-2']['f'],
                'rouge-l': rouge_scores_b['rouge-l']['f']
            })


            # 计算 BLEU
            weights = (0.25, 0.25, 0.25, 0.25)  # 1-gram到4-gram的权重
            bleu_score = sentence_bleu([reference_summary.split()], local_summary.split(),weights=weights,smoothing_function=SmoothingFunction().method1)
            bleu_all.append(bleu_score)

            bleu_score_b = sentence_bleu([reference_summary.split()], summary1.split(),weights=weights,smoothing_function=SmoothingFunction().method1)
            bleub_all.append(bleu_score_b)



            # 计算多样性（n-gram 覆盖率）
            diversity_local = compute_ngram_coverage(reference_summary, local_summary, n=2)
            diversity_all.append(diversity_local)

            diversity_blackbox = compute_ngram_coverage(reference_summary, summary1, n=2)
            diversityb_all.append(diversity_blackbox)

            # 计算连贯性（Jaccard 相似度）
            coherence_local = compute_coherence(reference_summary, local_summary)
            coherence_all.append(coherence_local)

            coherence_blackbox = compute_coherence(reference_summary, summary1)
            coherenceb_all.append(coherence_blackbox)



            # 关键词覆盖率和语义相似度
            kc,ss=evaluate_summary(reference_summary, local_summary,token_to_vector_dict)
            kc_all.append(kc)
            ss_all.append(ss)


            kc_b, ss_b = evaluate_summary(reference_summary,summary1,token_to_vector_dict)
            kcb_all.append(kc_b)
            ssb_all.append(ss_b)

        metrics = {
             'F1_Score': {
            'local': np.mean(F1_all),
            'blackbox': np.mean(F1b_all)
            },
            'ROUGE-1': {
                'local': np.mean([x['rouge-1'] for x in rouge_all]),
                'blackbox': np.mean([x['rouge-1'] for x in rougeb_all])
            },
            'ROUGE-2': {
                'local': np.mean([x['rouge-2'] for x in rouge_all]),
                'blackbox': np.mean([x['rouge-2'] for x in rougeb_all])
            },
            'ROUGE-L': {
                'local': np.mean([x['rouge-l'] for x in rouge_all]),
                'blackbox': np.mean([x['rouge-l'] for x in rougeb_all])
            },
            'BLEU': {
                'local': np.mean(bleu_all),
                'blackbox': np.mean(bleub_all)
            },
            'Diversity': {
                'local': np.mean(diversity_all),
                'blackbox': np.mean(diversityb_all)
            },
            'Coherence': {
                'local': np.mean(coherence_all),
                'blackbox': np.mean(coherenceb_all)
            },
            'Keyword_Coverage': {
                'local': np.mean(kc_all),
                'blackbox': np.mean(kcb_all)
            },
            'Semantic_Similarity': {
                'local': np.mean(ss_all),
                'blackbox': np.mean(ssb_all)
            }
        }

        # 记录到TensorBoard
        for metric_name, values in metrics.items():
            # 本地模型指标
            tb_local.add_scalar(f'{metric_name}', values['local'], epsilon)
            # 黑盒模型指标
            tb_blackbox.add_scalar(f'{metric_name}', values['blackbox'], epsilon)
            # 差异值
            tb_diff.add_scalar(f'{metric_name}_diff',
                                values['blackbox'] - values['local'],
                               epsilon)



        tb_blackbox.close()
        tb_local.close()
        tb_diff.close()



            # 记录响应数量
        log_response_counts(
            epsilon=epsilon,
            local_count=len(local_responses),
            blackbox_count=len(blackbox_responses)
        )

    tb_blackbox.close()  # 关闭 TensorBoard 记录器
    tb_local.close()  # 关闭 TensorBoard 记录器


if __name__ == "__main__":
    main()
