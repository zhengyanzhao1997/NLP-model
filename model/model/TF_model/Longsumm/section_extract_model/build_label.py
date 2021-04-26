import pandas as pd
import numpy as np
import json
from rouge import Rouge
import re
from tqdm import tqdm
import six
import logging
import os
from nltk.tokenize import sent_tokenize

def load_data(filename):
    """加载数据
    返回：[texts]
    """
    D = []
    with open(filename) as f:
        for l in f:
            texts = json.loads(l)
            D.append(texts)
    return D


def sentence_token_nltk(str):
    sent_tokenize_list = sent_tokenize(str)
    return sent_tokenize_list

metric_keys = ['main', 'rouge-1', 'rouge-2', 'rouge-l']

rouge = Rouge()

def compute_rouge(source, target):
    #source, target = ' '.join(source), ' '.join(target)
    try:
        scores = rouge.get_scores(hyps=source, refs=target)
        return {
            'rouge-1': scores[0]['rouge-1']['f'],
            'rouge-2': scores[0]['rouge-2']['f'],
            'rouge-l': scores[0]['rouge-l']['f'],
        }
    except ValueError:
        return {
            'rouge-1': 0.0,
            'rouge-2': 0.0,
            'rouge-l': 0.0,
        }

def compute_main_metric(source, target):
    metrics = compute_rouge(source, target)
    metrics['main'] = (
            metrics['rouge-1'] * 0.2 + metrics['rouge-2'] * 0.4 +
            metrics['rouge-l'] * 0.4
    )
    return metrics['main']

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z]+", "", text)
    return len(text)

def text_split(text, limited=False):
    text = text.replace('\n',' ')
    texts = sentence_token_nltk(text)
    texts = [x for x in texts if clean_text(x)>=10]
    if limited:
        texts = texts[-256:]
    return texts

def extract_matching(texts, summaries, start_i=0, start_j=0):
    if len(texts) == 0 or len(summaries) == 0:
        return []
    i = np.argmax([len(s) for s in summaries])
    j = np.argmax([compute_main_metric(t, summaries[i]) for t in texts])
    lm = extract_matching(texts[:j + 1], summaries[:i], start_i, start_j)
    rm = extract_matching(
        texts[j:], summaries[i + 1:], start_i + i + 1, start_j + j
    )
    return lm + [(start_i + i, start_j + j)] + rm

def extract_flow_data_exc(x):
    text,summary = x['original'],x['extract_summ']
    texts = []
    section_len = []
    for i in text:
        i['text'] = text_split(i['text'], False)
        texts.extend(i['text'])
        section_len.append(len(i['text']))
    section_lensum = np.cumsum(section_len)
    summaries = text_split(summary, False)
    try:               
        mapping = extract_matching(texts, summaries)
    except:
        print('error:iter_max')
        return '', [], '', 0
    labels = sorted(set([i[1] for i in mapping]))
    pred_summary = ' '.join([texts[i] for i in labels])
    metric = compute_main_metric(pred_summary, summary)
    section_label = []
    for i in labels:
        for m,j in enumerate(section_lensum):
            if i+1 <= j:
                if m == 0:
                    section_label.append((m,i))
                else: 
                    section_label.append((m,i-section_lensum[m-1]))
                break     
    return text,section_label,summary,metric

def extract_flow_data_abs(x):
    text,summary = x['article'],x['summary']
    texts = text_split(text, False)
    summaries = text_split(summary, False)
    mapping = extract_matching(texts, summaries)
    labels = sorted(set([i[1] for i in mapping]))
    pred_summary = ' '.join([texts[i] for i in labels])
    metric = compute_main_metric(pred_summary, summary)
    return texts, labels, summary, metric

def parallel_apply(
    func,
    iterable,
    workers,
    max_queue_size,
    callback=None,
    dummy=True,
    random_seeds=True
):
    if dummy:
        from multiprocessing.dummy import Pool, Queue
    else:
        from multiprocessing import Pool, Queue

    in_queue, out_queue, seed_queue = Queue(max_queue_size), Queue(), Queue()
    if random_seeds is True:
        random_seeds = [None] * workers
    elif random_seeds is None or random_seeds is False:
        random_seeds = []
    for seed in random_seeds:
        seed_queue.put(seed)

    def worker_step(in_queue, out_queue):
        """单步函数包装成循环执行
        """
        if not seed_queue.empty():
            np.random.seed(seed_queue.get())
        while True:
            i, d = in_queue.get()
            r = func(d)
            out_queue.put((i, r))

    # 启动多进程/线程
    pool = Pool(workers, worker_step, (in_queue, out_queue))

    if callback is None:
        results = []

    # 后处理函数
    def process_out_queue():
        out_count = 0
        for _ in range(out_queue.qsize()):
            i, d = out_queue.get()
            out_count += 1
            if callback is None:
                results.append((i, d))
            else:
                callback(d)
        return out_count

    # 存入数据，取出结果
    in_count, out_count = 0, 0
    for i, d in enumerate(iterable):
        in_count += 1
        while True:
            try:
                in_queue.put((i, d), block=False)
                break
            except six.moves.queue.Full:
                out_count += process_out_queue()
        if in_count % max_queue_size == 0:
            out_count += process_out_queue()

    while out_count != in_count:
        out_count += process_out_queue()

    pool.terminate()

    if callback is None:
        results = sorted(results, key=lambda r: r[0])
        return [r[1] for r in results]

def convert(data,extract_flow):
    """分句，并转换为抽取式摘要
    """
    D = parallel_apply(
        func=extract_flow,
        iterable=tqdm(data, desc=u'转换数据'),
        workers=16,
        max_queue_size=200
    )
    total_metric = sum([d[3] for d in D])
    result = [d for d in D if d[3] > 0.7]
    print(u'抽取结果的平均指标: %s' % (total_metric / len(result)))
    return result

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def save_json(data,path):
    with open(path, 'w', encoding='utf-8') as f:
        for d in data:
            f.write(json.dumps(d,ensure_ascii=False,cls=NpEncoder) + '\n')
        
def main():
    data_json = './before_bulid_label.json'
    data_random_order_json = data_json[:-5] + '_random_order.json'
    data_extract_json = data_json[:-5] + '_extract.json'
    
    data2 = load_data('./before_bulid_label.json')
    data = convert(data2, extract_flow_data_exc)

    if os.path.exists(data_random_order_json):
        idxs = json.load(open(data_random_order_json))
    else:
        idxs = list(range(len(data)))
        np.random.shuffle(idxs)
        json.dump(idxs, open(data_random_order_json, 'w'))

    data = [data[i] for i in idxs]
    save_json(data,data_extract_json)

    print(u'输入数据：%s' % data_json)
    print(u'数据顺序：%s' % data_random_order_json)
    print(u'输出路径：%s' % data_extract_json)
    
if __name__ == "__main__":
    main()
