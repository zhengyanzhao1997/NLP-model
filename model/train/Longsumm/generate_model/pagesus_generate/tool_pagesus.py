import pandas as pd
import numpy as np
import json
from rouge import Rouge
import re
from tqdm import tqdm
import six
import logging
import os
from collections import defaultdict
import tensorflow as tf

# 指标名
metric_keys = ['main', 'rouge-1', 'rouge-2', 'rouge-l']

# 计算rouge用
rouge = Rouge()

def compute_rouge(source, target):
    """计算rouge-1、rouge-2、rouge-l
    """
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
    """计算所有metrics
    """
    metrics = compute_rouge(source, target)
    metrics['main'] = (
        metrics['rouge-1'] * 0.2 + metrics['rouge-2'] * 0.4 +
        metrics['rouge-l'] * 0.4
    )
    return metrics


def parallel_apply(
    func,
    iterable,
    workers,
    max_queue_size,
    callback=None,
    dummy=False,
    random_seeds=True
):
    """多进程或多线程地将func应用到iterable的每个元素中。
    注意这个apply是异步且无序的，也就是说依次输入a,b,c，但是
    输出可能是func(c), func(a), func(b)。
    参数：
        callback: 处理单个输出的回调函数；
        dummy: False是多进程/线性，True则是多线程/线性；
        random_seeds: 每个进程的随机种子。
    """
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
        
        

class DataGenerator(object):
    """数据生成器模版
    """
    def __init__(self, data, batch_size=32, Max_len = 256, decode_max_len = 512 ,tokenizer=None,buffer_size=None):
        self.data = data
        self.tokenizer = tokenizer
        self.Max_len = Max_len
        self.batch_size = batch_size
        self.decode_max_len = decode_max_len
        if hasattr(self.data, '__len__'):
            self.steps = len(self.data) // self.batch_size
            if len(self.data) % self.batch_size != 0:
                self.steps += 1
        else:
            self.steps = None
        self.buffer_size = buffer_size or batch_size * 1000

    def __len__(self):
        return self.steps

    def sample(self, random=False):
        """采样函数，每个样本同时返回一个is_end标记
        """
        if random:
            if self.steps is None:

                def generator():
                    caches, isfull = [], False
                    for d in self.data:
                        caches.append(d)
                        if isfull:
                            i = np.random.randint(len(caches))
                            yield caches.pop(i)
                        elif len(caches) == self.buffer_size:
                            isfull = True
                    while caches:
                        i = np.random.randint(len(caches))
                        yield caches.pop(i)

            else:

                def generator():
                    indices = list(range(len(self.data)))
                    np.random.shuffle(indices)
                    for i in indices:
                        yield self.data[i]

            data = generator()
        else:
            data = iter(self.data)

        d_current = next(data)
        for d_next in data:
            yield False, d_current
            d_current = d_next

        yield True, d_current

    def __iter__(self, random=False):
        raise NotImplementedError

    def forfit(self, random=True):
        while True:
            for d in self.__iter__(random):
                yield d
                
def data_split(data, fold, num_folds, mode):
    """划分训练集和验证集
    """
    if mode == 'train':
        D = [d for i, d in enumerate(data) if i % num_folds != fold]
    else:
        D = [d for i, d in enumerate(data) if i % num_folds == fold]

    if isinstance(data, np.ndarray):
        return np.array(D)
    else:
        return D
    
def longest_common_subsequence(source, target):
    """最长公共子序列（source和target的最长非连续子序列）
    返回：子序列长度, 映射关系（映射对组成的list）
    注意：最长公共子序列可能不止一个，所返回的映射只代表其中一个。
    """
    c = defaultdict(int)
    for i, si in enumerate(source, 1):
        for j, tj in enumerate(target, 1):
            if si == tj:
                c[i, j] = c[i - 1, j - 1] + 1
            elif c[i, j - 1] > c[i - 1, j]:
                c[i, j] = c[i, j - 1]
            else:
                c[i, j] = c[i - 1, j]
    '''
    C:最长序列矩阵
    l:最长公共序列长度
    '''
    l, mapping = c[len(source), len(target)], []
    i, j = len(source) - 1, len(target) - 1
    while len(mapping) < l:
        if source[i] == target[j]:
            mapping.append((i, j))
            i, j = i - 1, j - 1
        elif c[i + 1, j] > c[i, j + 1]:
            j = j - 1
        else:
            i = i - 1
    return l, mapping[::-1]


class AutoRegressiveDecoder(object):
    """通用自回归生成模型解码基类
    包含beam search和random sample两种策略
    """
    def __init__(self, start_id, end_id, maxlen,max_decode_len, model,minlen=1):
        self.start_id = start_id
        self.end_id = end_id
        self.max_decode_len = max_decode_len
        self.maxlen = maxlen
        self.minlen = minlen
        self.models = {}
        self.model = model
        if start_id is None:
            self.first_output_ids = np.empty((1, 0), dtype=int)
            # array([], shape=(1, 0), dtype=int64)
        else:
            self.first_output_ids = np.array([[self.start_id]])
            
    def softmax(self, x, axis=-1):
        x = x - x.max(axis=axis, keepdims=True)
        x = np.exp(x)
        return x / x.sum(axis=axis, keepdims=True)
    
    @staticmethod
    def wraps(default_rtype='probas', use_states=False):
        """用来进一步完善predict函数
        目前包含：1. 设置rtype参数，并做相应处理；
                  2. 确定states的使用，并做相应处理；
                  3. 设置温度参数，并做相应处理。
        """
        def actual_decorator(predict):
            def new_predict(
                self,
                inputs,
                output_ids,
                states,
                temperature=1,
                rtype=default_rtype
            ):
                assert rtype in ['probas', 'logits']
                prediction = predict(self, inputs, output_ids, states)
                if not use_states:
                    prediction = (prediction, None)

                if default_rtype == 'logits':
                    prediction = (
                        self.softmax(prediction[0] / temperature), prediction[1]
                    )
                elif temperature != 1:
                    probas = np.power(prediction[0], 1.0 / temperature)
                    probas = probas / probas.sum(axis=-1, keepdims=True)
                    prediction = (probas, prediction[1])

                if rtype == 'probas':
                    return prediction
                else:
                    return np.log(prediction[0] + 1e-12), prediction[1]

            return new_predict

        return actual_decorator

    def last_token(self,end):
        """创建一个只返回最后一个token输出的新Model
        """
#         if model not in self.models:
        outputs = [
                tf.keras.layers.Lambda(lambda x: x[:,end])(output)
                for output in self.model.outputs]
        model_temp = tf.keras.models.Model(self.model.inputs, outputs)

        return model_temp

    def predict(self, inputs, output_ids, states=None):
        """用户需自定义递归预测函数
        说明：定义的时候，需要用wraps方法进行装饰，传入default_rtype和use_states，
             其中default_rtype为字符串logits或probas，probas时返回归一化的概率，
             rtype=logits时则返回softmax前的结果或者概率对数。
        返回：二元组 (得分或概率, states)
        """
        raise NotImplementedError

    def beam_search(self, inputs, topk, states=None, temperature=1, min_ends=1):
        """beam search解码
        说明：这里的topk即beam size；
        返回：最优解码序列。
        """
        output_ids, output_scores = self.first_output_ids, np.zeros(1)
        for step in range(self.maxlen):
            scores, states = self.predict(
                inputs, output_ids, states, temperature, 'logits'
            )  # 计算当前得分
            #每一次人输入的拼接在predict里完成
            if step == 0:  # 第1步预测后将输入重复topk次
                inputs = [np.repeat(i, topk, axis=0) for i in inputs]
            scores = output_scores.reshape((-1, 1)) + scores  # 综合累积得分-相加等于相乘，输出的是logist
            indices = scores.argpartition(-topk, axis=None)[-topk:]  # 仅保留topk
            indices_1 = indices // scores.shape[1] # 候选字数 # 行索引
            indices_2 = (indices % scores.shape[1]).reshape((-1, 1))  # 列索引
            output_ids = np.concatenate([output_ids[indices_1],indices_2],1)  # 更新输出
            output_scores = np.take_along_axis(
                scores, indices, axis=None
            )  # 更新得分
            end_counts = (output_ids == self.end_id).sum(1)  # 统计出现的end标记
            if output_ids.shape[1] >= self.minlen:  # 最短长度判断
                best_one = output_scores.argmax()  # 得分最大的那个
                if end_counts[best_one] == min_ends: # =1   # 如果已经终止
                    return output_ids[best_one]  # 直接输出
                else:  # 否则，只保留未完成部分
                    flag = (end_counts < min_ends)  # 标记未完成序列
                    if not flag.all():  # 如果有已完成的
                        inputs = [i[flag] for i in inputs]  # 扔掉已完成序列
                        output_ids = output_ids[flag]  # 扔掉已完成序列
                        output_scores = output_scores[flag]  # 扔掉已完成序列
                        end_counts = end_counts[flag]  # 扔掉已完成end计数
                        topk = flag.sum()  # topk相应变化
        # 达到长度直接输出
        return output_ids[output_scores.argmax()]
    
class Loss(tf.keras.layers.Layer):
    """特殊的层，用来定义复杂loss
    """
    def __init__(self, output_axis=None, **kwargs):
        super(Loss, self).__init__(**kwargs)
        self.output_axis = output_axis

    def call(self, inputs, mask=None):
        loss = self.compute_loss(inputs, mask)
        self.add_loss(loss)
        if self.output_axis is None:
            return inputs
        elif isinstance(self.output_axis, list):
            return [inputs[i] for i in self.output_axis]
        else:
            return inputs[self.output_axis]

    def compute_loss(self, inputs, mask=None):
        raise NotImplementedError
