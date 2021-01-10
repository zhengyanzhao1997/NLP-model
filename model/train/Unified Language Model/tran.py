import pandas as pd
import json
import re
import os
import numpy as np
import tensorflow as tf
from sklearn import model_selection
from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K
from tensorflow.keras import initializers, activations
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import StratifiedKFold
import random
from transformers import BertConfig,PretrainedConfig
from transformers_token import BertTokenizer
import glob
from transformers_until import *
import unicodedata, re

def unilm_mask_single(s):
    idxs = K.cumsum(s, axis=0)
    mask = idxs[None, :] <= idxs[:, None]
    mask = K.cast(mask, K.floatx())
    return mask

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

class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """
    def compute_loss(self, inputs, mask=None):
        y_true, y_mask, y_pred = inputs
        y_true = tf.cast(y_true,tf.float32)
        y_mask = tf.cast(y_mask,tf.float32)
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss

class AutoRegressiveDecoder(object):
    """通用自回归生成模型解码基类
    包含beam search和random sample两种策略
    """
    def __init__(self, start_id, end_id, maxlen, model,minlen=1):
        self.start_id = start_id
        self.end_id = end_id
        self.maxlen = maxlen
        self.minlen = minlen
        self.models = {}
        self.model = model
        if start_id is None:
            self.first_output_ids = np.empty((1, 0), dtype=int)
            # array([], shape=(1, 0), dtype=int64)
        else:
            self.first_output_ids = np.array([[self.start_id]])

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
                        softmax(prediction[0] / temperature), prediction[1]
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
        #inputs = [token_ids,segment_ids]
        output_ids, output_scores = self.first_output_ids, np.zeros(1)
        # output_ids = [] , output_scores = 0
        for step in range(self.maxlen):
            scores, states = self.predict(
                inputs, output_ids, states, temperature, 'logits'
            )  # 计算当前得分
            #每一次人输入的拼接在predict里完成

            if step == 0:  # 第1步预测后将输入重复topk次
                inputs = [np.repeat(i, topk, axis=0) for i in inputs]
            scores = output_scores.reshape((-1, 1)) + scores  # 综合累积得分-相加等于相乘，输出的是logist
            # output_scores = [1.16165232,1.75142511]#上一次最优的两个的值
            # 分别由上面两个最优值作为x产生，故在各自产生的概率上加上之前的值
            # [[0.99853728 0.67273463 1.50580529 1.16165232 1.4321206 ]
            # [1.44454842 1.68150066 1.24661511 1.42612343 1.75142511]]
            indices = scores.argpartition(-topk, axis=None)[-topk:]  # 仅保留topk
            #[3 ,9]
            indices_1 = indices // scores.shape[1] # 候选字数 # 行索引
            # [0 ,1]
            indices_2 = (indices % scores.shape[1]).reshape((-1, 1))  # 列索引
            # [[3],[4]]
            output_ids = np.concatenate([output_ids[indices_1],indices_2],1)  # 更新输出
            #[[1,2,2,3,3], + [[3]
            # [2,3,1,4,4]]    [4]]
            output_scores = np.take_along_axis(
                scores, indices, axis=None
            )  # 更新得分
            #按indices的一维切片去获得索引 [1.16165232,1.75142511]
            end_counts = (output_ids == self.end_id).sum(1)  # 统计出现的end标记
            #[分别统计两条路 end出现次数 0,1]
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

class AutoTitle(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        ids,seg_id,mask_att = inputs
        ides_temp = ids.copy()
        seg_id_temp = seg_id.copy()
        mask_att_temp = mask_att.copy()
        len_out_put = len(output_ids[0])
        for i in range(len(ids)):
            get_len = len(np.where(ids[i] != 0)[0])
            end_ = get_len + len_out_put
            ides_temp[i][get_len:end_] = output_ids[i]
            seg_id_temp[i][get_len:end_] = np.ones_like(output_ids[i])
            mask_att_temp[i] = unilm_mask_single(seg_id_temp[i])
        return self.last_token(end_-1).predict([ides_temp,seg_id_temp,mask_att_temp])
    
    def generate(self,text,tokenizer,maxlen,topk=1):
        max_c_len = maxlen - self.maxlen
        input_dict = tokenizer(text,max_length=max_c_len,truncation=True,padding=True)
        token_ids = input_dict['input_ids']
        segment_ids = input_dict['token_type_ids']
        ids = np.zeros((1,maxlen),dtype='int32')
        seg_id = np.zeros((1,maxlen),dtype='int32')
        mask_att = np.zeros((1,maxlen,maxlen),dtype='int32')
        len_ = len(token_ids)
        ids[0][:len_] = token_ids
        seg_id[0][:len_] = segment_ids
        mask_id = unilm_mask_single(seg_id[0])
        mask_att[0] = mask_id
        output_ids = self.beam_search([ids,seg_id,mask_att],topk=topk)  # 基于beam search
        return tokenizer.decode(output_ids)
    

def load_vocab(dict_path, encoding='utf-8', simplified=False, startswith=None):
    """从bert的词典文件中读取词典
    """
    def _is_punctuation(ch):
        """标点符号类字符判断（全/半角均在此内）
        提醒：unicodedata.category这个函数在py2和py3下的
        表现可能不一样，比如u'§'字符，在py2下的结果为'So'，
        在py3下的结果是'Po'。
        """
        code = ord(ch)
        return 33 <= code <= 47 or \
            58 <= code <= 64 or \
            91 <= code <= 96 or \
            123 <= code <= 126 or \
            unicodedata.category(ch).startswith('P')
    
    def stem(token):
        """获取token的“词干”（如果是##开头，则自动去掉##）
        """
        if token[:2] == '##':
            return token[2:]
        else:
            return token
        
    def _cjk_punctuation():
        return u'\uff02\uff03\uff04\uff05\uff06\uff07\uff08\uff09\uff0a\uff0b\uff0c\uff0d\uff0f\uff1a\uff1b\uff1c\uff1d\uff1e\uff20\uff3b\uff3c\uff3d\uff3e\uff3f\uff40\uff5b\uff5c\uff5d\uff5e\uff5f\uff60\uff62\uff63\uff64\u3000\u3001\u3003\u3008\u3009\u300a\u300b\u300c\u300d\u300e\u300f\u3010\u3011\u3014\u3015\u3016\u3017\u3018\u3019\u301a\u301b\u301c\u301d\u301e\u301f\u3030\u303e\u303f\u2013\u2014\u2018\u2019\u201b\u201c\u201d\u201e\u201f\u2026\u2027\ufe4f\ufe51\ufe54\u00b7\uff01\uff1f\uff61\u3002'

    def _is_cjk_character(ch):
        """CJK类字符判断（包括中文字符也在此列）
        参考：https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        """
        code = ord(ch)
        return 0x4E00 <= code <= 0x9FFF or \
            0x3400 <= code <= 0x4DBF or \
            0x20000 <= code <= 0x2A6DF or \
            0x2A700 <= code <= 0x2B73F or \
            0x2B740 <= code <= 0x2B81F or \
            0x2B820 <= code <= 0x2CEAF or \
            0xF900 <= code <= 0xFAFF or \
            0x2F800 <= code <= 0x2FA1F
    
    token_dict = {}
    with open(dict_path, encoding=encoding) as reader:
        for line in reader:
            token = line.split()
            token = token[0] if token else line.strip()
            token_dict[token] = len(token_dict)

    if simplified:  # 过滤冗余部分token
        new_token_dict, keep_tokens = {}, []
        startswith = startswith or []
        for t in startswith:
            new_token_dict[t] = len(new_token_dict)
            keep_tokens.append(token_dict[t])

        for t, _ in sorted(token_dict.items(), key=lambda s: s[1]):
            if t not in new_token_dict:
                keep = True
                if len(t) > 1:
                    for c in stem(t):
                        if (
                            _is_cjk_character(c) or
                            _is_punctuation(c)
                        ):
                            keep = False
                            break
                if keep:
                    new_token_dict[t] = len(new_token_dict)
                    keep_tokens.append(token_dict[t])

        return new_token_dict, keep_tokens
    else:
        return token_dict

def build_model(pretrained_path,config,MAX_LEN,vocab_size,keep_tokens):
    ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    token_id = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    att = tf.keras.layers.Input((MAX_LEN,MAX_LEN), dtype=tf.int32)
    config.output_hidden_states = True
    bert_model = TFBertModel.from_pretrained(pretrained_path,config=config,from_pt=True)
    bert_model.bert.set_input_embeddings(tf.gather(bert_model.bert.embeddings.word_embeddings,keep_tokens))
    x, _ , hidden_states = bert_model(ids,token_type_ids=token_id,attention_mask=att)
    layer_1 = hidden_states[-1]
    word_embeeding = bert_model.bert.embeddings.word_embeddings
    embeddding_trans = tf.transpose(word_embeeding)
    
#     sof_output = tf.keras.layers.Dense(vocab_size,activation='softmax')(layer_1)

    sof_output = tf.matmul(layer_1,embeddding_trans)
    sof_output = tf.keras.layers.Activation('softmax')(sof_output)
    output = CrossEntropy(2)([ids,token_id,sof_output])
    model = tf.keras.models.Model(inputs=[ids,token_id,att],outputs=output)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(optimizer=optimizer)
    model.summary()
    return model

class DataGenerator(object):
    """数据生成器模版
    """
    def __init__(self, data, batch_size=32, Max_len = 256 ,tokenizer=None,buffer_size=None):
        self.data = data
        self.tokenizer = tokenizer
        self.Max_len = Max_len
        self.batch_size = batch_size
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
                
class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        ids = np.zeros((self.batch_size,self.Max_len),dtype='int32')
        seg_id = np.zeros((self.batch_size,self.Max_len),dtype='int32')
        mask_att = np.zeros((self.batch_size,self.Max_len,self.Max_len),dtype='int32')
        index = 0
        for is_end, txt in self.sample(random):
            text = open(txt, encoding='utf-8').read()
            text = text.split('\n')
            if len(text) > 1:
                title = text[0]
                content = '\n'.join(text[1:])
                input_dict = self.tokenizer(content,title,max_length=self.Max_len,truncation=True,padding=True)
                len_ = len(input_dict['input_ids'])
                token_ids = input_dict['input_ids']
                segment_ids = input_dict['token_type_ids']
                ids[index][:len_] = token_ids
                seg_id[index][:len_] = segment_ids
                mask_id = unilm_mask_single(seg_id[index])
                mask_att[index] = mask_id
                index += 1
            if  index == self.batch_size or is_end:
                yield [ids[:index],seg_id[:index],mask_att[:index]]
                ids = np.zeros((self.batch_size,self.Max_len),dtype='int32')
                seg_id = np.zeros((self.batch_size,self.Max_len),dtype='int32')
                mask_att = np.zeros((self.batch_size,self.Max_len,self.Max_len),dtype='int32')
                index = 0

def just_show(autotitle,tokenizer,maxlen):
    s1 = u'夏天来临，皮肤在强烈紫外线的照射下，晒伤不可避免，因此，晒后及时修复显得尤为重要，否则可能会造成长期伤害。专家表示，选择晒后护肤品要慎重，芦荟凝胶是最安全，有效的一种选择，晒伤严重者，还请及时就医。'
    s2 = u'8月28日，网络爆料称，华住集团旗下连锁酒店用户数据疑似发生泄露。从卖家发布的内容看，数据包含华住旗下汉庭、禧玥、桔子、宜必思等10余个品牌酒店的住客信息。泄露的信息包括华住官网注册资料、酒店入住登记的身份信息及酒店开房记录，住客姓名、手机号、邮箱、身份证号、登录账号密码等。卖家对这个约5亿条数据打包出售。第三方安全平台威胁猎人对信息出售者提供的三万条数据进行验证，认为数据真实性非常高。当天下午，华住集团发声明称，已在内部迅速开展核查，并第一时间报警。当晚，上海警方消息称，接到华住集团报案，警方已经介入调查。'
    for s in [s1, s2]:
        print(u'生成标题:', autotitle.generate(s,tokenizer,maxlen,topk=3))
    print()
    
class Evaluator(tf.keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self,tokenizer,maxlen,autotitle):
        self.lowest = 1e10
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.autotitle = autotitle

    def on_epoch_end(self, epoch, logs=None):
        # 保存最优
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            self.model.save_weights('./model/best_model.hdf5')
        # 演示效果
        just_show(self.autotitle,self.tokenizer,self.maxlen)

def main():
    pretrained_path = './torch_unilm_model'
    vocab_path = os.path.join(pretrained_path,'vocab.txt')
    new_token_dict, keep_tokens = load_vocab(vocab_path,simplified=True,startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'])
    tokenizer = BertTokenizer(new_token_dict)
    vocab_size = tokenizer.vocab_size
    print(vocab_size)
    
    config_path = os.path.join(pretrained_path,'config.json')
    config = BertConfig.from_json_file(config_path)
    MAX_LEN = 256
    txts = glob.glob('./THUCNews/*/*.txt')
    batch_size = 8
    train_generator = data_generator(txts,batch_size,MAX_LEN,tokenizer)
    model = build_model(pretrained_path,config,MAX_LEN,vocab_size,keep_tokens)
    steps_per_epoch = 1000
    epochs = 10000
    autotitle = AutoTitle(start_id=None, end_id=new_token_dict['[SEP]'], maxlen=32,model=model)
    evaluator = Evaluator(tokenizer,MAX_LEN,autotitle)
    model.fit_generator(train_generator.forfit(),steps_per_epoch=steps_per_epoch,epochs=epochs,callbacks=[evaluator])
    
if __name__ == '__main__':
    main()
