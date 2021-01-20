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
from tool import *

def load_data(filename):
    D = []
    with open(filename) as f:
        for l in f:
            D.append(json.loads(l))
    return D

class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """    
    def compute_seq2seq_loss(self,inputs,k_sparse,mask=None):
        y_true, y_mask, y_pred ,_,_ = inputs
        y_mask = tf.cast(y_mask,y_pred.dtype)
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        pos_loss = tf.gather(y_pred,y_true[..., None],batch_dims=len(tf.shape(y_true[..., None]))-1)[...,0]
        y_pred = tf.nn.top_k(y_pred, k=k_sparse)[0]
        neg_loss = tf.math.reduce_logsumexp(y_pred, axis=-1)
        loss = neg_loss - pos_loss
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss
    
    def compute_copy_loss(self, inputs, mask=None):
        _,y_mask,_,y_true,y_pred = inputs
        y_mask = tf.cast(y_mask,y_pred.dtype)
        y_true = tf.cast(y_true,y_pred.dtype)
        y_mask = K.cumsum(y_mask[:, ::-1], axis=1)[:, ::-1]
        y_mask = K.cast(K.greater(y_mask, 0.5), K.floatx())
        y_mask = y_mask[:, 1:]  # mask标记，减少一位
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        y_true = y_true[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss
    
    def compute_loss(self, inputs, mask=None):
        sparse_seq2seq_loss = self.compute_seq2seq_loss(inputs,k_sparse=10)
        copy_loss = self.compute_copy_loss(inputs)
        return sparse_seq2seq_loss + copy_loss
    
class AutoTitle(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    def get_ngram_set(self, x, n):
        """生成ngram合集，返回结果格式是:
        {(n-1)-gram: set([n-gram的第n个字集合])}
        """
        result = {}
        for i in range(len(x) - n + 1):
            k = tuple(x[i:i + n])
            if k[:-1] not in result:
                result[k[:-1]] = set()
            result[k[:-1]].add(k[-1])
        return result
    
    @AutoRegressiveDecoder.wraps(default_rtype='logits', use_states=True)
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
        prediction = self.last_token(end_-1).predict([ides_temp,seg_id_temp,mask_att_temp])
        '''
        假设现在的topK = 2 所以每次只predict 二组的可能输出 len(ides_temp) = 2
        那我们初始化[0,0] 代表每一组输出组目前的ngram情况
        1. 当目前组输出的label为0时：没有输出限制，则从所有字典中选择输出，states = label = 0
        2. 当目前组输出的label为1时：输出限制为B，则从所有输入中选择输出，states = label = 1
        3. 当目前组输出的label为2时：输出限制为I,若目前 states=0，则说明之前未输出B，则I无效,将lable=2 mask掉
        若目前 states + 1 = n >= 2，则有效，且目前处于n-gram状态，要输出的值与输入中n个连续的字组成ngram + 1,
        则考虑目前已经输出的 n-1 个字符是否属于输入中的连续片断，若是则将该片断对应的后续子集作为候选集
        若否，则退回至 1 - gram
        注意：states在每次predict后都会被保存
        '''
        if states is None:
            states = [0]
        elif len(states) == 1 and len(ides_temp) > 1:
            states = states * len(ides_temp)
        
        # 根据copy标签来调整概率分布
        probas = np.zeros_like(prediction[0]) - 1000  # 最终要返回的概率分布 初始化负数
        for i, token_ids in enumerate(inputs[0]):
            if states[i] == 0:
                prediction[1][i, 2] *= -1  # 0不能接2 mask掉 2这个值
            label = prediction[1][i].argmax()  # 当前label
            if label < 2:
                states[i] = label #[1,0]
            else:
                states[i] += 1 #如果当前
                
            if states[i] > 0:
                ngrams = self.get_ngram_set(ides_temp, states[i])
                '''
                if satates = 1 :开头
                因此 ngrams = 1 所有的token
                prefix = 全场 跳到 1garm 
                if satates > 1 说明这个地方的label = 2 前需要和前面几个2与1组成n garm
                则 ngrams = n 所有的token组合
                prefix = output_ids 的最后 n-1 个 token
                若存在 在 就是指定集合下的候选集
                '''
                prefix = tuple(output_ids[i, 1 - states[i]:])
                if prefix in ngrams:  # 如果确实是适合的ngram
                    candidates = ngrams[prefix]
                else:  # 没有的话就退回1gram
                    ngrams = self.get_ngram_set(ides_temp, 1)
                    candidates = ngrams[tuple()]
                    states[i] = 1
                candidates = list(candidates)
                probas[i, candidates] = prediction[0][i, candidates]
            else:
                probas[i] = prediction[0][i]
            idxs = probas[i].argpartition(-10)
            probas[i, idxs[:-10]] = -1000
            #把probas最小的k_sparse的值mask掉？？？
        return probas, states
    
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
    
def build_model(pretrained_path,config,MAX_LEN,vocab_size,keep_tokens):
    ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    token_id = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    att = tf.keras.layers.Input((MAX_LEN,MAX_LEN), dtype=tf.int32)
    label = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    
    config.output_hidden_states = True
    bert_model = TFBertModel.from_pretrained(pretrained_path,config=config,from_pt=True)
    bert_model.bert.set_input_embeddings(tf.gather(bert_model.bert.embeddings.word_embeddings,keep_tokens))
    x, _ , hidden_states = bert_model(ids,token_type_ids=token_id,attention_mask=att)
    layer_1 = hidden_states[-1]
    label_out = tf.keras.layers.Dense(3,activation='softmax')(layer_1)
    word_embeeding = bert_model.bert.embeddings.word_embeddings
    embeddding_trans = tf.transpose(word_embeeding)
    sof_output = tf.matmul(layer_1,embeddding_trans)
    output = CrossEntropy([2,4])([ids,token_id,sof_output,label,label_out])
    model_pred = tf.keras.models.Model(inputs=[ids,token_id,att],outputs=[sof_output,label_out])
    model = tf.keras.models.Model(inputs=[ids,token_id,att,label],outputs=output)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(optimizer=optimizer)
    model.summary()
    return model , model_pred

def random_masking(token_ids):
    """对输入进行随机mask，增加泛化能力
    """
    rands = np.random.random(len(token_ids))
    return [
        t if r > 0.15 else np.random.choice(token_ids)
        for r, t in zip(rands, token_ids)
    ]

class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        ids = np.zeros((self.batch_size,self.Max_len),dtype='int32')
        seg_id = np.zeros((self.batch_size,self.Max_len),dtype='int32')
        mask_att = np.zeros((self.batch_size,self.Max_len,self.Max_len),dtype='int32')
        label = np.zeros((self.batch_size,self.Max_len),dtype='int32')
        index = 0
        for is_end, d in self.sample(random):
            i = np.random.choice(2) + 1 if random else 1
            source, target = d['source_%s' % i], d['target']
            input_dict = self.tokenizer(source,target,max_length=self.Max_len,truncation=True,padding=True)
            len_ = len(input_dict['input_ids'])
            token_ids = random_masking(input_dict['input_ids'])
            sep_index = token_ids.index(self.tokenizer.vocab['[SEP]']) + 1
            source_labels, target_labels = generate_copy_labels(token_ids[:sep_index],token_ids[sep_index:])
            labels = source_labels + target_labels[1:]
            segment_ids = input_dict['token_type_ids']
            ids[index][:len_] = token_ids
            seg_id[index][:len_] = segment_ids
            mask_id = unilm_mask_single(seg_id[index])
            mask_att[index] = mask_id
            label[index][:len(labels)] = labels
            index += 1
            if  index == self.batch_size or is_end:
                yield [ids[:index],seg_id[:index],mask_att[:index],label[:index]]
                ids = np.zeros((self.batch_size,self.Max_len),dtype='int32')
                seg_id = np.zeros((self.batch_size,self.Max_len),dtype='int32')
                mask_att = np.zeros((self.batch_size,self.Max_len,self.Max_len),dtype='int32')
                label = np.zeros((self.batch_size,self.Max_len),dtype='int32')
                index = 0
        
def evaluate(autotitle,data,tokenizer,maxlen):
    evaluater = 0
    for d in tqdm(data, desc=u'评估中'):
        pred_summary = autotitle.generate(d['source_1'],tokenizer,maxlen,topk=5)
        print(pred_summary)
        evaluater += compute_main_metric(pred_summary,d['target'],'token')
    return evaluater/len(data)       

class Evaluator(tf.keras.callbacks.Callback):
    """训练回调
    """
    def __init__(self,valid_data,autotitle,tokenizer,maxlen):
        self.best_metric = 0.0
        self.valid_data = valid_data
        self.autotitle = autotitle
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def on_epoch_end(self, epoch, logs=None):
        eva = evaluate(self.autotitle,self.valid_data,self.tokenizer,self.maxlen)
        if  eva >= self.best_metric:  # 保存最优
            self.best_metric = eva
            self.model.save_weights('weights/seq2seq_model.hdf5')
            print('eva raise to %s'%eva)
        else:
            print('eva is %s,not raise'%eva)
            
def generate_copy_labels(source, target):
    """构建copy机制对应的label
    longest_common_subsequence：最长子串的动态规划算法
    """
    mapping = longest_common_subsequence(source, target)[1]
    source_labels = [0] * len(source)
    target_labels = [0] * len(target)
    i0, j0 = -2, -2
    for i, j in mapping:
        if i == i0 + 1 and j == j0 + 1:
            source_labels[i] = 2
            target_labels[j] = 2
        else:
            source_labels[i] = 1
            target_labels[j] = 1
        i0, j0 = i, j
    return source_labels, target_labels

def main():
    pretrained_path = '/root/zhengyanzhao/comment/emotion_extract/summariztion/torch_unilm_model'
    vocab_path = os.path.join(pretrained_path,'vocab.txt')
    new_token_dict, keep_tokens = load_vocab(vocab_path,simplified=True,startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'])
    tokenizer = BertTokenizer(new_token_dict)
    vocab_size = tokenizer.vocab_size
    config_path = os.path.join(pretrained_path,'config.json')
    config = BertConfig.from_json_file(config_path)
    config.model_type = 'NEZHA'
    MAX_LEN = 1024
    batch_size = 8
    data = load_data('sfzy_seq2seq.json')
    fold = 0
    num_folds = 15
    train_data = data_split(data, fold, num_folds, 'train')
    valid_data = data_split(data, fold, num_folds, 'valid')
    train_generator = data_generator(train_data,batch_size,MAX_LEN,tokenizer)
    model,model_pred = build_model(pretrained_path,config,MAX_LEN,vocab_size,keep_tokens)
    autotitle = AutoTitle(start_id=None, end_id=new_token_dict['[SEP]'],maxlen=512,model=model_pred)
    evaluator = Evaluator(valid_data,autotitle,tokenizer,MAX_LEN)
    epochs = 50
    model.fit_generator(train_generator.forfit(),steps_per_epoch=len(train_generator),epochs=epochs,callbacks=[evaluator])

if __name__ == '__main__':
    main()    
