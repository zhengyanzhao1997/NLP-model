import pandas as pd
import json
import re
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel,BertConfig,TFBertLMHeadModel
from sklearn import model_selection
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K
from tensorflow.keras import initializers, activations
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import StratifiedKFold
import random
from tqdm import tqdm
import unicodedata, re
from tool import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

def load_data(filename):
    D = []
    with open(filename) as f:
        for l in f:
            D.append(json.loads(l))
    return D

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

class CrossEntropy_k(Loss):
  
    def compute_loss(self,inputs,mask=None):
        y_true, y_mask, y_pred = inputs
        #y_true = tf.cast(y_true,y_pred.dtype)
        y_mask = tf.cast(y_mask,y_pred.dtype)
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        pos_loss = tf.gather(y_pred,y_true[..., None],batch_dims=len(tf.shape(y_true[..., None]))-1)[...,0]
        y_pred = tf.nn.top_k(y_pred, k = 20)[0]
        neg_loss = tf.math.reduce_logsumexp(y_pred, axis=-1)
        loss = neg_loss - pos_loss
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss

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
        return self.model.predict([ides_temp,seg_id_temp,mask_att_temp])[:,end_-1]
    
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

    def _is_punctuation(ch):
        code = ord(ch)
        return 33 <= code <= 47 or \
            58 <= code <= 64 or \
            91 <= code <= 96 or \
            123 <= code <= 126
    
    def stem(token):
        """获取token的“词干”（如果是##开头，则自动去掉##）
        """
        if token[:2] == '##':
            return token[2:]
        else:
            return token
        
    def contain_word_num(word):
        return re.match(r"[^a-zA-Z0-9]", word)
    
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
                if len(t) > 0:
                    c = stem(t)
                    if len(c) > 1:
                        if contain_word_num(c):
                            keep = False
                    if len(c) == 1:
                        if not _is_punctuation(c) and not re.match(r"[a-zA-Z0-9]", c):
                            keep = False
                if keep:
                    new_token_dict[t] = len(new_token_dict)
                    keep_tokens.append(token_dict[t])

        return new_token_dict, keep_tokens
    else:
        return token_dict

def build_model(pretrained_path,config,MAX_LEN,vocab_size):#keep_tokens):
    ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    token_id = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    att = tf.keras.layers.Input((MAX_LEN,MAX_LEN), dtype=tf.int32)
    config.output_hidden_states = True
    config.is_decoder = True
    config.hierarchical = True
    #bert_model = TFBertModel.from_pretrained(pretrained_path,config=config)
    #bert_model.bert.set_input_embeddings(tf.gather(bert_model.bert.embeddings.word_embeddings,keep_tokens))
    bert_model = TFBertLMHeadModel.from_pretrained(pretrained_path,config=config,from_pt=True)
    x = bert_model(ids,token_type_ids=token_id,attention_mask=att)
    out_put = x.logits
    #word_embeeding = bert_model.bert.embeddings.word_embeddings
    #embeddding_trans = tf.transpose(word_embeeding)
    #sof_output = tf.matmul(layer_1,embeddding_trans)
    out_put = tf.keras.layers.Activation('softmax')(out_put)
    output = CrossEntropy(2)([ids,token_id,out_put])
    model = tf.keras.models.Model(inputs=[ids,token_id,att],outputs=output)
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    model.compile(optimizer=optimizer)
    model.summary()
    return model

class data_generator(DataGenerator):
    def __iter__(self, random=True):
        ids = np.zeros((self.batch_size,self.Max_len),dtype='int32')
        seg_id = np.zeros((self.batch_size,self.Max_len),dtype='int32')
        mask_att = np.zeros((self.batch_size,self.Max_len,self.Max_len),dtype='int32')
        index = 0
        for is_end, d in self.sample(random):
            summary = d[1]
            content = d[0]
            input_dict = self.tokenizer(content,summary,max_length=self.Max_len,truncation=True,padding=True)
            len_ = len(input_dict['input_ids'])
            token_ids = input_dict['input_ids']
            segment_ids = input_dict['token_type_ids']
            ids[index][:len_] = token_ids
            seg_id[index][:len_] = segment_ids
            mask_id = unilm_mask_single(seg_id[index])
            mask_att[index] = mask_id
            index += 1
            if  index == self.batch_size or is_end:
                if not is_end:
                    yield [ids[:index],seg_id[:index],mask_att[:index]]
                ids = np.zeros((self.batch_size,self.Max_len),dtype='int32')
                seg_id = np.zeros((self.batch_size,self.Max_len),dtype='int32')
                mask_att = np.zeros((self.batch_size,self.Max_len,self.Max_len),dtype='int32')
                index = 0

def just_show(autotitle,tokenizer,maxlen,valid_data):
    for d in tqdm(valid_data,desc='predicting'):
        label = d[1]
        generate_s = autotitle.generate(d[0],tokenizer,maxlen,topk=3)
        score = compute_main_metric(generate_s,d[1])
        print(u'标签摘要:',label)
        print(u'生成摘要:',generate_s)
        print(u'评估分数:',score)
    
class Evaluator(tf.keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self,tokenizer,maxlen,autotitle,valid_data):
        self.lowest = 1e10
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.autotitle = autotitle
        self.valid_data = valid_data

    def on_epoch_end(self, epoch, logs=None):
        # 保存最优
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            self.model.save_weights('./pretain/best_model.hdf5')
        # 演示效果
        just_show(self.autotitle,self.tokenizer,self.maxlen,self.valid_data)

def main():
    pretrained_path = './nuilm_small/'
    vocab_path = os.path.join(pretrained_path,'vocab.txt')
    #new_token_dict, keep_tokens = load_vocab(vocab_path,simplified=True,startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'])
    #tokenizer = BertTokenizer(new_token_dict)
    tokenizer = BertTokenizer.from_pretrained(pretrained_path)
    vocab_size = tokenizer.vocab_size
    print(vocab_size)
    config_path = os.path.join(pretrained_path,'config.json')
    config = BertConfig.from_json_file(config_path)
    MAX_LEN = 3072
    batch_size = 8
    data  = load_data('../pre_train_summary/nuion_data_pre.json')
    print(len(data))
    print(data[0][0])
    print(data[0][1])
    valid_data = data[:1]
    train_data = data[1:]
    train_generator = data_generator(train_data,batch_size,MAX_LEN,0,tokenizer)

    K.clear_session()
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    with strategy.scope():
        model = build_model(pretrained_path,config,MAX_LEN,vocab_size)#,keep_tokens)
    
    epochs = 5
    autotitle = AutoTitle(start_id=None, end_id=tokenizer.vocab['[SEP]'],maxlen=600,model=model)
    evaluator = Evaluator(tokenizer,MAX_LEN,autotitle,valid_data)
    model.fit_generator(train_generator.forfit(),steps_per_epoch=len(train_generator)-1,epochs=epochs,callbacks=[evaluator])
    
if __name__ == '__main__':
    main()
