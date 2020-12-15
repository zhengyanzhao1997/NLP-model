import pandas as pd
import json
import re
import os
import numpy as np
import tensorflow as tf
from sklearn import model_selection
from transformers import *
from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K
from tensorflow.keras import initializers, activations
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import StratifiedKFold
from typing import Dict, List, Optional, Union
import random

def load_data(path):
    text_list = []
    spo_list = []
    with open(path) as json_file:
        for i in json_file:
            text_list.append(eval(i)['text'])
            spo_list.append(eval(i)['spo_list'])
    c = list(zip(text_list,spo_list))
    random.shuffle(c)
    text_list,spo_list = zip(*c)
    return text_list,spo_list

def load_valid_data(path):
    text_list = []
    spo_list = []
    with open(path) as json_file:
        for i in json_file:
            text_list.append(eval(i)['text'])
            spo_list.append(eval(i)['spo_list'])
    return text_list,spo_list

def load_ps(path):
    with open(path,'r')  as f:
        data = pd.DataFrame([eval(i) for i in f])['predicate']
    p2id = {}
    id2p = {}
    data = list(set(data))
    for i in range(len(data)):
        p2id[data[i]] = i
        id2p[i] = data[i]
    return p2id,id2p

def proceed_data(text_list,spo_list,p2id,t2id,tokenizer,MAX_LEN):
    ct = len(text_list)
    input_ids = np.zeros((ct,MAX_LEN),dtype='int32')
    attention_mask = np.zeros((ct,MAX_LEN),dtype='int32')
    start_tokens = np.zeros((ct,MAX_LEN,len(t2id)+1),dtype='int32')
    end_tokens = np.zeros((ct,MAX_LEN,len(t2id)+1),dtype='int32')
    send_s_po = np.zeros((ct,MAX_LEN),dtype='int32')
    c_relation = np.zeros((ct,MAX_LEN,MAX_LEN,len(p2id)),dtype='int32')
    valid_index = 0
    for k in range(ct):
        if len(spo_list[k])==0:
            continue
        context_k = text_list[k].lower().replace(' ','')
        enc_context = tokenizer.encode(context_k,max_length=MAX_LEN,truncation=True) 
        if_vaild = 0
        for j in range(len(spo_list[k])):
            s_type = spo_list[k][j]['subject_type']
            s_type_index = t2id[s_type]
            answers_text_k = spo_list[k][j]['subject'].lower().replace(' ','')
            chars = np.zeros((len(context_k)))
            index = context_k.find(answers_text_k)
            chars[index:index+len(answers_text_k)]=1
            offsets = []
            idx=0
            for t in enc_context[1:]:
                w = tokenizer.decode([t])
                if '#' in w and len(w)>1:
                    w = w.replace('#','')
                if w == '[UNK]':
                    w = '。'
                offsets.append((idx,idx+len(w)))
                idx += len(w)
            toks = []
            for i,(a,b) in enumerate(offsets):
                sm = np.sum(chars[a:b])
                if sm>0: 
                    toks.append(i) 
            if len(toks)>0:
                if_vaild = 1
                start_tokens[valid_index,toks[0]+1,s_type_index] = 1
                end_tokens[valid_index,toks[-1]+1,s_type_index] = 1
                S_end = toks[-1]+1
                send_s_po[valid_index,toks[-1]+1] = s_type_index
                
                o_type = spo_list[k][j]['object_type']
                o_type_index = t2id[o_type]
                object_text_k = spo_list[k][j]['object'].lower().replace(' ','')
                chars = np.zeros((len(context_k)))
                index = context_k.find(object_text_k)
                chars[index:index+len(object_text_k)]=1
                offsets = []
                idx=0
                for t in enc_context[1:]:
                    w = tokenizer.decode([t])
                    if '#' in w and len(w)>1:
                        w = w.replace('#','')
                    if w == '[UNK]':
                        w = '。'
                    offsets.append((idx,idx+len(w)))
                    idx += len(w)
                toks = []
                for i,(a,b) in enumerate(offsets):
                    sm = np.sum(chars[a:b])
                    if sm>0: 
                        toks.append(i) 
                if len(toks)>0:
                    start_tokens[valid_index,toks[0]+1,o_type_index] = 1
                    end_tokens[valid_index,toks[-1]+1,o_type_index] = 1
                    O_end = toks[-1]+1
                    predict = spo_list[k][j]['predicate']
                    P_index = p2id[predict]
                    c_relation[valid_index,S_end,O_end,P_index] = 1
                    send_s_po[valid_index,toks[-1]+1] = o_type_index 
        if if_vaild == 1:
            input_ids[valid_index,:len(enc_context)] = enc_context
            attention_mask[valid_index,:len(enc_context)] = 1
            valid_index += 1
            print(valid_index)
    return input_ids[:valid_index],attention_mask[:valid_index],start_tokens[:valid_index],end_tokens[:valid_index],send_s_po[:valid_index],c_relation[:valid_index]

def rematch_text_word(tokenizer,text,enc_context,enc_start,enc_end):
    span = [a.span()[0] for a in re.finditer(' ', text)]
    decode_list = [tokenizer.decode([i]) for i in enc_context][1:]
    start = 0
    end = 0
    len_start = 0
    for i in range(len(decode_list)):
        if i ==  enc_start - 1:
            start = len_start
        j = decode_list[i]
        if '#' in j and len(j)>1:
            j = j.replace('#','')
        if j == '[UNK]':
            j = '。'
        len_start += len(j)
        if i == enc_end - 1:
            end = len_start
            break
    for span_index in span:
        if start >= span_index:
            start += 1
            end += 1
        if end > span_index and span_index>start:
            end += 1
    return text[start:end]

# class Biaffine(tf.keras.layers.Layer):
#     def __init__(self, in_size, out_size, bias_x=False, bias_y=False):
#         super(Biaffine, self).__init__()
#         self.bias_x = bias_x
#         self.bias_y = bias_y
#         self.w1 = self.add_weight(
#             name='weight1', 
#             shape=(in_size + int(bias_x), out_size, in_size + int(bias_y)),
#             trainable=True)
#         self.w2 = self.add_weight(
#             name='weight2', 
#             shape=(2*in_size + 2*int(bias_x), out_size),
#             trainable=True)
        
#     def call(self, input1, input2):
#         if self.bias_x:
#             input1 = tf.concat((input1, tf.ones_like(input1[..., :1])), axis=-1)
#         if self.bias_y:
#             input2 = tf.concat((input2, tf.ones_like(input2[..., :1])), axis=-1)
#         # bxi,oij,byj->boxy
#         logits_1 = tf.einsum('bxi,ioj,byj->bxyo', input1, self.w1, input2)
#         input3 = tf.concat((input1,input2),axis=-1)
#         logits_2 = tf.einsum('io,byi->byo', self.w2, input3)
#         logits_2 = logits_2[:,tf.newaxis,:,:]
#         return logits_1+logits_2

class Biaffine_2(tf.keras.layers.Layer):
    def __init__(self, in_size, out_size,MAX_LEN):
        super(Biaffine_2, self).__init__()
        self.w1 = self.add_weight(
            name='weight1', 
            shape=(in_size, out_size, in_size),
            trainable=True)
        self.w2 = self.add_weight(
            name='weight2', 
            shape=(2*in_size + 1, out_size),
            trainable=True)
        self.MAX_LEN = MAX_LEN
        
    def call(self, input1, input2):
        f1 = tf.expand_dims(input1,2)
        f2 = tf.expand_dims(input2,1)
        f1 = tf.tile(f1,multiples=(1,1,self.MAX_LEN,1))
        f2 = tf.tile(f2,multiples=(1,self.MAX_LEN,1,1))
        concat_f1f2 = tf.concat((f1,f2),axis=-1)
        concat_f1f2 = tf.concat((concat_f1f2,tf.ones_like(concat_f1f2[..., :1])), axis=-1)
        # bxi,oij,byj->boxy
        logits_1 = tf.einsum('bxi,ioj,byj->bxyo', input1, self.w1, input2)
        logits_2 = tf.einsum('bijy,yo->bijo',concat_f1f2,self.w2)
        return logits_1+logits_2 
    
# def build_model(pretrained_path,config,MAX_LEN,Cs_num,cs_em_size,R_num):
#     ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
#     att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
#     cs = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    
#     config.output_hidden_states = True
#     bert_model = TFBertModel.from_pretrained(pretrained_path,config=config,from_pt=True)
#     x, _, hidden_states = bert_model(ids,attention_mask=att)
#     layer_1 = hidden_states[-1]
    
#     start_logits = tf.keras.layers.Dense(Cs_num,activation = 'sigmoid')(layer_1)
#     start_logits = tf.keras.layers.Lambda(lambda x: x**2,name='s_start')(start_logits)
    
#     end_logits = tf.keras.layers.Dense(Cs_num,activation = 'sigmoid')(layer_1)
#     end_logits = tf.keras.layers.Lambda(lambda x: x**2,name='s_end')(end_logits)
    
#     cs_emb = tf.keras.layers.Embedding(Cs_num,cs_em_size)(cs)
#     concat_cs = tf.keras.layers.Concatenate(axis=-1)([layer_1,cs_emb])
    
#     f1 = tf.keras.layers.Dense(128,activation='relu')(concat_cs)
#     f2 = tf.keras.layers.Dense(128,activation='relu')(concat_cs)
    
#     Biaffine_layer = Biaffine(128,R_num,bias_x=True, bias_y=True)
#     output_logist = Biaffine_layer(f1,f2)
#     output_logist = tf.keras.layers.Activation('sigmoid')(output_logist)
#     output_logist = tf.keras.layers.Lambda(lambda x: x**4,name='relation')(output_logist)
    
#     model = tf.keras.models.Model(inputs=[ids,att,cs], outputs=[start_logits,end_logits,output_logist])
#     model_2 = tf.keras.models.Model(inputs=[ids,att], outputs=[start_logits,end_logits])
#     model_3 = tf.keras.models.Model(inputs=[ids,att,cs], outputs=[output_logist])
#     return model,model_2,model_3

# def build_model_2(pretrained_path,config,MAX_LEN,Cs_num,cs_em_size,R_num):
#     ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
#     att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
#     cs = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    
#     config.output_hidden_states = True
#     bert_model = TFBertModel.from_pretrained(pretrained_path,config=config,from_pt=True)
#     x, _, hidden_states = bert_model(ids,attention_mask=att)
#     layer_1 = hidden_states[-1]
#     layer_2 = hidden_states[-2]
    
#     start_logits = tf.keras.layers.Dense(Cs_num,activation = 'sigmoid')(layer_1)
#     start_logits = tf.keras.layers.Lambda(lambda x: x**2,name='s_start')(start_logits)
    
#     end_logits = tf.keras.layers.Dense(Cs_num,activation = 'sigmoid')(layer_1)
#     end_logits = tf.keras.layers.Lambda(lambda x: x**2,name='s_end')(end_logits)
    
#     cs_emb = tf.keras.layers.Embedding(Cs_num,cs_em_size)(cs)
    
#     concat_cs = tf.keras.layers.Concatenate(axis=-1)([layer_1,layer_2,cs_emb])
    
#     f1 = tf.keras.layers.Dropout(0.2)(concat_cs)
#     f1 = tf.keras.layers.Dense(512,activation='relu')(f1)
#     f1 = tf.keras.layers.Dense(128,activation='relu')(f1)
    
#     f2 = tf.keras.layers.Dropout(0.2)(concat_cs)
#     f2 = tf.keras.layers.Dense(512,activation='relu')(f2)
#     f2 = tf.keras.layers.Dense(128,activation='relu')(f2)
    
#     Biaffine_layer = Biaffine_2(128,R_num,MAX_LEN)
#     output_logist = Biaffine_layer(f1,f2)
#     output_logist = tf.keras.layers.Activation('sigmoid')(output_logist)
#     output_logist = tf.keras.layers.Lambda(lambda x: x**4,name='relation')(output_logist)
    
#     model = tf.keras.models.Model(inputs=[ids,att,cs], outputs=[start_logits,end_logits,output_logist])
#     model_2 = tf.keras.models.Model(inputs=[ids,att], outputs=[start_logits,end_logits])
#     model_3 = tf.keras.models.Model(inputs=[ids,att,cs], outputs=[output_logist])
#     return model,model_2,model_3

def build_model_3(pretrained_path,config,MAX_LEN,Cs_num,cs_em_size,R_num):
    ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    cs = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    
    config.output_hidden_states = True
    bert_model = TFBertModel.from_pretrained(pretrained_path,config=config,from_pt=True)
    x, _, hidden_states = bert_model(ids,attention_mask=att)
    layer_1 = hidden_states[-1]
    layer_2 = hidden_states[-2]
    
    
    start_logits = tf.keras.layers.Dense(256,activation = 'relu')(layer_1)
    start_logits = tf.keras.layers.Dense(Cs_num,activation = 'sigmoid')(start_logits)
    start_logits = tf.keras.layers.Lambda(lambda x: x**2,name='s_start')(start_logits)
    
    end_logits = tf.keras.layers.Dense(256,activation = 'relu')(layer_1)
    end_logits = tf.keras.layers.Dense(Cs_num,activation = 'sigmoid')(end_logits)
    end_logits = tf.keras.layers.Lambda(lambda x: x**2,name='s_end')(end_logits)
    
    cs_emb = tf.keras.layers.Embedding(Cs_num,cs_em_size)(cs)
    
    concat_cs = tf.keras.layers.Concatenate(axis=-1)([layer_1,layer_2])
    
    f1 = tf.keras.layers.Dropout(0.2)(concat_cs)
    f1 = tf.keras.layers.Dense(256,activation='relu')(f1)
    f1 = tf.keras.layers.Dense(128,activation='relu')(f1)
    f1 = tf.keras.layers.Concatenate(axis=-1)([f1,cs_emb])
    
    f2 = tf.keras.layers.Dropout(0.2)(concat_cs)
    f2 = tf.keras.layers.Dense(256,activation='relu')(f2)
    f2 = tf.keras.layers.Dense(128,activation='relu')(f2)
    f2 = tf.keras.layers.Concatenate(axis=-1)([f2,cs_emb])
    
    Biaffine_layer = Biaffine_2(128+cs_em_size,R_num,MAX_LEN)
    output_logist = Biaffine_layer(f1,f2)
    output_logist = tf.keras.layers.Activation('sigmoid')(output_logist)
    output_logist = tf.keras.layers.Lambda(lambda x: x**4,name='relation')(output_logist)
    
    model = tf.keras.models.Model(inputs=[ids,att,cs], outputs=[start_logits,end_logits,output_logist])
    model_2 = tf.keras.models.Model(inputs=[ids,att], outputs=[start_logits,end_logits])
    model_3 = tf.keras.models.Model(inputs=[ids,att,cs], outputs=[output_logist])
    return model,model_2,model_3


# def build_model_4(pretrained_path,config,MAX_LEN,Cs_num,cs_em_size,R_num):
#     ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
#     att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
#     cs = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    
#     config.output_hidden_states = True
#     bert_model = TFBertModel.from_pretrained(pretrained_path,config=config,from_pt=True)
#     x, _, hidden_states = bert_model(ids,attention_mask=att)
#     layer_1 = hidden_states[-1]
#     layer_2 = hidden_states[-2]
    
#     start_logits = tf.keras.layers.Dense(256,activation = 'relu')(layer_2)
#     start_logits = tf.keras.layers.Dense(Cs_num,activation = 'sigmoid')(start_logits)
#     start_logits = tf.keras.layers.Lambda(lambda x: x**2,name='s_start')(start_logits)
    
#     end_logits = tf.keras.layers.Dense(256,activation = 'relu')(layer_1)
#     end_logits = tf.keras.layers.Dense(Cs_num,activation = 'sigmoid')(end_logits)
#     end_logits = tf.keras.layers.Lambda(lambda x: x**2,name='s_end')(end_logits)
    
#     cs_emb = tf.keras.layers.Embedding(Cs_num,cs_em_size)(cs)
    
#     f1 = tf.keras.layers.Dropout(0.2)(layer_1)
#     f1 = tf.keras.layers.Dense(256,activation='relu')(f1)
#     f1 = tf.keras.layers.Dropout(0.2)(f1)
#     f1 = tf.keras.layers.Dense(128,activation='relu')(f1)
#     f1 = tf.keras.layers.Dropout(0.2)(f1)
#     f1 = tf.keras.layers.Dense(64,activation='relu')(f1)
#     f1 = tf.keras.layers.Concatenate(axis=-1)([f1,cs_emb])
    
#     f2 = tf.keras.layers.Dropout(0.2)(layer_1)
#     f2 = tf.keras.layers.Dense(256,activation='relu')(f2)
#     f2 = tf.keras.layers.Dropout(0.2)(f2)
#     f2 = tf.keras.layers.Dense(128,activation='relu')(f2)
#     f2 = tf.keras.layers.Dropout(0.2)(f2)
#     f2 = tf.keras.layers.Dense(64,activation='relu')(f2)
#     f2 = tf.keras.layers.Concatenate(axis=-1)([f2,cs_emb])
    
#     Biaffine_layer = Biaffine_2(64+cs_em_size,R_num,MAX_LEN)
#     output_logist = Biaffine_layer(f1,f2)
#     output_logist = tf.keras.layers.Activation('sigmoid')(output_logist)
#     output_logist = tf.keras.layers.Lambda(lambda x: x**4,name='relation')(output_logist)
    
#     model = tf.keras.models.Model(inputs=[ids,att,cs], outputs=[start_logits,end_logits,output_logist])
#     model_2 = tf.keras.models.Model(inputs=[ids,att], outputs=[start_logits,end_logits])
#     model_3 = tf.keras.models.Model(inputs=[ids,att,cs], outputs=[output_logist])
#     return model,model_2,model_3

def s_new_loss(true,pred):
    true = tf.cast(true,tf.float32)
    loss = K.sum(K.binary_crossentropy(true, pred))
    return loss

class Metrics(tf.keras.callbacks.Callback):
    def __init__(self,model_2,model_3,id2p,va_text_list,va_spo_list,va_input_ids,va_attention_mask,tokenizer):
        super(Metrics, self).__init__()
        self.model_2 = model_2
        self.model_3 = model_3
        self.id2p = id2p
        self.va_input_ids = va_input_ids
        self.va_attention_mask = va_attention_mask
        self.va_spo_list = va_spo_list
        self.va_text_list = va_text_list
        self.tokenizer = tokenizer
        
    def on_train_begin(self, logs=None):
        self.val_f1s = []
        self.best_val_f1 = 0
    
    def evaluate_data(self):
        Q_l = 0
        C_l = 0
        Y1 = self.model_2.predict([self.va_input_ids,self.va_attention_mask])
        for m in range(len(Y1[0])):
            question=[]
            answer=[]
            for z in self.va_spo_list[m]:
                question.append((z['subject'],z['predicate'],z['object']))
            start = np.where(Y1[0][m]>0.5)[0]
            start_tp = np.where(Y1[0][m]>0.5)[1]
            end = np.where(Y1[1][m]>0.5)[0]
            end_tp = np.where(Y1[1][m]>0.5)[1]
            subjects_str = []
            end_list = []
            s_top = np.zeros((1,128))
            for i,t in zip(start,start_tp):
                j = end[end >= i]
                te = end_tp[end >= i]
                if len(j) > 0 and te[0] == t:
                    j = j[0]
                    end_list.append(j)
                    subjects_str.append(rematch_text_word(self.tokenizer,self.va_text_list[m],self.va_input_ids[m],i,j))
                    s_top[0][j] = t
            if end_list:
                relation = self.model_3.predict([[self.va_input_ids[m]], [self.va_attention_mask[m]],s_top])
                s_e_o = np.where(relation[0]>0.5)
                for i in range(len(s_e_o[0])):
                    s_end = s_e_o[0][i]
                    o_end = s_e_o[1][i]
                    predicate = s_e_o[2][i]
                    if s_end in end_list and o_end in end_list:
                        s = subjects_str[end_list.index(s_end)]
                        o = subjects_str[end_list.index(o_end)]
                        p = self.id2p[predicate]
                        answer.append((s,p,o)) 
            print(answer)
            Q = set(question)
            S = set(answer)
            C_l += len(Q&S)
            Q_l += len(Q)+len(S)
        return 2*C_l/Q_l
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        _val_f1 = self.evaluate_data()
        self.val_f1s.append(_val_f1)
        logs['val_f1'] = _val_f1
        if _val_f1 > self.best_val_f1:
            self.model.save_weights('./muti_model/biaffine_04_f1={}_model.hdf5'.format(_val_f1))
            self.best_val_f1 = _val_f1
            print("best f1: {}".format(self.best_val_f1))
        else:
            print("val f1: {}, but not the best f1".format(_val_f1))
        return  
    
def proceed_var_data(text_list,spo_list,tokenizer,MAX_LEN):
    ct = len(text_list)
    MAX_LEN = MAX_LEN
    input_ids = np.zeros((ct,MAX_LEN),dtype='int32')
    attention_mask = np.zeros((ct,MAX_LEN),dtype='int32')
    for k in range(ct):
        context_k = text_list[k].lower().replace(' ','')
        enc_context = tokenizer.encode(context_k,max_length=MAX_LEN,truncation=True) 
        input_ids[k,:len(enc_context)] = enc_context
        attention_mask[k,:len(enc_context)] = 1
    return input_ids,attention_mask

def generator(input_ids,attention_mask,send_s_po,start_tokens,end_tokens,c_relation,batch_size):
    i=0
    while 1:
        input_ids_b = input_ids[i*batch_size:(i+1)*batch_size]
        attention_mask_b = attention_mask[i*batch_size:(i+1)*batch_size]
        send_s_po_b = send_s_po[i*batch_size:(i+1)*batch_size]
        start_tokens_b = start_tokens[i*batch_size:(i+1)*batch_size]
        end_tokens_b = end_tokens[i*batch_size:(i+1)*batch_size]
        c_relation_b = c_relation[i*batch_size:(i+1)*batch_size]
        # 最重要的就是这个yield，它代表返回，返回以后循环还是会继续，然后再返回。就比如有一个机器一直在作累加运算，但是会把每次累加中间结果告诉你一样，直到把所有数加完
        yield({'input_1': input_ids_b, 'input_2': attention_mask_b,'input_3':send_s_po_b}, 
              {'s_start': start_tokens_b,'s_end':end_tokens_b,'relation':c_relation_b})
        i = (i+1)%(len(input_ids)//batch_size)
        
def main():
    pretrained_path = '/root/zhengyanzhao/comment/emotion/model/'
    config_path = os.path.join(pretrained_path, 'bert_config.json')
    vocab_path = os.path.join(pretrained_path, 'vocab.txt')
    tokenizer = BertTokenizer.from_pretrained(vocab_path)
    config = BertConfig.from_json_file(config_path)
    MAX_LEN = 128
    text_list,spo_list = load_data('./train_data.json')
    p2id,id2p = load_ps('./all_50_schemas')
    s_type = []
    for x in spo_list:
        for s in x:
            s_type.append(s['object_type'])
            s_type.append(s['subject_type'])
    s2id = {}
    i = 1
    for st in list(set(s_type)):
        s2id[st] = i
        i += 1
    input_ids,attention_mask,start_tokens,end_tokens,send_s_po,c_relation = proceed_data(text_list,spo_list,p2id,s2id,tokenizer,MAX_LEN)
    del text_list
    del spo_list
    
    va_text_list,va_spo_list = load_valid_data('./dev_data.json')
    va_input_ids,va_attention_mask = proceed_var_data(va_text_list,va_spo_list,tokenizer,MAX_LEN)
    
    batch_size = 16
    steps_per_epoch = len(input_ids)//batch_size
    eopch = 20
    
    K.clear_session()
    model,model_2,model_3 = build_model_3(pretrained_path,config,MAX_LEN,Cs_num=len(s2id)+1,cs_em_size=16,R_num=len(p2id))
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(loss={'s_start': s_new_loss,'s_end': s_new_loss,'relation': s_new_loss},loss_weights=[1,1,0.1],optimizer=optimizer)
#     model.fit([input_ids,attention_mask,send_s_po],[start_tokens,end_tokens,c_relation],epochs=10,batch_size=64,callbacks=[Metrics(model_2,model_3,id2p,va_text_list,va_spo_list,va_input_ids,va_attention_mask,tokenizer)])
    model.fit_generator(generator(input_ids,attention_mask,send_s_po,start_tokens,end_tokens,c_relation,batch_size),epochs=eopch,steps_per_epoch=steps_per_epoch,verbose=1,
                       callbacks=[Metrics(model_2,model_3,id2p,va_text_list,va_spo_list,va_input_ids,va_attention_mask,tokenizer)])
if __name__ == '__main__':
    main()
