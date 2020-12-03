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
import random


def load_data(path):
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

def proceed_data(text_list,spo_list,p2id,id2p,tokenizer,MAX_LEN):
    id_label = {}
    ct = len(text_list)
    MAX_LEN = MAX_LEN
    input_ids = np.zeros((ct,MAX_LEN),dtype='int32')
    attention_mask = np.zeros((ct,MAX_LEN),dtype='int32')
    start_tokens = np.zeros((ct,MAX_LEN),dtype='int32')
    end_tokens = np.zeros((ct,MAX_LEN),dtype='int32')
    send_s_po = np.zeros((ct,2),dtype='int32')
    object_start_tokens = np.zeros((ct,MAX_LEN,len(p2id)),dtype='int32')
    object_end_tokens = np.zeros((ct,MAX_LEN,len(p2id)),dtype='int32')
    invalid_index = []
    for k in range(ct):
        context_k = text_list[k].lower().replace(' ','')
        enc_context = tokenizer.encode(context_k,max_length=MAX_LEN,truncation=True) 
        if len(spo_list[k])==0:
            invalid_index.append(k)
            continue
        start = []
        end = []
        S_index = []
        for j in range(len(spo_list[k])):
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
            input_ids[k,:len(enc_context)] = enc_context
            attention_mask[k,:len(enc_context)] = 1
            if len(toks)>0:
                start_tokens[k,toks[0]+1] = 1
                end_tokens[k,toks[-1]+1] = 1
                start.append(toks[0]+1)
                end.append(toks[-1]+1)
                S_index.append(j)
                #随机抽取可以作为负样本提高准确率（不认同）
        if len(start) > 0:
            start_np = np.array(start)
            end_np = np.array(end)
            start_ = np.random.choice(start_np)
            end_ = np.random.choice(end_np[end_np >= start_])
            send_s_po[k,0] = start_
            send_s_po[k,1] = end_
            s_index = start.index(start_)
            #随机选取object的首位，如果选取错误，则作为负样本
            if end_ == end[s_index]:
                for index in range(len(start)):
                    if start[index] == start_ and end[index] == end_:
                        object_text_k = spo_list[k][S_index[index]]['object'].lower().replace(' ','')
                        predicate = spo_list[k][S_index[index]]['predicate']
                        p_id = p2id[predicate]
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
                            id_label[p_id] = predicate
                            object_start_tokens[k,toks[0]+1,p_id] = 1
                            object_end_tokens[k,toks[-1]+1,p_id] = 1
        else:
            invalid_index.append(k)
    return input_ids,attention_mask,start_tokens,end_tokens,send_s_po,object_start_tokens,object_end_tokens,invalid_index,id_label

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

class Metrics(tf.keras.callbacks.Callback):
    def __init__(self,model_2,model_3,id2tag,va_spo_list,va_input_ids,va_attention_mask,tokenizer):
        super(Metrics, self).__init__()
        self.model_2 = model_2
        self.model_3 = model_3
        self.id2tag = id2tag
        self.va_input_ids = va_input_ids
        self.va_attention_mask = va_attention_mask
        self.va_spo_list = va_spo_list
        self.tokenizer = tokenizer
        
    def on_train_begin(self, logs=None):
        self.val_f1s = []
        self.best_val_f1 = 0
    
    def get_same_element_index(self,ob_list):
        return [i for (i, v) in enumerate(ob_list) if v == 1]
    
    def evaluate_data(self):
        question=[]
        answer=[]
        Y1 = self.model_2.predict([self.va_input_ids,self.va_attention_mask])
        for i in range(len(Y1[0])):
            for z in self.va_spo_list[i]:
                question.append((z['subject'][0],z['subject'][-1],z['predicate'],z['object'][0],z['object'][-1]))
            x_ = [self.tokenizer.decode([t]) for t in self.va_input_ids[i]]
            x1 = np.array(Y1[0][i]>0.5,dtype='int32')
            x2 = np.array(Y1[1][i]>0.5,dtype='int32')
            union = x1 + x2
            index_list = self.get_same_element_index(list(union))
            start = 0
            S_list=[]
            while start+1 < len(index_list):
                S_list.append((index_list[start],index_list[start+1]+1))
                start += 2
            for os_s,os_e in S_list:
                S = ''.join(x_[os_s:os_e])
                Y2 = self.model_3.predict([[self.va_input_ids[i]],[self.va_attention_mask[i]],np.array([[os_s,os_e]])])
                for m in range(len(self.id2tag)):
                    x3 = np.array(Y2[0][0][:,m]>0.4,dtype='int32')
                    x4 = np.array(Y2[1][0][:,m]>0.4,dtype='int32')
                    if sum(x3)>0 and sum(x4)>0:
                        predict = self.id2tag[m]
                        union = x3 + x4
                        index_list = self.get_same_element_index(list(union))
                        start = 0
                        P_list=[]
                        while start+1 < len(index_list):
                            P_list.append((index_list[start],index_list[start+1]+1))
                            start += 2
                        for os_s,os_e in P_list:
                            if os_e>=os_s:
                                P = ''.join(x_[os_s:os_e])
                                answer.append((S[0],S[-1],predict,P[0],P[-1]))
        Q = set(question)
        S = set(answer)
        f1 = 2*len(Q&S)/(len(Q)+len(S))
        return f1
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        _val_f1 = self.evaluate_data()
        self.val_f1s.append(_val_f1)
        logs['val_f1'] = _val_f1
        if _val_f1 > self.best_val_f1:
            self.model.save_weights('./model_/02_f1={}_model.hdf5'.format(_val_f1))
            self.best_val_f1 = _val_f1
            print("best f1: {}".format(self.best_val_f1))
        else:
            print("val f1: {}, but not the best f1".format(_val_f1))
        return   

def delete_invalid_data(invlid_index,data):
    return np.delete(data,invlid_index,axis=0)


class LayerNormalization(tf.keras.layers.Layer):
    """(Conditional) Layer Normalization
    hidden_*系列参数仅为有条件输入时(conditional=True)使用
    """
    def __init__(
        self,
        center=True,
        scale=True,
        epsilon=None,
        conditional=False,
        hidden_units=None,
        hidden_activation='linear',
        hidden_initializer='glorot_uniform',
        **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        self.hidden_activation = activations.get(hidden_activation)
        self.hidden_initializer = initializers.get(hidden_initializer)
        self.epsilon = epsilon or 1e-12
    def compute_mask(self, inputs, mask=None):
        if self.conditional:
            masks = mask if mask is not None else []
            masks = [m[None] for m in masks if m is not None]
            if len(masks) == 0:
                return None
            else:
                return K.all(K.concatenate(masks, axis=0), axis=0)
        else:
            return mask
        
    def build(self, input_shape):
        super(LayerNormalization, self).build(input_shape)
        if self.conditional:
            shape = (input_shape[0][-1],)
        else:
            shape = (input_shape[-1],)
        if self.center:
            self.beta = self.add_weight(
                shape=shape, initializer='zeros', name='beta')
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape, initializer='ones', name='gamma')
        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = tf.keras.layers.Dense(
                    units=self.hidden_units,
                    activation=self.hidden_activation,
                    use_bias=False,
                    kernel_initializer=self.hidden_initializer)
            if self.center:
                self.beta_dense = tf.keras.layers.Dense(
                    units=shape[0], use_bias=False, kernel_initializer='zeros')
            if self.scale:
                self.gamma_dense = tf.keras.layers.Dense(
                    units=shape[0], use_bias=False, kernel_initializer='zeros')

    def call(self, inputs):
        """如果是条件Layer Norm，则默认以list为输入，第二个是condition
        """
        if self.conditional:
            inputs, cond = inputs
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)
            for _ in range(K.ndim(inputs) - K.ndim(cond)):
                cond = K.expand_dims(cond, 1)
            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma
        outputs = inputs
        if self.center:
            mean = K.mean(outputs, axis=-1, keepdims=True)
            outputs = outputs - mean
        if self.scale:
            variance = K.mean(K.square(outputs), axis=-1, keepdims=True)
            std = K.sqrt(variance + self.epsilon)
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta
        return outputs
    
def extract_subject(inputs):
    """根据subject_ids从output中取出subject的向量表征
    """
    output, subject_ids = inputs
    start = tf.gather(output,subject_ids[:,0],axis=1,batch_dims=0)
    end = tf.gather(output,subject_ids[:,1],axis=1,batch_dims=0)
    subject = tf.keras.layers.Concatenate(axis=2)([start, end])
    return subject[:,0]

def build_model_2(pretrained_path,config,MAX_LEN,p2id):
    ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    s_po_index =  tf.keras.layers.Input((2,), dtype=tf.int32)
    
    config.output_hidden_states = True
    bert_model = TFBertModel.from_pretrained(pretrained_path,config=config,from_pt=True)
    x, _, hidden_states = bert_model(ids,attention_mask=att)

    layer_1 = hidden_states[-1]
    
    start_logits = tf.keras.layers.Dense(1,activation = 'sigmoid')(layer_1)
    start_logits = tf.keras.layers.Lambda(lambda x: x**2)(start_logits)
    
    end_logits = tf.keras.layers.Dense(1,activation = 'sigmoid')(layer_1)
    end_logits = tf.keras.layers.Lambda(lambda x: x**2)(end_logits)
    
    subject_1 = extract_subject([layer_1,s_po_index])
    Normalization_1 = LayerNormalization(conditional=True)([layer_1, subject_1])
    
    op_out_put_start = tf.keras.layers.Dense(len(p2id),activation = 'sigmoid')(Normalization_1)
    op_out_put_start = tf.keras.layers.Lambda(lambda x: x**4)(op_out_put_start)
    
    op_out_put_end = tf.keras.layers.Dense(len(p2id),activation = 'sigmoid')(Normalization_1)
    op_out_put_end = tf.keras.layers.Lambda(lambda x: x**4)(op_out_put_end)

    
    model = tf.keras.models.Model(inputs=[ids,att,s_po_index], outputs=[start_logits,end_logits,op_out_put_start,op_out_put_end])
    model_2 = tf.keras.models.Model(inputs=[ids,att], outputs=[start_logits,end_logits])
    model_3 = tf.keras.models.Model(inputs=[ids,att,s_po_index], outputs=[op_out_put_start,op_out_put_end])
    return model,model_2,model_3

def new_loss(true,pred):
    true = tf.cast(true,tf.float32)
    loss = K.sum(K.binary_crossentropy(true, pred))
    return loss

def main():
    text_list,spo_list = load_data('./train_data.json')
    p2id,id2p = load_ps('./all_50_schemas')
    
    pretrained_path = '/root/zhengyanzhao/comment/emotion/model/'
    config_path = os.path.join(pretrained_path, 'bert_config.json')
    vocab_path = os.path.join(pretrained_path, 'vocab.txt')
    tokenizer = BertTokenizer.from_pretrained(vocab_path)
    config = BertConfig.from_json_file(config_path)
    MAX_LEN = 128
    input_ids,attention_mask,start_tokens,end_tokens,send_s_po,object_start_tokens,object_end_tokens,invalid_index,id_label = proceed_data(text_list,spo_list,p2id,id2p,tokenizer,MAX_LEN)
    
    label_id = pd.DataFrame([x for x in id_label.items()])
    label_id.to_csv('id_label_9.csv')
    id2label = pd.read_csv('./id_label_9.csv')
    id2tag = {}
    for i in range(len(id2label['1'])):
        id2tag[id2label['0'][i]]=id2label['1'][i] 
        
    input_ids = delete_invalid_data(invalid_index,input_ids)
    attention_mask = delete_invalid_data(invalid_index,attention_mask)
    start_tokens = delete_invalid_data(invalid_index,start_tokens)
    end_tokens = delete_invalid_data(invalid_index,end_tokens)
    send_s_po = delete_invalid_data(invalid_index,send_s_po)
    object_start_tokens = delete_invalid_data(invalid_index,object_start_tokens)
    object_end_tokens = delete_invalid_data(invalid_index,object_end_tokens)
    
    K.clear_session()
    model,model_2,model_3 = build_model_2(pretrained_path,config,MAX_LEN,p2id)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(loss={'lambda': new_loss,
                    'lambda_1': new_loss,
                    'lambda_2': new_loss,
                    'lambda_3': new_loss},optimizer=optimizer)
    
#     sv = tf.keras.callbacks.ModelCheckpoint("./model_/weights8.hdf5", monitor='val_loss', verbose=2, save_best_only=True,save_weights_only=True, mode='auto', save_freq='epoch')
#     model.fit([input_ids_train,attention_mask_train,send_s_po_train],\
#               [start_tokens_train,end_tokens_train,object_start_tokens_train,object_end_tokens_train], \
#             epochs=5, batch_size=32,callbacks=[sv],\
#               validation_data=([input_ids_test,attention_mask_test,send_s_po_test],\
#                                [start_tokens_test,end_tokens_test,object_start_tokens_test,object_end_tokens_test]))

    va_text_list,va_spo_list = load_data('./dev_data.json')
    va_input_ids,va_attention_mask = proceed_var_data(va_text_list,va_spo_list,tokenizer,MAX_LEN)
    
    model.fit([input_ids,attention_mask,send_s_po],\
              [start_tokens,end_tokens,object_start_tokens,object_end_tokens], \
            epochs=20, batch_size=32,callbacks=[Metrics(model_2,model_3,id2tag,va_spo_list,va_input_ids,va_attention_mask,tokenizer)])

if __name__ == '__main__':
    main()
