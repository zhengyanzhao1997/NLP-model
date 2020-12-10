import pandas as pd
import json
import re
import os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
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
    sop_count = 0
    with open(path) as json_file:
        for i in json_file:
            text_list.append(eval(i)['text'])
            spo_list.append(eval(i)['spo_list'])
            sop_count += len(eval(i)['spo_list'])
    c = list(zip(text_list,spo_list))
    random.shuffle(c)
    text_list,spo_list = zip(*c)
    return text_list,spo_list,sop_count

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



def proceed_data(text_list,spo_list,p2id,id2p,tokenizer,MAX_LEN,sop_count):
    id_label = {}
    ct = len(text_list)
    MAX_LEN = MAX_LEN
    print(sop_count)
    input_ids = np.zeros((sop_count,MAX_LEN),dtype='int32')
    attention_mask = np.zeros((sop_count,MAX_LEN),dtype='int32')
    start_tokens = np.zeros((sop_count,MAX_LEN),dtype='int32')
    end_tokens = np.zeros((sop_count,MAX_LEN),dtype='int32')
    send_s_po = np.zeros((sop_count,2),dtype='int32')
    object_start_tokens = np.zeros((sop_count,MAX_LEN,len(p2id)),dtype='int32')
    object_end_tokens = np.zeros((sop_count,MAX_LEN,len(p2id)),dtype='int32')
    index_vaild = -1
    for k in range(ct):
        context_k = text_list[k].lower().replace(' ','')
        enc_context = tokenizer.encode(context_k,max_length=MAX_LEN,truncation=True) 
        if len(spo_list[k])==0:
            continue          
        start = []
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
            if len(toks)>0:
                S_start = toks[0]+1
                S_end = toks[-1]+1
                if (S_start,S_end) not in start:
                    index_vaild += 1
                    start.append((S_start,S_end))
                    input_ids[index_vaild,:len(enc_context)] = enc_context
                    attention_mask[index_vaild,:len(enc_context)] = 1
                    start_tokens[index_vaild,S_start] = 1
                    end_tokens[index_vaild,S_end] = 1
                    send_s_po[index_vaild,0] = S_start
                    send_s_po[index_vaild,1] = S_end
                    S_index.append([j,index_vaild])
                else:
                    S_index.append([j,index_vaild])
        if len(S_index) > 0:
            for index_ in range(len(S_index)):
                #随机选取object的首位，如果选取错误，则作为负样本
                object_text_k = spo_list[k][S_index[index_][0]]['object'].lower().replace(' ','')
                predicate = spo_list[k][S_index[index_][0]]['predicate']
                p_id = p2id[predicate]
                chars = np.zeros((len(context_k)))
                index = context_k.find(object_text_k)
                chars[index:index+len(object_text_k)]=1
                offsets = [] 
                idx = 0
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
                    P_start = toks[0]+1
                    P_end = toks[-1]+1
                    object_start_tokens[S_index[index_][1]][P_start,p_id] = 1
                    object_end_tokens[S_index[index_][1]][P_end,p_id] = 1
    return input_ids[:index_vaild],attention_mask[:index_vaild],start_tokens[:index_vaild],\
end_tokens[:index_vaild],send_s_po[:index_vaild],object_start_tokens[:index_vaild],\
object_end_tokens[:index_vaild],id_label

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
    def __init__(self,model_2,model_3,id2tag,va_text_list,va_spo_list,va_input_ids,va_attention_mask,tokenizer):
        super(Metrics, self).__init__()
        self.model_2 = model_2
        self.model_3 = model_3
        self.id2tag = id2tag
        self.va_input_ids = va_input_ids
        self.va_attention_mask = va_attention_mask
        self.va_spo_list = va_spo_list
        self.va_text_list = va_text_list
        self.tokenizer = tokenizer
        
    def on_train_begin(self, logs=None):
        self.val_f1s = []
        self.best_val_f1 = 0
    
    def get_same_element_index(self,ob_list):
        return [i for (i, v) in enumerate(ob_list) if v == 1]
    
    def evaluate_data(self):
        Y1 = self.model_2.predict([self.va_input_ids,self.va_attention_mask])
        question=[]
        answer=[]
        for m in range(len(Y1[0])):
            for z in self.va_spo_list[m]:
                question.append((z['subject'],z['predicate'],z['object']))
            start = np.where(Y1[0][m]>0.5)[0]
            end = np.where(Y1[1][m]>0.5)[0]
            subjects = []
            for i in start:
                j = end[end >= i]
                if len(j) > 0:
                    j = j[0]
                    subjects.append((i, j))
            if subjects:
                token_ids_2 = np.repeat([self.va_input_ids[m]], len(subjects), 0)
                attention_mask_2 = np.repeat([self.va_attention_mask[m]], len(subjects), 0)
                subjects = np.array(subjects)
                object_preds_start,object_preds_end = self.model_3.predict([token_ids_2, attention_mask_2, subjects])
                for subject,object_start,object_end in zip(subjects,object_preds_start,object_preds_end):
                    sub = rematch_text_word(self.tokenizer,self.va_text_list[m],self.va_input_ids[m],subject[0],subject[1])
                    start = np.argwhere(object_start > 0.5)
                    end = np.argwhere(object_end > 0.5)
                    for _start, predicate1 in start:
                        for _end, predicate2 in end:
                            if _start <= _end and predicate1 == predicate2:
                                ans = rematch_text_word(self.tokenizer,self.va_text_list[m],self.va_input_ids[m],_start,_end)
                                answer.append((sub,self.id2tag[predicate1],ans))
                                break
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
            self.model.save_weights('./model_/09_f1={}_model.hdf5'.format(_val_f1))
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

def get_initializer(initializer_range: float = 0.02) -> tf.initializers.TruncatedNormal:
    """
    Creates a :obj:`tf.initializers.TruncatedNormal` with the given range.

    Args:
        initializer_range (`float`, defaults to 0.02): Standard deviation of the initializer range.

    Returns:
        :obj:`tf.initializers.TruncatedNormal`: The truncated normal initializer.
    """
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)

def shape_list(tensor: tf.Tensor) -> List[int]:
    """
    Deal with dynamic shape in tensorflow cleanly.

    Args:
        tensor (:obj:`tf.Tensor`): The tensor we want the shape of.

    Returns:
        :obj:`List[int]`: The shape of the tensor as a list.
    """
    static = tensor.shape.as_list()
    dynamic = tf.shape(tensor)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

class TFBertSelfAttention(tf.keras.layers.Layer):
    def __init__(self,hidden_size,num_attention_heads,**kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (self.hidden_size, self.num_attention_heads)
            )
        assert self.hidden_size % self.num_attention_heads == 0
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = tf.keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(0.02), name="query"
        )
        self.key = tf.keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(0.02), name="key"
        )
        self.value = tf.keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(0.02), name="value"
        )
        self.dropout = tf.keras.layers.Dropout(0.1)

    def transpose_for_scores(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_attention_heads, self.attention_head_size))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, hidden_states, attention_mask, head_mask, output_attentions, training=False):
        batch_size = shape_list(hidden_states)[0]
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = tf.matmul(
            query_layer, key_layer, transpose_b=True
        )  # (batch size, num_heads, seq_len_q, seq_len_k)
        dk = tf.cast(shape_list(key_layer)[-1], attention_scores.dtype)  # scale attention_scores
        attention_scores = attention_scores / tf.math.sqrt(dk)
        if attention_mask is not None:
            extended_attention_mask = tf.cast(attention_mask,tf.float32)
            extended_attention_mask = extended_attention_mask[:, tf.newaxis, tf.newaxis, :]
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            # Apply the attention mask is (precomputed for all layers in TFBertModel call() function)
            attention_scores = attention_scores + extended_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = tf.nn.softmax(attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs, training=training)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = tf.matmul(attention_probs, value_layer)
        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])
        context_layer = tf.reshape(
            context_layer, (batch_size, -1, self.all_head_size)
        )  # (batch_size, seq_len_q, all_head_size)
        outputs = (context_layer, attention_probs) if output_attentions else context_layer
        return outputs

def extract_subject(inputs):
    """根据subject_ids从output中取出subject的向量表征
    """
    output, subject_ids = inputs
    start = tf.gather(output,subject_ids[:,0],axis=1,batch_dims=0)
    end = tf.gather(output,subject_ids[:,1],axis=1,batch_dims=0)
    subject = tf.keras.layers.Concatenate(axis=2)([start, end])
    return subject[:,0]

def build_model(pretrained_path,config,MAX_LEN,p2id):
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
    
    position_emb_s = bert_model.bert.get_input_embeddings().position_embeddings(s_po_index[:,0])
    position_emb_e = bert_model.bert.get_input_embeddings().position_embeddings(s_po_index[:,1])
    position_embedding = position_emb_s + position_emb_e
    position_embedding = position_embedding[:,tf.newaxis,:]
    add_position = Normalization_1 + position_embedding
    
    self_attenion = TFBertSelfAttention(768,1)(add_position,att,head_mask=None,output_attentions=False)
    dense = tf.keras.layers.Dense(768,activation='relu')(self_attenion)
    dense = tf.keras.layers.Dropout(0.2)(dense)
    dense = tf.keras.layers.Dense(512,activation='relu')(dense)
    
    op_out_put_start = tf.keras.layers.Dense(len(p2id),activation = 'sigmoid')(dense)
    op_out_put_start = tf.keras.layers.Lambda(lambda x: x**4)(op_out_put_start)
    
    op_out_put_end = tf.keras.layers.Dense(len(p2id),activation = 'sigmoid')(dense)
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
    text_list,spo_list,sop_count = load_data('./train_data.json')
    p2id,id2p = load_ps('./all_50_schemas')

    pretrained_path = '/root/zhengyanzhao/comment/emotion/model/'
    config_path = os.path.join(pretrained_path, 'bert_config.json')
    vocab_path = os.path.join(pretrained_path, 'vocab.txt')
    tokenizer = BertTokenizer.from_pretrained(vocab_path)
    config = BertConfig.from_json_file(config_path)
    MAX_LEN = 128
    input_ids,attention_mask,start_tokens,end_tokens,send_s_po,object_start_tokens,object_end_tokens,id_label = proceed_data(text_list,spo_list,p2id,id2p,tokenizer,MAX_LEN,sop_count)
    
    shuffle_ix = np.random.permutation(np.arange(len(input_ids)))
    input_ids = input_ids[shuffle_ix]
    attention_mask = attention_mask[shuffle_ix]
    start_tokens = start_tokens[shuffle_ix]
    end_tokens = end_tokens[shuffle_ix]
    send_s_po = send_s_po[shuffle_ix]
    object_start_tokens = object_start_tokens[shuffle_ix]
    object_end_tokens = object_end_tokens[shuffle_ix]
    
    del text_list
    del spo_list
    del sop_count
    
    label_id = pd.DataFrame([x for x in id_label.items()])
    label_id.to_csv('id_label_16.csv')
    id2label = pd.read_csv('./id_label_16.csv')
    id2tag = {}
    for i in range(len(id2label['1'])):
        id2tag[id2label['0'][i]]=id2label['1'][i] 
        
    K.clear_session()
    model,model_2,model_3 = build_model(pretrained_path,config,MAX_LEN,p2id)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(loss={'lambda': new_loss,
                    'lambda_1': new_loss,
                    'lambda_2': new_loss,
                    'lambda_3': new_loss},optimizer=optimizer)

    va_text_list,va_spo_list,sop_count = load_data('./dev_data.json')
    va_input_ids,va_attention_mask = proceed_var_data(va_text_list,va_spo_list,tokenizer,MAX_LEN)
    model.fit([input_ids,attention_mask,send_s_po],\
              [start_tokens,end_tokens,object_start_tokens,object_end_tokens], \
            epochs=10, batch_size=32,callbacks=[Metrics(model_2,model_3,id2tag,va_text_list,va_spo_list,va_input_ids,va_attention_mask,tokenizer)])

if __name__ == '__main__':
    main()
