import pandas as pd
import json
import re
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf
from transformers import PegasusTokenizer,PegasusConfig,TFPegasusForConditionalGeneration
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
from tool_pagesus import *

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

def load_data(filename):
    D = []
    with open(filename) as f:
        for l in f:
            D.append(json.loads(l))
    return D

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
        y_true, y_pred = inputs
        y_true = tf.cast(y_true,tf.float32)
        y_true = y_true[:, 1:]
        y_mask = tf.cast(y_true > 0,tf.float32) 
        y_pred = y_pred[:, :-1]
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss

# class CrossEntropy_k(Loss):
  
#     def compute_loss(self,inputs,mask=None):
#         y_true, y_mask, y_pred = inputs
#         #y_true = tf.cast(y_true,y_pred.dtype)
#         y_mask = tf.cast(y_mask,y_pred.dtype)
#         y_true = y_true[:, 1:]  # 目标token_ids
#         y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
#         y_pred = y_pred[:, :-1]  # 预测序列，错开一位
#         pos_loss = tf.gather(y_pred,y_true[..., None],batch_dims=len(tf.shape(y_true[..., None]))-1)[...,0]
#         y_pred = tf.nn.top_k(y_pred, k = 20)[0]
#         neg_loss = tf.math.reduce_logsumexp(y_pred, axis=-1)
#         loss = neg_loss - pos_loss
#         loss = K.sum(loss * y_mask) / K.sum(y_mask)
#         return loss

class AutoTitle(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        ids,mask_att = inputs
        shape_list = np.array(output_ids).shape
        top_index = shape_list[0]
        get_index = shape_list[-1]
        decoder_input_ids = np.zeros((top_index,self.max_decode_len),dtype='int32')
        for i in range(top_index):
            decoder_input_ids[i][:len(output_ids[i])] = output_ids[i]
        return self.model([ids,mask_att,decoder_input_ids])[:,get_index-1]
    
    def generate(self,text,tokenizer,max_encode,topk=1):
        input_dict = tokenizer(text,max_length = max_encode,truncation=True,padding=True)
        token_ids = input_dict['input_ids']
        ids = np.zeros((1,max_encode),dtype='int32')
        mask_att = np.zeros((1,max_encode),dtype='int32')
        len_ = len(token_ids)
        ids[0][:len_] = token_ids
        mask_att[0][:len_] = input_dict['attention_mask']
        output_ids = self.beam_search([ids,mask_att],topk=topk)
        return tokenizer.decode(output_ids)

def build_model(pretrained_path,psus_config,Max_len,decode_max_len):
    input_ids = tf.keras.layers.Input((Max_len,), dtype=tf.int32)
    attention_mask = tf.keras.layers.Input((Max_len,), dtype=tf.int32)
    decoder_input_ids = tf.keras.layers.Input((decode_max_len), dtype=tf.int32)
    
    psus_config.encoder_attention_type = 'full'
    Pegasus = TFPegasusForConditionalGeneration.from_pretrained(pretrained_path,config=psus_config,from_pt=True)
    logits = Pegasus({"input_ids":input_ids,"attention_mask":attention_mask,"decoder_input_ids":decoder_input_ids}).logits
    
    out_put = tf.keras.layers.Activation('softmax')(logits)
    output = CrossEntropy(1)([decoder_input_ids,out_put])
    
    model = tf.keras.models.Model(inputs=[input_ids,attention_mask,decoder_input_ids],outputs=output)
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    model.compile(optimizer=optimizer)
    model.summary()
    return model

def just_predict(autotitle,tokenizer,maxlen,valid_data):
    pred_result = {}
    for d in tqdm(valid_data,desc='predicting'):
        text_list = d['text']
        text_list = [i['content'] for i in text_list]
        result_signle = []
        for text in text_list:
            generate_s = autotitle.generate(text,tokenizer,maxlen,topk=8)
            result_signle.append(generate_s)
        pred_result[str(d['id'])] = ' '.join(result_signle)
    return pred_result

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

def main():
    pagesus_pretrain_path = './page_arciv/'
    tokenizer = PegasusTokenizer.from_pretrained(pagesus_pretrain_path)
    config_path = os.path.join(pagesus_pretrain_path,'config.json')
    psus_config = PegasusConfig.from_json_file(config_path)
    MAX_LEN = 1024
    decode_max_len = 256
    data  = load_data('./final_test_data_list.json')
    model = build_model(pagesus_pretrain_path,psus_config,MAX_LEN,decode_max_len)
    model.load_weights('./pagesus_section/best_model.hdf5')
    autotitle = AutoTitle(start_id=tokenizer.pad_token_id, end_id=tokenizer.eos_token_id,maxlen=256,max_decode_len=decode_max_len,model=model)
    
    result = just_predict(autotitle,tokenizer,MAX_LEN,data)
    with open('./pred_result.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(result,ensure_ascii=False,cls=NpEncoder))
    
if __name__ == '__main__':
    main()
