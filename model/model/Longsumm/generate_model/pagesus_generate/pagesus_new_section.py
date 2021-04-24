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

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
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

def impose_max_length(summary_text, max_tokens=600):
    text = summary_text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = re.split(r"\s+", text)
    tokens = [x for x in tokens if re.match(r"^[a-z0-9]+$", x)]
    tokens = tokens[0:min(max_tokens,len(tokens))]
    return " ".join(tokens)

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

class data_generator(DataGenerator):
    def __iter__(self, random=True):
        input_ids = np.zeros((self.batch_size,self.Max_len),dtype='int32')
        attention_mask = np.zeros((self.batch_size,self.Max_len),dtype='int32')
        decoder_input_ids = np.zeros((self.batch_size,self.decode_max_len),dtype='int32')
        index = 0
        for is_end, d in self.sample(random):
            summary = d['summary']
            content = d['artical']
            encode_dict = self.tokenizer(content,max_length=self.Max_len,truncation=True,padding=True)
            decoder_input_id = [0] + self.tokenizer.encode(summary,max_length=self.decode_max_len-1,truncation=True)
            input_id = encode_dict['input_ids']
            len_encode = len(input_id)
            len_decode = len(decoder_input_id)
            input_ids[index][:len_encode] = input_id
            attention_mask[index][:len_encode] = encode_dict['attention_mask']
            decoder_input_ids[index][:len_decode] = decoder_input_id
            index += 1
            if  index == self.batch_size or is_end:
                if not is_end:
                    yield [input_ids[:index],attention_mask[:index],decoder_input_ids[:index]]
                input_ids = np.zeros((self.batch_size,self.Max_len),dtype='int32')
                attention_mask = np.zeros((self.batch_size,self.Max_len),dtype='int32')
                decoder_input_ids = np.zeros((self.batch_size,self.decode_max_len),dtype='int32')
                index = 0

def just_show(autotitle,tokenizer,maxlen,valid_data):
    #score_avg = 0
    for d in tqdm(valid_data,desc='predicting'):
        label = d['summary']
        artical = d['artical']
        print(u'段落原文:',artical)
        print(u'标签摘要:',label)
        generate_s = autotitle.generate(artical,tokenizer,maxlen,topk=3)
        print(u'生成摘要:',generate_s)
        #score = compute_main_metric(impose_max_length(generate_s),impose_max_length(label))
        score = compute_main_metric(generate_s,label)
        print(u'评估分数:',score)
        #score_avg += score['main']
    #return score_avg/len(valid_data)

class Evaluator(tf.keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self,tokenizer,maxlen,autotitle,valid_data):
        self.lowest = 1e10
        self.score = 0
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.autotitle = autotitle
        self.valid_data = valid_data

    def on_epoch_end(self, epoch, logs=None):
        # 保存最优
        just_show(self.autotitle,self.tokenizer,self.maxlen,self.valid_data)
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
        #if score > self.score:
            self.model.save_weights('./pagesus_section/best_model.hdf5')
            #self.score = score
    # 演示效果

def main():
    pagesus_pretrain_path = './page_arciv/'
    tokenizer = PegasusTokenizer.from_pretrained(pagesus_pretrain_path)
    config_path = os.path.join(pagesus_pretrain_path,'config.json')
    psus_config = PegasusConfig.from_json_file(config_path)
    MAX_LEN = 1024
    decode_max_len = 256
    batch_size = 4
    data  = load_data('/home_zyz/new_section/abstract_final_extract.json')
    random.shuffle(data)
    print(len(data))
    print(data[0]['artical'])
    print(data[0]['summary'])
    valid_data = data[:5]
    train_data = data[5:]
    train_generator = data_generator(train_data,batch_size,MAX_LEN,decode_max_len,tokenizer)

    K.clear_session()
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    with strategy.scope():
        model = build_model(pagesus_pretrain_path,psus_config,MAX_LEN,decode_max_len)
    
    epochs = 50
    autotitle = AutoTitle(start_id=tokenizer.pad_token_id, end_id=tokenizer.eos_token_id,maxlen=256,max_decode_len = decode_max_len,model=model)
    evaluator = Evaluator(tokenizer,MAX_LEN,autotitle,valid_data)
    model.fit(train_generator.forfit(),steps_per_epoch=len(train_generator)-1,epochs=epochs,callbacks=[evaluator])
    
if __name__ == '__main__':
    main()
