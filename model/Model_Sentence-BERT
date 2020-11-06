import numpy as np
import os
import pandas as pd
import tensorflow as tf
from sklearn import model_selection
from transformers import *
from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
from tensorflow.python.ops import array_ops

class BertNerModel(tf.keras.Model):
    dense_layer = 512
    class_num = 1
    drop_out_rate = 0.5
    def __init__(self,pretrained_path,config,*inputs,**kwargs):
        super(BertNerModel,self).__init__()
        config.output_hidden_states = True
        self.bert = TFBertModel.from_pretrained(pretrained_path,config=config,from_pt=True)
#         self.bert_layer = TFBertMainLayer(config, name='bert')
#         self.bert_layer.trainable = True
#         self.concat_layer = tf.keras.layers.Concatenate(name='concat_bert')
        self.liner_layer = tf.keras.layers.Dense(self.dense_layer,activation='relu')
        self.sigmoid = tf.keras.layers.Dense(self.class_num,activation='sigmoid')
        self.drop_out = tf.keras.layers.Dropout(self.drop_out_rate)      
#         self.faltten = tf.keras.layers.Flatten()
    def call(self,input_1):
        hidden_states_1,_,_ = self.bert((input_1['input_ids'],input_1['token_type_ids'],input_1['attention_mask']))
        hidden_states_2,_,_ = self.bert((input_1['input_ids_2'],input_1['token_type_ids_2'],input_1['attention_mask_2']))
        hidden_states_1 = tf.math.reduce_mean(hidden_states_1,1)
        hidden_states_2 = tf.math.reduce_mean(hidden_states_2,1)
        concat_layer = tf.concat((hidden_states_1,hidden_states_2,tf.abs(tf.math.subtract(hidden_states_1, hidden_states_2))),1,)
        drop_out_l = self.drop_out(concat_layer)
        Dense_l = self.liner_layer(drop_out_l)
        outputs = self.sigmoid(Dense_l)
        print(outputs.shape)
        return outputs
