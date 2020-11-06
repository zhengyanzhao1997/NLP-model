import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import model_selection
from transformers import *
from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import os
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback

class BertNerModel(TFBertPreTrainedModel):
    dense_layer = 256
    dense_layer2 = 128
    class_num = 6
    drop_out_rate = 0.5
    def __init__(self, config, *inputs, **kwargs):
        super(BertNerModel,self).__init__(config, *inputs, **kwargs)
        config.output_hidden_states = True
        self.bert_layer = TFBertMainLayer(config, name='bert')
        self.bert_layer.trainable = True
        self.liner_layer = tf.keras.layers.Dense(self.dense_layer,activation='relu')
        self.liner_layer2 = tf.keras.layers.Dense(self.dense_layer2,activation='relu')
        self.soft_max = tf.keras.layers.Dense(self.class_num,activation='softmax')
        self.drop_out = tf.keras.layers.Dropout(self.drop_out_rate)
    def call(self, inputs):
        hidden_states = self.bert_layer(inputs)
        tensor = tf.concat((hidden_states[2][-1][:,0],hidden_states[2][-2][:,0],hidden_states[2][-3][:,0],hidden_states[1]),1,)
        drop_out_l = self.drop_out(tensor)
        Dense_l = self.liner_layer(drop_out_l)
        Dense_l2 = self.liner_layer2(Dense_l)
        outputs = self.soft_max(Dense_l2)
        return outputs
