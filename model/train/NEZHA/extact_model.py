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
from tqdm import tqdm
from tool import *

input_size = 768
hidden_size = 384
epochs = 20
batch_size = 64
threshold = 0.2
num_folds = 15
max_len = 256

class ResidualGatedConv1D(tf.keras.layers.Layer):
    """门控卷积
    """
    def __init__(self, filters, kernel_size, dilation_rate=1, **kwargs):
        super(ResidualGatedConv1D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.supports_masking = True

    def build(self, input_shape):
        super(ResidualGatedConv1D, self).build(input_shape)
        self.conv1d = tf.keras.layers.Conv1D(
            filters=self.filters * 2,
            kernel_size=self.kernel_size,
            dilation_rate=self.dilation_rate,
            padding='same',
        )
        self.layernorm = tf.keras.layers.LayerNormalization()

        if self.filters != input_shape[-1]:
            self.dense = tf.keras.layers.Dense(self.filters, use_bias=False)

        self.alpha = self.add_weight(
            name='alpha', shape=[1], initializer='zeros'
        )

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            inputs = inputs * mask[:, :, None]

        outputs = self.conv1d(inputs)
        # 2*filters 相当于两组filters来 一组*sigmoid(另一组)
        gate = K.sigmoid(outputs[..., self.filters:])
        outputs = outputs[..., :self.filters] * gate
        outputs = self.layernorm(outputs)

        if hasattr(self, 'dense'):
            #用于对象是否包含对应的属性值
            inputs = self.dense(inputs)

        return inputs + self.alpha * outputs
    
    
def bulid_extract_model(max_len,input_size,hidden_size):
    input_ = tf.keras.layers.Input((max_len,input_size))
    x = tf.keras.layers.Masking()(input_)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(hidden_size, use_bias=False)(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = ResidualGatedConv1D(hidden_size, 3, dilation_rate=1)(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = ResidualGatedConv1D(hidden_size, 3, dilation_rate=2)(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = ResidualGatedConv1D(hidden_size, 3, dilation_rate=4)(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = ResidualGatedConv1D(hidden_size, 3, dilation_rate=8)(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = ResidualGatedConv1D(hidden_size, 3, dilation_rate=1)(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = ResidualGatedConv1D(hidden_size, 3, dilation_rate=1)(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    out_put = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=input_, outputs=out_put)
    return model


def evaluate(model,data,data_x,threshold=0.2):
    '''
    data : [sample_num,3,…]
    0:spilt_text
    1:label
    2:summary_text
    '''
    evaluater = 0
    pred = model.predict(data_x)[:,:,0]
    # [sample_num,256]
    for d,yp in tqdm(zip(data,pred),desc='evaluating'):
        yp = yp[:len(d[0])]
        yp = np.where(yp > threshold)[0]
        pred_sum = ''.join([d[0][i] for i in yp])
        evaluater += compute_main_metric(pred_sum,d[2],'token')
    return evaluater/len(data)


class Evaluator(tf.keras.callbacks.Callback):
    """训练回调
    """
    def __init__(self,threshold,valid_data,valid_x,fold):
        self.best_metric = 0.0
        self.threshold = threshold
        self.valid_data = valid_data
        self.valid_x = valid_x
        self.fold = fold

    def on_epoch_end(self, epoch, logs=None):
        eva = evaluate(self.model,self.valid_data, self.valid_x, self.threshold + 0.1)
        if  eva >= self.best_metric:  # 保存最优
            self.best_metric = eva
            self.model.save_weights('weights/extract_model_%s.hdf5' % self.fold)
            print('eva raise to %s'%eva)
        else:
            print('eva is %s,not raise'%eva)
                       
def data_split(data, fold, num_folds, mode):
    """划分训练集和验证集
    """
    if mode == 'train':
        D = [d for i, d in enumerate(data) if i % num_folds != fold]
    else:
        D = [d for i, d in enumerate(data) if i % num_folds == fold]

    if isinstance(data, np.ndarray):
        return np.array(D)
    else:
        return D

def load_data(filename):
    """加载数据
    返回：[(texts, labels, summary)]
    """
    D = []
    with open(filename) as f:
        for l in f:
            D.append(json.loads(l))
    return D
    
    
def main():
    
    data = load_data('sfzy_small_extract.json')
    data_x = np.load('vector_exteact.npy')
    data_y = np.zeros_like(data_x[...,:1])

    for i, d in enumerate(data):
        for j in d[1]:
            data_y[i][j][0] = 1

    for fold in range(num_folds):
        K.clear_session()
        model = bulid_extract_model(max_len,input_size,hidden_size)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['acc'])
        train_data = data_split(data, fold, num_folds, 'train')
        valid_data = data_split(data, fold, num_folds, 'valid')
        train_x = data_split(data_x, fold, num_folds, 'train')
        valid_x = data_split(data_x, fold, num_folds, 'valid')
        train_y = data_split(data_y, fold, num_folds, 'train')
        valid_y = data_split(data_y, fold, num_folds, 'valid')
        # 启动训练
        evaluator = Evaluator(threshold,valid_data,valid_x,fold)
        model.fit(
            train_x,
            train_y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[evaluator]
        )

if __name__ == '__main__':
    main()
