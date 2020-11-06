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

def data_proceed(path,batch_size,tokenizer):
    data = pd.read_csv(path)
    data = data.sample(frac=1)
    inputs_1 = tokenizer(list(data['sentence1']), padding=True, truncation=True, return_tensors="tf",max_length=30)
    inputs_2 = tokenizer(list(data['sentence2']), padding=True, truncation=True, return_tensors="tf",max_length=30)
    inputs_1  = dict(inputs_1)
    inputs_1['input_ids_2'] = inputs_2['input_ids']
    inputs_1['token_type_ids_2'] = inputs_2['token_type_ids']
    inputs_1['attention_mask_2'] = inputs_2['attention_mask']
    label = list(data['label'])
    steps = len(label)//batch_size
    x = tf.data.Dataset.from_tensor_slices((dict(inputs_1),label))
    return x,steps

# class Metrics(tf.keras.callbacks.Callback):
#     def __init__(self, valid_data):
#         super(Metrics, self).__init__()
#         self.validation_data = valid_data
        
#     def on_train_begin(self, logs=None):
#         self.val_f1s = []
#         self.best_val_f1 = 0

#     def on_epoch_end(self, epoch, logs=None):
#         logs = logs or {}
#         val_predict = np.argmax(self.model(self.validation_data[0]), -1)
#         val_targ = self.validation_data[1]
#         _val_f1 = f1_score(val_targ, val_predict, average='macro')
#         self.val_f1s.append(_val_f1)
#         logs['val_f1'] = _val_f1
#         if _val_f1 > self.best_val_f1:
#             self.model.save_pretrained('./checkpoints/weights-f1={}.hdf5'.format(_val_f1))
#             self.best_val_f1 = _val_f1
#             print("best f1: {}".format(self.best_val_f1))
#         else:
#             print("val f1: {}, but not the best f1".format(_val_f1))
#         return

# class WarmupExponentialDecay(Callback):
#     def __init__(self,lr_base=0.0002,lr_min=0.0,decay=0,warmup_epochs=0):
#         self.num_passed_batchs = 0   #一个计数器
#         self.warmup_epochs=warmup_epochs  
#         self.lr=lr_base #learning_rate_base
#         self.lr_min=lr_min #最小的起始学习率,此代码尚未实现
#         self.decay=decay  #指数衰减率
#         self.steps_per_epoch=0 #也是一个计数器
#     def on_batch_begin(self, batch, logs=None):
#         # params是模型自动传递给Callback的一些参数
#         if self.steps_per_epoch==0:
#             #防止跑验证集的时候呗更改了
#             if self.params['steps'] == None:
#                 self.steps_per_epoch = np.ceil(1. * self.params['samples'] / self.params['batch_size'])
#             else:
#                 self.steps_per_epoch = self.params['steps']
#         if self.num_passed_batchs < self.steps_per_epoch * self.warmup_epochs:
#             K.set_value(self.model.optimizer.lr,
#                         self.lr*(self.num_passed_batchs + 1) / self.steps_per_epoch / self.warmup_epochs)
#         else:
#             K.set_value(self.model.optimizer.lr,
#                         self.lr*((1-self.decay)**(self.num_passed_batchs-self.steps_per_epoch*self.warmup_epochs)))
#         self.num_passed_batchs += 1
#     def on_epoch_begin(self,epoch,logs=None):
#     #用来输出学习率的,可以删除
#         print("learning_rate:",K.get_value(self.model.optimizer.lr)) 


def focal_loss(target_tensor,prediction_tensor, alpha=0.25, gamma=2):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    zeros = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)
    target_tensor = tf.cast(target_tensor,prediction_tensor.dtype)
    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - prediction_tensor, zeros)
    
    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, prediction_tensor)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.math.log(tf.clip_by_value(prediction_tensor, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.math.log(tf.clip_by_value(1.0 - prediction_tensor, 1e-8, 1.0))
    return tf.math.reduce_sum(per_entry_cross_ent)

def main():
    if not os.path.exists('./checkpoints_2'):
        os.makedirs('./checkpoints_2')  
    pretrained_path = '/root/zhengyanzhao/comment/emotion/model/'
    train_path = './Text-Matching-master/data/LCQMC_train.csv'
    test_path = './Text-Matching-master/data/LCQMC_test.csv'
    config_path = os.path.join(pretrained_path, 'bert_config.json')
    vocab_path = os.path.join(pretrained_path, 'vocab.txt')
    config = BertConfig.from_json_file(config_path)
    tokenizer = BertTokenizer.from_pretrained(vocab_path)
    bert_ner_model = BertNerModel(pretrained_path,config=config)
    train_data,steps_per_epoch = data_proceed(train_path,64,tokenizer)
    test_data,validation_steps = data_proceed(test_path,64,tokenizer)
    train_data = train_data.shuffle(999).repeat().batch(64)
    test_data = test_data.batch(64)
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    bert_ner_model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['acc'])
#     if not os.path.exists('./checkpoints'):
#         os.makedirs('./checkpoints')  
#     ck_callback = tf.keras.callbacks.ModelCheckpoint('./checkpoints_2/weights.{epoch:02d}.hdf5',
#                                                  monitor='val_acc', 
#                                                  mode='max', verbose=1,
#                                                  save_best_only=True,
#                                                  save_weights_only=True)
#     tb_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs', profile_batch=0)
    bert_ner_model.fit(train_data,epochs=5,verbose=1,steps_per_epoch=steps_per_epoch,
                           validation_data=test_data,validation_steps=validation_steps)
#                        callbacks=[WarmupExponentialDecay(lr_base=2e-5,decay=0.00002,warmup_epochs=2),ck_callback,tb_callback])
    
if __name__ == '__main__':
    main()
