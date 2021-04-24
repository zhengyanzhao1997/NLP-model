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
    
def encode_(x,y,tokenizer):
    train_texts, val_texts, train_tags, val_tags = train_test_split(x,y,test_size=0.2,random_state=1234)
    batch_x1 = tokenizer(train_texts, padding=True, truncation=True, return_tensors="tf",max_length=60)
    batch_x2 = tokenizer(val_texts, padding=True, truncation=True, return_tensors="tf",max_length=60)
    label_1  = tf.constant(train_tags)
    label_2  = tf.constant(val_tags)
    dataset_train = tf.data.Dataset.from_tensor_slices((dict(batch_x1),label_1))
    dataset_test = tf.data.Dataset.from_tensor_slices((dict(batch_x2),label_2))
    return dataset_train,dataset_test

class Metrics(Callback):
    def __init__(self):
        super(Metrics, self).__init__()
        
    def on_train_begin(self, logs=None):
        self.best_acc = 0

    def on_epoch_end(self, epoch, logs=None):
        print(logs)
        val_acc = logs['val_sparse_categorical_accuracy']
        if val_acc > self.best_acc:
            self.model.save_pretrained('./acc={}_model'.format(val_acc))
            self.best_acc = val_acc
            print("best acc: {}".format(self.best_acc))
        else:
            print("val acc: {}, but not the best acc".format(val_acc))
        return

# def focal_loss(label,pred,class_num=6, gamma=2):
#     label = tf.squeeze(tf.cast(tf.one_hot(tf.cast(label,tf.int32),class_num),pred.dtype)) 
#     pred = tf.clip_by_value(pred, 1e-8, 1.0)
#     w1 = tf.math.pow((1.0-pred),gamma)
#     L =  - tf.math.reduce_sum(w1 * label * tf.math.log(pred))
#     return L

def sparse_categorical_crossentropy(y_true, y_pred):
    y_true = tf.reshape(y_true, tf.shape(y_pred)[:-1])
    y_true = tf.cast(y_true, tf.int32)
    y_true = tf.one_hot(y_true, K.shape(y_pred)[-1])
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

def loss_with_gradient_penalty(model,epsilon=1):
    def loss_with_gradient_penalty_2(y_true, y_pred):
        loss = tf.math.reduce_mean(sparse_categorical_crossentropy(y_true, y_pred))
        embeddings = model.variables[0]
        gp = tf.math.reduce_sum(tf.gradients(loss, [embeddings])[0].values**2)
        return loss + 0.5 * epsilon * gp
    return loss_with_gradient_penalty_2

# def loss_with_gradient_penalty_constaint(model,epsilon=1):
#     def loss_with_gradient_penalty_2(y_true, y_pred):
#         loss = tf.math.reduce_mean(sparse_categorical_crossentropy(y_true, y_pred))
#         embeddings = model.variables[0]
#         gp = tf.math.reduce_sum(tf.gradients(loss, [embeddings])[0].values)
#         return loss + epsilon * gp
#     return loss_with_gradient_penalty_2

class WarmupExponentialDecay(Callback):
    def __init__(self,lr_base=0.0002,decay=0,warmup_epochs=0,steps_per_epoch=0):
        self.num_passed_batchs = 0   #一个计数器
        self.warmup_epochs=warmup_epochs  
        self.lr=lr_base #learning_rate_base
        self.decay=decay  #指数衰减率
        self.steps_per_epoch=steps_per_epoch #也是一个计数器
    def on_batch_begin(self, batch, logs=None):
        # params是模型自动传递给Callback的一些参数
        if self.steps_per_epoch==0:
            #防止跑验证集的时候呗更改了
            if self.params['steps'] == None:
                self.steps_per_epoch = np.ceil(1. * self.params['samples'] / self.params['batch_size'])
            else:
                self.steps_per_epoch = self.params['steps']
        if self.num_passed_batchs < self.steps_per_epoch * self.warmup_epochs:
            K.set_value(self.model.optimizer.lr,
                        self.lr*(self.num_passed_batchs + 1) / self.steps_per_epoch / self.warmup_epochs)
        else:
            K.set_value(self.model.optimizer.lr,
                        self.lr*((1-self.decay)**(self.num_passed_batchs-self.steps_per_epoch*self.warmup_epochs)))
        self.num_passed_batchs += 1
    def on_epoch_begin(self,epoch,logs=None):
    #用来输出学习率的,可以删除
        print("learning_rate:",K.get_value(self.model.optimizer.lr)) 

    
def main():
#     if not os.path.exists('./checkpoints'):
#         os.makedirs('./checkpoints')   
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs', profile_batch=0)
    pretrained_path = 'model/'
    config_path = os.path.join(pretrained_path, 'bert_config.json')
    vocab_path = os.path.join(pretrained_path, 'vocab.txt')
    # 加载config
    config = BertConfig.from_json_file(config_path)
    tokenizer = BertTokenizer.from_pretrained(vocab_path)
    bert_ner_model = BertNerModel.from_pretrained(pretrained_path,config=config,from_pt=True)
    
    data = pd.read_csv('data_proceed.csv')
    data = data.dropna()
    emotion_g2id = {}
    for i,j in enumerate(set(data['情绪标签'])):
        emotion_g2id[j]=i
    data['情绪标签'] = data['情绪标签'].apply(lambda x:emotion_g2id[x])
    
    data_size = len(data['情绪标签'])
    train_size = data_size*0.8
    train_test = data_size*0.2
    steps_per_epoch = train_size//16
    validation_step = train_test//16
    dataset_train,dataset_test = encode_(list(data['文本']),list(data['情绪标签']),tokenizer)
#     batch_x1 = tokenizer(list(data['文本']), padding=True, truncation=True, return_tensors="tf",max_length=60)
#     label_1  = tf.constant(list(data['情绪标签']))
#     dataset_train = tf.data.Dataset.from_tensor_slices((dict(batch_x1),label_1))
    dataset_train = dataset_train.shuffle(1234).repeat().batch(16)
#     steps_per_epoch = data_size//64
    dataset_test = dataset_test.batch(16)
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    bert_ner_model.compile(optimizer=optimizer, loss=[loss_with_gradient_penalty(bert_ner_model,0.5)] ,metrics=['sparse_categorical_accuracy'])
#     bert_ner_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])
    bert_ner_model.fit(dataset_train,epochs=5,verbose=1,steps_per_epoch=steps_per_epoch,
                           validation_data=dataset_test,validation_steps=validation_step,callbacks=[Metrics()])
#     bert_ner_model.save_pretrained('./my_mrpc_model/') 

if __name__ == '__main__':
    main()
