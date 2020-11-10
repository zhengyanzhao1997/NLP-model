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
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import StratifiedKFold
import random

def data_load(path):
    with open(path) as json_file:
        data = json.load(json_file)
    context = [x['context'] for x in data['data'][0]['paragraphs']]
    question = [x['qas'][0]['question'] for x in data['data'][0]['paragraphs']]
    answers_text = [x['qas'][0]['answers'][0]['text'] for x in data['data'][0]['paragraphs']]
    answer_start = [x['qas'][0]['answers'][0]['answer_start'] for x in data['data'][0]['paragraphs']]
    return context,question,answers_text,answer_start

def train_data_proceed(tokenizer,context,question,answers_text,answer_start,MAX_LEN):
    ct = len(context)
    input_ids = np.zeros((ct,MAX_LEN),dtype='int32')
    attention_mask = np.zeros((ct,MAX_LEN),dtype='int32')
    start_tokens = np.zeros((ct,MAX_LEN),dtype='int32')
    end_tokens = np.zeros((ct,MAX_LEN),dtype='int32')
    
    for k in range(ct):
        context_k = context[k]
        question_k = question[k]
        answers_text_k = answers_text[k]
        answer_start_k = answer_start[k]
        answer_start_k = answer_start_k - len(re.findall(' ',context_k[:answer_start_k]))
        context_k = context_k.replace(' ','')
        '''
        question [sep] context
        '''
        if len(question_k) + 4 + len(context_k)>= MAX_LEN:
            '''
            answer_start_k+len(answers_text_k)+x - (answer_start_k-x)+4+len(question_k) <=512
            2x = 512-len(answers_text_k) -1-len(question_k)
            '''
            x = (MAX_LEN-len(answers_text_k) - 4 -len(question_k))//2-1
            end = answer_start_k+len(answers_text_k)+x
            if answer_start_k-x <0:
                begain = 0
                idx = answer_start_k
            else:
                begain = answer_start_k-x
                idx = x
            context_k = context_k[begain:end]
        else:
            idx = answer_start_k

        chars = np.zeros((len(context_k)))
        chars[idx:idx+len(answers_text_k)]=1
        #这里以及避免了多答案问题
        enc_context = tokenizer.encode(context_k) 
        enc_question = tokenizer.encode(question_k) 
        offsets = [] 
        idx=0
        for t in enc_context[1:]:
            w = tokenizer.decode([t])
            if '#' in w and len(w)>1:
                w = w.replace('#','')
                # '##cm'
            if w == '[UNK]':
                #len('[UNK]')==5
                w = '。'
            offsets.append((idx,idx+len(w)))
            idx += len(w)
        toks = []
        for i,(a,b) in enumerate(offsets):
            sm = np.sum(chars[a:b])
            if sm>0: 
                toks.append(i) 
        input_ids[k,:len(enc_question)+len(enc_context)-1] = enc_question + enc_context[1:]
        attention_mask[k,:len(enc_question)+len(enc_context)-1] = 1
        if len(toks)>0:
            start_tokens[k,toks[0]+len(enc_question)] = 1
            end_tokens[k,toks[-1]+len(enc_question)] = 1
    return input_ids,attention_mask,start_tokens,end_tokens

def test_data_proceed(tokenizer,context_t,question_t,answers_text_t,answer_start_t,MAX_LEN):
    ct = len(context_t)
    input_ids_t = np.zeros((ct,MAX_LEN),dtype='int32')
    attention_mask_t = np.zeros((ct,MAX_LEN),dtype='int32')
    for k in range(len(context_t)):
        enc_context_t = tokenizer.encode(context_t) 
        enc_question_t = tokenizer.encode(question_t)
        if len(enc_question_t)+len(enc_context_t)-1 >MAX_LEN:
            x = enc_question_t + enc_context_t[1:]
            input_ids_t[k] = x[:MAX_LEN]
        else:
            input_ids_t[k,:len(enc_question_t)+len(enc_context_t)-1] = enc_question_t + enc_context_t[1:]
        attention_mask_t[k,:len(enc_question_t)+len(enc_context_t)-1] = 1
    return input_ids_t,attention_mask_t

# def random_int_list(start, stop, length):
#     start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
#     length = int(abs(length)) if length else 0
#     random_list = []
#     for i in range(length):
#         random_list.append(random.randint(start, stop))
#     return random_list

def build_model(pretrained_path,config,MAX_LEN):
    ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)   
    bert_model = TFBertModel.from_pretrained(pretrained_path,config=config,from_pt=True)
    x = bert_model(ids,attention_mask=att)
    
    x1 = tf.keras.layers.Dropout(0.1)(x[0]) 
    print(x1.shape)
    x1 = tf.keras.layers.Conv1D(1,1)(x1)
    '''
    (None, 96, 768)
    (None, 96, 1)
    769个参数相当于做了一次加权+b
    如果不指定该函数，将不会使用任何激活函数(即使用线性激活函数:a(x)=x)
    flatten后得到展开的768
    接一个softmax
    '''
    print(x1.shape)
    x1 = tf.keras.layers.Flatten()(x1)
    print(x1.shape)
    x1 = tf.keras.layers.Activation('softmax')(x1)
    
    x2 = tf.keras.layers.Dropout(0.1)(x[0]) 
    x2 = tf.keras.layers.Conv1D(1,1)(x2)
    x2 = tf.keras.layers.Flatten()(x2)
    x2 = tf.keras.layers.Activation('softmax')(x2)

    model = tf.keras.models.Model(inputs=[ids, att], outputs=[x1,x2])
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    return model


def build_model_2(pretrained_path,config,MAX_LEN):
    ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    config.output_hidden_states = True
    bert_model = TFBertModel.from_pretrained(pretrained_path,config=config,from_pt=True)
    
    x, _, hidden_states = bert_model(ids,attention_mask=att)
    layer_1 = hidden_states[-1]
    layer_2 = hidden_states[-2]
        
    x1 = tf.keras.layers.Dropout(0.1)(layer_1)
    x1 = tf.keras.layers.Conv1D(128, 2, padding='same')(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)
    x1 = tf.keras.layers.Conv1D(64, 2, padding='same')(x1)
    x1 = tf.keras.layers.Dense(1, dtype='float32')(x1)
    start_logits = tf.keras.layers.Flatten()(x1)
    start_logits = tf.keras.layers.Activation('softmax')(start_logits)

    x2 = tf.keras.layers.Dropout(0.1)(layer_2)
    x2 = tf.keras.layers.Conv1D(128, 2, padding='same')(x2)
    x2 = tf.keras.layers.LeakyReLU()(x2)
    x2 = tf.keras.layers.Conv1D(64, 2, padding='same')(x2)
    x2 = tf.keras.layers.Dense(1, dtype='float32')(x2)
    end_logits = tf.keras.layers.Flatten()(x2)
    end_logits = tf.keras.layers.Activation('softmax')(end_logits)
        
    model = tf.keras.models.Model(inputs=[ids, att], outputs=[start_logits,end_logits])
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    return model
    
def main():
    
    MAX_LEN=400
    
    pretrained_path = '/root/zhengyanzhao/comment/emotion/model/'
    config_path = os.path.join(pretrained_path, 'bert_config.json')
    vocab_path = os.path.join(pretrained_path, 'vocab.txt')
    config = BertConfig.from_json_file(config_path)
    tokenizer = BertTokenizer.from_pretrained(vocab_path)
    
    context,question,answers_text,answer_start = data_load('./train.json')
    input_ids,attention_mask,start_tokens,end_tokens = train_data_proceed(tokenizer,context,question,answers_text,answer_start,MAX_LEN)
    
    context_t,question_t,answers_text_t,answer_start_t = data_load('./dev.json')
    input_ids_t,attention_mask_t=test_data_proceed(tokenizer,context_t,question_t,answers_text_t,answer_start_t,MAX_LEN)

    
#     for i in random_int_list(0,len(context),10):
#     x_ = [tokenizer.decode([t]) for t in input_ids[i]]
#     token_ans = ''.join(x_[np.argmax(start_tokens[i]):np.argmax(end_tokens[i])+1])
#     print(token_ans+'  '+answers_text[i])   
#     model = build_model(pretrained_path,config,MAX_LEN)
#     model.fit([input_ids, attention_mask], [start_tokens, end_tokens], epochs=3, batch_size=16,verbose=2)
#           , callbacks=[sv],
#         validation_data=([input_ids[idxV,],attention_mask[idxV,],token_type_ids[idxV,]], 
#         [start_tokens[idxV,], end_tokens[idxV,]]))


#     jac = []
    VER='v0'
    DISPLAY=2 # USE display=1 FOR INTERACTIVE
    oof_start = np.zeros((input_ids.shape[0],MAX_LEN))
    oof_end = np.zeros((input_ids.shape[0],MAX_LEN))
    preds_start = np.zeros((input_ids_t.shape[0],MAX_LEN))
    preds_end = np.zeros((input_ids_t.shape[0],MAX_LEN))
    skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=777)
    for fold,(idxT,idxV) in enumerate(skf.split(input_ids,answer_start)):
        '''
        fold:当前K折折次
        idxT:当前训练集index
        idxV:当前测试集index
        len(idxV)*4=len(idxT)
        '''
        print('#'*25)
        print('### FOLD %i'%(fold+1))
        print('#'*25)

        K.clear_session()
        model = build_model(pretrained_path,config,MAX_LEN)

        sv = tf.keras.callbacks.ModelCheckpoint(
            './model_/%s-roberta-%i.h5'%(VER,fold), monitor='val_loss', verbose=2, save_best_only=True,
            save_weights_only=True, mode='auto', save_freq='epoch')

        model.fit([input_ids[idxT,], attention_mask[idxT,]],[start_tokens[idxT,],end_tokens[idxT,]], 
            epochs=3, batch_size=8, verbose=DISPLAY, callbacks=[sv],
            validation_data=([input_ids[idxV,],attention_mask[idxV,]], 
            [start_tokens[idxV,], end_tokens[idxV,]]))

        print('Loading model...')
        model.load_weights('./model_/%s-roberta-%i.h5'%(VER,fold))

#         print('Predicting OOF...')
#         oof_start[idxV,],oof_end[idxV,] = model.predict([input_ids[idxV,],attention_mask[idxV,],token_type_ids[idxV,]])

        print('Predicting Test...')
        preds = model.predict([input_ids_t,attention_mask_t],verbose=DISPLAY)
        preds_start += preds[0]/skf.n_splits
        preds_end += preds[1]/skf.n_splits
        '''
        5折验证结果进行投票，选出得分最高的点作为开始和结束点
        '''
#         all = []
#         for k in idxV:
#             a = np.argmax(oof_start[k,])
#             b = np.argmax(oof_end[k,])
#             if a>b: 
#                 st = train.loc[k,'text'] # IMPROVE CV/LB with better choice here
#             else:
#                 text1 = " "+" ".join(train.loc[k,'text'].split())
#                 enc = tokenizer.encode(text1)
#                 st = tokenizer.decode(enc[a-1:b])
#             all.append(jaccard(st,train.loc[k,'selected_text']))
#         jac.append(np.mean(all))
#         print('>>>> FOLD %i Jaccard ='%(fold+1),np.mean(all))
#         print()
    all_ = []
    for k in range(len(input_ids_t)):
        a = np.argmax(preds_start[k,])
        b = np.argmax(preds_end[k,])
        if a>b: 
            st = context_t[k]
        else:
            x_ = [tokenizer.decode([t]) for t in input_ids_t[k]]
            st = ''.join(x_[a:b+1])
        all_.append(st)

    ans_data = pd.DataFrame(context_t,columns=['context'])
    ans_data['question'] = question_t
    ans_data['answers_text'] = answers_text_t
    ans_data['pred_answers'] = all_
    ans_data.to_csv('result.csv')
                               
                               
if __name__ == '__main__':
    main()
