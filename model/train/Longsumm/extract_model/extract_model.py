import json
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from transformers import *
import tensorflow.keras.backend as K
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

def load_data(filename):
    """加载数据
    返回：[texts]
    """
    D = []
    with open(filename) as f:
        for l in f:
            texts = json.loads(l)[0]
            D.append(texts)
    return D

def sequence_padding(inputs, length=None, padding=0, mode='post'):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = max([len(x) for x in inputs])

    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        x = x[:length]
        if mode == 'post':
            pad_width[0] = (0, length - len(x))
        elif mode == 'pre':
            pad_width[0] = (length - len(x), 0)
        else:
            raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)
    return np.array(outputs)

def main():
    #pagesus_pretrain_path = '/home_zyz/bert_extract/goole_pagesus/'
    #psus_config = PegasusConfig.from_pretrained('/home_zyz/bert_extract/goole_pagesus/config.json')
    #tokenizer = PegasusTokenizer.from_pretrained('/home_zyz/bert_extract/goole_pagesus/')
    #psus_config.encoder_attention_type = 'full'
    #bert_model = TFPegasusForConditionalGeneration.from_pretrained(pagesus_pretrain_path,config=psus_config,from_pt=True)
    #encoder = bert_model.get_encoder()
    
    pretrained_path = '/home_zyz/extract_model/robertalarge/'
    config_path = os.path.join(pretrained_path, 'config.json')
    tokenizer = RobertaTokenizer(vocab_file=pretrained_path + 'vocab.json',
                                 merges_file=pretrained_path + 'merges.txt',
                                 lowercase=True, add_prefix_space=True)
    config = BertConfig.from_json_file(config_path)
    config.output_hidden_states = True
    bert_model = TFRobertaModel.from_pretrained(pretrained_path, config=config)

    texts = load_data('/home_zyz/abstract_generate/union_abstract_data_extract_2.json')
    average_pooling = tf.keras.layers.GlobalAveragePooling1D()

    def conver_token(texts):
        token_ = []
        for text in tqdm(texts, desc='转换向量'):
            print(len(text))
            if len(text) > 400:
                text = text[:400]
            token = tokenizer(text, max_length=256,truncation=True,padding=True, return_tensors="tf")
            vecotor = bert_model(token)[0]
            pooling = average_pooling(vecotor, mask=token['attention_mask'])
            token_.append(pooling)
        return token_

    pooling = conver_token(texts)
    b = sequence_padding(pooling,length=400)
    np.save('/home_zyz/abstract_generate/union_abstract_data_extract_npy', b)

if __name__=='__main__':
    main()
