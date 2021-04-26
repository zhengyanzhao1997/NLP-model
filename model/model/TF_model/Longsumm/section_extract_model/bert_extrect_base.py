import json
import numpy as np
import tensorflow.keras.backend as K
from typing import List
import transformers.models.bert.modeling_tf_bert as tf_bert
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel,BertConfig
from sklearn.metrics import f1_score
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

def load_data(filename):
    """加载数据
    返回：[texts]
    """
    D = []
    with open(filename) as f:
        for l in f:
            texts = json.loads(l)
            D.append(texts)
    return D

def preceed_data(data,Max_len,sentence_max,tokenizer,train=False):
    #data = [i for i in data if len(i['text_list']) > 0 and len(i['label']) > 0]
    data_len = len(data)
    ids = np.zeros((data_len, Max_len), dtype='int32')
    mask_att = np.zeros((data_len, Max_len), dtype='int32')
    token_type = np.zeros((data_len, Max_len), dtype='int32')
    # sentence_index = np.zeros((data_len, sentence_max), dtype='int32')
    sentence_mask = np.zeros((data_len, Max_len), dtype='int32')
    sentence_label = np.zeros((data_len,Max_len), dtype='int32')
    for index,sigle_data in enumerate(data):
        text_list = sigle_data['text_list']
        label_list = sigle_data['label']
        token_id = []
        label_index = []
        token_type_ids = []
        for text_ in text_list:
            token = tokenizer.encode(text_)
            if len(token_id) + len(token) >= Max_len or len(label_index) >= sentence_max:
                break
            token_id += token
            label_index.append(len(token_id) - len(token))
            token_type_id = (len(label_index) + 1)%2
            token_type_ids.extend([token_type_id]*len(token))
        ids[index][:len(token_id)] = token_id
        mask_att[index][:len(token_id)] = 1
        token_type[index][:len(token_id)] = token_type_ids
        # sentence_index[index][:len(label_index)] = label_index
        for m,sentece_idx in enumerate(label_index):
            sentence_mask[index][sentece_idx] = 1
            if m in label_list:
                sentence_label[index][sentece_idx] = 1
    if  train:
        return {'ids':ids,'att':mask_att,'token_type':token_type,'sentence_mask':sentence_mask,'sentence_label':sentence_label}
    return ({"ids":ids,'att':mask_att,'token_type':token_type,'sentence_mask':sentence_mask},sentence_label)

def shape_list(tensor: tf.Tensor) -> List[int]:
    dynamic = tf.shape(tensor)
    if tensor.shape == tf.TensorShape(None):
        return dynamic
    static = tensor.shape.as_list()
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def expand_mask_axis(mask):
    mask_new = tf.cast(mask, tf.float32)
    extended_attention_mask = mask_new[:, tf.newaxis, tf.newaxis, :]
    return extended_attention_mask

class SinusoidalPositionalEmbedding(tf.keras.layers.Embedding):
    def __init__(self, num_positions: int, embedding_dim: int, **kwargs):
        if embedding_dim % 2 != 0:
            raise NotImplementedError(f"odd embedding_dim {embedding_dim} not supported")
        super().__init__(
            num_positions,
            embedding_dim,
            **kwargs,
        )

    def build(self, input_shape: tf.TensorShape):
        super().build(input_shape)  # Instantiates self.weight so it can be loaded
        weight: np.ndarray = self._init_weight(self.input_dim, self.output_dim)
        self.set_weights([weight])  # overwrite self.weight to correct value

    @staticmethod
    def _init_weight(n_pos: int, dim: int):
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        # index 0 is all zero
        position_enc[:, 0: dim // 2] = np.sin(position_enc[:, 0::2])
        position_enc[:, dim // 2:] = np.cos(position_enc[:, 1::2])
        # convert to tensor
        table = tf.convert_to_tensor(position_enc, dtype=tf.float32)
        tf.stop_gradient(table)
        return table

    def call(self, input_shape: tf.TensorShape, past_key_values_length: int = 0):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input_shape[:2]

        positions = tf.range(
            past_key_values_length, seq_len + past_key_values_length, delta=1, dtype=tf.int32, name="range"
        )
        return super().call(positions)


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
    def compute_loss(self, inputs, mask=None):
        y_true, y_mask, y_pred = inputs
        if tf.shape(y_pred)[-1] == 1:
            y_pred = tf.squeeze(y_pred)
        y_true = tf.cast(y_true, tf.float32)
        y_mask = tf.cast(y_mask, tf.float32)
        loss = K.binary_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss

def build_model(pretrained_path, bert_cofig, token_max, sentence_max):
    ids = tf.keras.layers.Input((token_max,), dtype=tf.int32, name='ids')
    att = tf.keras.layers.Input((token_max,), dtype=tf.int32, name='att')
    token_type = tf.keras.layers.Input((token_max,), dtype=tf.int32, name='token_type')
    sentence_mask = tf.keras.layers.Input((token_max,), dtype=tf.int32, name='sentence_mask')
    sentence_label = tf.keras.layers.Input((token_max,), dtype=tf.int32, name='sentence_label')

    bert_model = TFBertModel.from_pretrained(pretrained_path, config=bert_cofig)

    out_put_state = bert_model({'input_ids': ids, 'attention_mask': att, 'token_type_ids': token_type})
    out_put_state = out_put_state[0]

    # sentence_embed = tf.gather(out_put_state, sentence_index, axis=1, batch_dims=0)[:, 0]
    # embed_positions = SinusoidalPositionalEmbedding(sentence_max, bert_cofig.hidden_size)
    # input_shape = shape_list(sentence_embed)
    # embed_pos = embed_positions(input_shape)

    # input_trans = sentence_embed + embed_pos
    # drop_out = tf.keras.layers.Dropout(0.1)(input_trans)
    # dense_output = tf.keras.layers.Dense(bert_cofig.hidden_size, activation='relu')(drop_out)

    # bert_layer_1 = tf_bert.TFBertLayer(bert_cofig)
    # bert_layer_2 = tf_bert.TFBertLayer(bert_cofig)
    # sentence_mask_float = expand_mask_axis(sentence_mask)

    # transform_1 = bert_layer_1(dense_output, sentence_mask_float, None, None)[0]
    # transform_2 = bert_layer_2(transform_1, sentence_mask_float, None, None)[0]

    out_put_1 = tf.keras.layers.Dense(1, activation='sigmoid', name='out_put_1')(out_put_state)
    out_put_2 = CrossEntropy(2)([sentence_label, sentence_mask, out_put_1])

    model_train = tf.keras.models.Model(inputs=[ids, att, token_type, sentence_mask, sentence_label],
                                        outputs=out_put_2)
    model_evaluate = tf.keras.models.Model(inputs=[ids, att, token_type, sentence_mask],
                                           outputs=out_put_1)
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    model_train.compile(optimizer=optimizer)
    return model_train, model_evaluate

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

def evaluate(model, data, threshold=0.2):
    data_x = data[0]
    mask = data_x['sentence_mask']
    label =  data[1]
    evaluate = 0
    pred = model.predict(data_x)[:, :, 0]
    count = 0
    for a,s,yp in zip(label,mask,pred):
        yp = yp[np.where(s==1)[0]]
        a = a[np.where(s==1)[0]]
        yp = np.array(yp > threshold + 0.1,dtype=np.int)
        if np.sum(a) == 0:
            continue
        count+=1
        f1 = f1_score(yp, a, average='binary',zero_division=1)
        evaluate += f1
    return evaluate / count

class Evaluator(tf.keras.callbacks.Callback):
    """训练回调
    """
    def __init__(self,model_evaluate,threshold,valid_data,fold,Max_len,sentence_max,tokenizer):
        self.best_metric = 0.0
        self.threshold = threshold
        self.valid_data = preceed_data(valid_data,Max_len,sentence_max,tokenizer)
        self.fold = fold
        self.model_evaluate = model_evaluate

    def on_epoch_end(self, epoch, logs=None):
        eva = evaluate(self.model_evaluate,self.valid_data,self.threshold + 0.1)
        if  eva >= self.best_metric:  # 保存最优
            self.best_metric = eva
            self.model_evaluate.save_weights('./model_save/extract_model_%s.hdf5' % self.fold)
            print('eva raise to %s'%eva)
        else:
            print('eva is %s,not raise'%eva)

class WarmupExponentialDecay(tf.keras.callbacks.Callback):
    def __init__(self,lr_base=4e-3,decay=0,warmup_step=10000,a=-0.5,b=-1.5):
        self.warmup_step=warmup_step  
        self.lr=lr_base #learning_rate_base
        self.a = a
        self.b = b
        self.steps_passed = 1 #也是一个计数器
    def on_batch_begin(self, batch, logs=None):
        # params是模型自动传递给Callback的一些参数
        updata_lr = self.lr * min(self.steps_passed**self.a,self.steps_passed*self.warmup_step**self.b)
        K.set_value(self.model.optimizer.lr,updata_lr)
        self.steps_passed += 1
    def on_epoch_begin(self,epoch,logs=None):
        print("learning_rate:",K.get_value(self.model.optimizer.lr)) 

def main():
    data = load_data('./textlist_label.json')
    #data = [i for i in data if len(i['text_list']) > 0 and len(i['label']) > 0]
    data = [i for i in data if len(i['text_list']) > 0 ]
    print(len(data))
    token_max = 1024
    sentence_max = 80
    batch_size = 8
    epochs = 25
    num_folds = 10
    threshold = 0.2
    pretrain_path = "./bert_unces/tf_model.h5"
    tokenizer = BertTokenizer.from_pretrained('./bert_unces/')
    config = BertConfig.from_pretrained('./bert_unces/config.json')
    config.hierarchical = True
    for fold in range(num_folds):
        K.clear_session()
        #strategy = tf.distribute.MirroredStrategy()
        #print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        #with strategy.scope():
        model_train, model_evaluate = build_model(pretrain_path, config, token_max, sentence_max)

        train_data = data_split(data, fold, num_folds, 'train')
        valid_data = data_split(data, fold, num_folds, 'valid')
        print(len(train_data))
        print(len(valid_data))
        steps_per_epoch = len(train_data) // batch_size + 1

        train_data = preceed_data(train_data, token_max, sentence_max, tokenizer, train=True)
        
        train_data = tf.data.Dataset.from_tensor_slices(train_data)
        train_data = train_data.batch(batch_size).shuffle(10086).repeat()
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        train_data = train_data.with_options(options)
        
        evaluator_ = Evaluator(model_evaluate, threshold, valid_data, fold, token_max, sentence_max, tokenizer)
        warm_up = WarmupExponentialDecay()
        model_train.fit(train_data, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=[evaluator_])

if __name__ == "__main__":
    main()

