from transformers import PegasusTokenizer,PegasusConfig,TFPegasusForConditionalGeneration
import tensorflow.keras.backend as K
from typing import List
import transformers.models.bert.modeling_tf_bert as tf_bert
from transformers import BertConfig
from tool import *
from tqdm import tqdm
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0,4,5,6"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

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

def shape_list(tensor: tf.Tensor) -> List[int]:
    dynamic = tf.shape(tensor)
    if tensor.shape == tf.TensorShape(None):
        return dynamic
    static = tensor.shape.as_list()
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

class CrossEntropy(Loss):
    def compute_loss(self, inputs, mask=None):
        y_true, y_mask, y_pred = inputs
        if tf.shape(y_pred)[-1] == 1:
            y_pred = tf.squeeze(y_pred)
        y_true = tf.cast(y_true,tf.float32)
        y_mask = tf.cast(y_mask,tf.float32)
        loss = K.binary_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss

def build_model(pretrained_path, psus_config, bert_cofig, token_max, sentence_max):
    ids = tf.keras.layers.Input((token_max,), dtype=tf.int32,name='ids')
    att = tf.keras.layers.Input((token_max,), dtype=tf.int32,name='att')
    sentence_index = tf.keras.layers.Input((sentence_max,), dtype=tf.int32,name='sentence_index')
    sentence_mask = tf.keras.layers.Input((sentence_max,), dtype=tf.int32,name='sentence_mask')
    sentence_label  = tf.keras.layers.Input((sentence_max,), dtype=tf.int32,name='sentence_label')

    psus_config.max_position_embeddings = token_max
    psus_config.encoder_attention_type = 'spare'
    # bert_cofig.hidden_size = psus_config.d_model
    # bert_cofig.num_attention_heads = psus_config.encoder_attention_heads
    bert_model = TFPegasusForConditionalGeneration.from_pretrained(pretrained_path, config=psus_config, from_pt=True)
    encoder = bert_model.get_encoder()

    out_put_state = encoder({'input_ids': ids, 'attention_mask': att})
    out_put_state = out_put_state[0]
    sentence_embed = tf.gather(out_put_state, sentence_index, axis=1, batch_dims=0)[:, 0]
    embed_positions = SinusoidalPositionalEmbedding(sentence_max, psus_config.d_model)
    input_shape = shape_list(sentence_embed)
    embed_pos = embed_positions(input_shape)

    input_trans = sentence_embed + embed_pos
    drop_out = tf.keras.layers.Dropout(0.1)(input_trans)
    dense_output = tf.keras.layers.Dense(bert_cofig.hidden_size,activation='relu')(drop_out)

    bert_layer_1 = tf_bert.TFBertLayer(bert_cofig)
    bert_layer_2 = tf_bert.TFBertLayer(bert_cofig)
    sentence_mask_float = expand_mask_axis(sentence_mask)

    transform_1 = bert_layer_1(dense_output, sentence_mask_float, None, None)[0]
    transform_2 = bert_layer_2(transform_1, sentence_mask_float, None, None)[0]

    out_put_1 = tf.keras.layers.Dense(1, activation='sigmoid',name='out_put_1')(transform_2)
    out_put_2 = CrossEntropy(2)([sentence_label,sentence_mask,out_put_1])
    
    model_train = tf.keras.models.Model(inputs=[ids, att, sentence_index, sentence_mask, sentence_label],outputs = out_put_2)
    model_evaluate = tf.keras.models.Model(inputs=[ids, att, sentence_index, sentence_mask],outputs = out_put_1)
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    model_train.compile(optimizer=optimizer)
    return model_train,model_evaluate

def load_data(path):
    data2 = []
    with open(path, encoding='utf-8') as f:
        for l in f:
            m = json.loads(l)
            data2.append(m)
    return data2

class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=True):
        ids = np.zeros((self.batch_size,self.Max_len),dtype='int32')
        mask_att = np.zeros((self.batch_size,self.Max_len),dtype='int32')
        sentence_index = np.zeros((self.batch_size,self.sentence_max), dtype='int32')
        sentence_mask = np.zeros((self.batch_size,self.sentence_max), dtype='int32')
        sentence_label = np.zeros((self.batch_size,self.sentence_max), dtype='int32')
        index = 0
        for is_end, sigle_data in self.sample(random):
            text_list = sigle_data[0]
            label_list = sigle_data[1]
            token_id = []
            label_index = []
            for text_ in text_list:
                token = self.tokenizer.encode(text_)
                if len(token_id) + len(token) >= self.Max_len or len(label_index) >= self.sentence_max:
                    break
                token_id += token
                label_index.append(len(token_id)-1)
            ids[index][:len(token_id)] = token_id
            mask_att[index][:len(token_id)] = 1
            sentence_index[index][:len(label_index)] = label_index
            sentence_mask[index][:len(label_index)] = 1
            for label_ in label_list:
                if label_ <= len(label_index) - 1:
                    sentence_label[index][label_] = 1
            index += 1
            if  index == self.batch_size or is_end:
                yield {'ids':ids,'att':mask_att,'sentence_index':sentence_index,
                    'sentence_mask':sentence_mask,'sentence_label':sentence_label}
                ids = np.zeros((self.batch_size,self.Max_len),dtype='int32')
                mask_att = np.zeros((self.batch_size,self.Max_len),dtype='int32')
                sentence_index = np.zeros((self.batch_size,self.sentence_max,), dtype='int32')
                sentence_mask = np.zeros((self.batch_size,self.sentence_max,), dtype='int32')
                sentence_label = np.zeros((self.batch_size,self.sentence_max,), dtype='int32')
                index = 0

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

def preceed_vaild_data(data,Max_len,sentence_max,tokenizer):
    data_len = len(data)
    ids = np.zeros((data_len, Max_len), dtype='int32')
    mask_att = np.zeros((data_len, Max_len), dtype='int32')
    sentence_index = np.zeros((data_len, sentence_max), dtype='int32')
    sentence_mask = np.zeros((data_len, sentence_max), dtype='int32')
    for index,sigle_data in enumerate(data):
        text_list = sigle_data[0]
        token_id = []
        label_index = []
        for text_ in text_list:
            token = tokenizer.encode(text_)
            if len(token_id) + len(token) >= Max_len or len(label_index) >= sentence_max:
                break
            token_id += token
            label_index.append(len(token_id) - 1)
        ids[index][:len(token_id)] = token_id
        mask_att[index][:len(token_id)] = 1
        sentence_index[index][:len(label_index)] = label_index
        sentence_mask[index][:len(label_index)] = 1
        # for label_ in label_list:
        #     if label_ <= len(label_index) - 1:
        #         sentence_label[index][label_] = 1
    summary = [x[2] for x in data]
    artical = [x[0] for x in data]
    return ({"ids":ids,'att':mask_att,'sentence_index':sentence_index,'sentence_mask':sentence_mask},artical,summary)

def evaluate(model,data,threshold=0.2):
    
    data_x = data[0]
    artical = data[1]
    summary = data[2]
    evaluater = 0  

    #data_x = tf.data.Dataset.from_tensor_slices(({'ids':data_x[0],'att':data_x[1],'sentence_index':data_x[2],'sentence_mask':data_x[3]}))
    #options = tf.data.Options()
    #options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    #data_x = data_x.with_options(options)
    
    pred = model.predict(data_x)[:,:,0]
    for a,s,yp in zip(artical,summary,pred):
        yp = yp[:len(a)]
        yp = np.where(yp > threshold)[0]
        pred_sum = ' '.join([a[i] for i in yp])
        evaluater += compute_main_metric(pred_sum,s)
    return evaluater/len(data_x)

class Evaluator(tf.keras.callbacks.Callback):
    """训练回调
    """
    def __init__(self,model_evaluate,threshold,valid_data,fold,Max_len,sentence_max,tokenizer):
        self.best_metric = 0.0
        self.threshold = threshold
        self.valid_data = preceed_vaild_data(valid_data,Max_len,sentence_max,tokenizer)
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

def main():
    batch_size = 4
    Max_len = 8192
    sentence_max = 512
    epochs = 15
    num_folds = 15
    threshold = 0.2

    pagesus_pretrain_path = './goole_pagesus/'
    psus_config = PegasusConfig.from_pretrained('./goole_pagesus/config.json')
    bert_cofig = BertConfig.from_pretrained('./config-roberta-base.json')
    tokenizer = PegasusTokenizer.from_pretrained(pagesus_pretrain_path)

    data = load_data('./subtract_extract_2.json')

    for fold in range(num_folds):
        K.clear_session()
        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        with strategy.scope():
            model_train,model_evaluate = build_model(pretrained_path=pagesus_pretrain_path, psus_config=psus_config, bert_cofig=bert_cofig,
                            token_max=Max_len,
                            sentence_max=sentence_max)

        train_data = data_split(data, fold, num_folds, 'train')
        valid_data = data_split(data, fold, num_folds, 'valid')

        evaluator_ = Evaluator(model_evaluate,threshold, valid_data, fold,Max_len,sentence_max,tokenizer)
        train_generator = data_generator(train_data, batch_size, Max_len, sentence_max, tokenizer)
        model_train.fit(train_generator.forfit(), steps_per_epoch=len(train_generator), epochs=epochs,callbacks=[evaluator_])

if __name__ == "__main__":
    main()


