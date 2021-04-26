import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tqdm import tqdm
from rouge import Rouge
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

metric_keys = ['main', 'rouge-1', 'rouge-2', 'rouge-l']

# 计算rouge用
rouge = Rouge()

def compute_rouge(source, target):
    #source, target = ' '.join(source), ' '.join(target)
    try:
        scores = rouge.get_scores(hyps=source, refs=target)
        return {
            'rouge-1': scores[0]['rouge-1']['f'],
            'rouge-2': scores[0]['rouge-2']['f'],
            'rouge-l': scores[0]['rouge-l']['f'],
        }
    except ValueError:
        return {
            'rouge-1': 0.0,
            'rouge-2': 0.0,
            'rouge-l': 0.0,
        }

def compute_main_metric(source, target):
    metrics = compute_rouge(source, target)
    metrics['main'] = (
            metrics['rouge-1'] * 0.2 + metrics['rouge-2'] * 0.4 +
            metrics['rouge-l'] * 0.4
    )
    return metrics['main']

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
    x = ResidualGatedConv1D(hidden_size, 3, dilation_rate=16)(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = ResidualGatedConv1D(hidden_size, 3, dilation_rate=1)(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = ResidualGatedConv1D(hidden_size, 3, dilation_rate=1)(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    out_put = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=input_, outputs=out_put)
    return model

def evaluate(data,pred,threshold=0.2):
    evaluater = 0
    data_result = []
    for d,yp in tqdm(zip(data,pred),desc='evaluating'):
        a = d[0]
        s = d[2]
        if len(a)>400:
            a = a[:400]
        ex_result = [[a[i],yp[i]] for i in range(len(a))]
        #yp = np.where(yp > threshold)[0]
        #pred_sum = ' '.join([d[0][i] for i in yp])
        data_result.append({'extract':ex_result,'summary':s})
    return data_result

def load_data(filename):
    """加载数据
    返回：[(texts, labels, summary)]
    """
    D = []
    with open(filename) as f:
        for l in f:
            D.append(json.loads(l))
    return D

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def main():
    threshold = 0.2
    fold = 15
    max_len = 400
    input_size = 1024
    hidden_size = 512
    
    data = load_data('./pre_train_summary/out_tain_select.json')
    
    for sp in range(3):
        if sp == 2:
            data_sp = data[sp*5000:]
        else:
            data_sp = data[sp*5000:(sp+1)*5000]
        data_x = np.load('./pre_train_summary/arciv_out_train_%s.npy'%sp)
        data_x = data_x[:,:400]
        pred_result = np.zeros((len(data_x),max_len))
        
        for f in range(fold):
            print(f)
            K.clear_session()
            model = bulid_extract_model(max_len, input_size, hidden_size)
            model.load_weights('/home_zyz/extract_model/model_add_06/extract_model_%s.hdf5' % f)
            pred = model.predict(data_x)[:,:,0]
            pred_result += pred
        pred_result = pred_result/fold
        data_result = evaluate(data_sp,pred_result)

        with open('./pre_train_summary/arciv_out_train_key_select06_%s.json'%sp, 'w', encoding='utf-8') as f:
            for d in data_result:
                f.write(json.dumps(d,ensure_ascii=False,cls=NpEncoder) + '\n')

if __name__ == "__main__":
    main()
