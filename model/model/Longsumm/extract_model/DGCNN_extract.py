import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tqdm import tqdm
from rouge import Rouge
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['acc'])
    return model

def evaluate(model,data,data_x,threshold=0.2):
    evaluater = 0
    pred = model.predict(data_x)[:,:,0]
    # [sample_num,256]
    for d,yp in tqdm(zip(data,pred),desc='evaluating'):
        yp = yp[:len(d[0])]
        yp = np.where(yp > threshold)[0]
        pred_sum = ' '.join([d[0][i] for i in yp])
        evaluater += compute_main_metric(pred_sum,d[2])
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
            self.model.save_weights('model_add_07/extract_model_%s.hdf5' % self.fold)
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
    input_size = 1024
    hidden_size = 512
    epochs = 30
    batch_size = 32
    threshold = 0.2
    num_folds = 15
    max_len = 400
    data = load_data('/home/zyz_temp/update_exteact_data_noabstractsc_afex.json')
    data_x = np.load('/home/zyz_temp/new_cs_afex.npy')
    data_y = np.zeros_like(data_x[..., :1])
    for i, d in enumerate(data):
        for j in d[1]:
            data_y[i][j][0] = 1
    data_add = load_data('/home/zyz_temp/arciv_union_add_exarciv_extract_in_train.json')
    data_add_x = np.load('/home/zyz_temp/arciv_in_train.npy')
    data_add_y  = np.zeros_like(data_add_x[..., :1])
    valid_index = []
    for i, d in enumerate(data_add):
        for j in d[1]:
            if j < len(data_add_y[i]):
                data_add_y[i][j][0] = 1
        if sum(data_add_y[i]) > 0:
            valid_index.append(i)
    data_add_x = data_add_x[valid_index]
    data_add_y = data_add_y[valid_index]
    print(len(data_add_x))
    for fold in range(num_folds):
        valid_data = data_split(data, fold, num_folds, 'valid')
        train_x = data_split(data_x, fold, num_folds, 'train')
        valid_x = data_split(data_x, fold, num_folds, 'valid')
        train_y = data_split(data_y, fold, num_folds, 'train')
        # 启动训练
        train_x_add = np.concatenate((train_x,data_add_x),axis=0)
        train_y_add = np.concatenate((train_y,data_add_y),axis=0)
        permutation = np.random.permutation(len(train_x_add))        # 利用np.random.permutaion函数，获得打乱后的行数，输出permutation
        train_x_add = train_x_add[permutation]                                       # 得到打乱后数据a
        train_y_add = train_y_add[permutation] 
        
        K.clear_session()
        model = bulid_extract_model(max_len, input_size, hidden_size)
        evaluator = Evaluator(threshold, valid_data, valid_x, fold)
        model.fit(
            train_x_add,
            train_y_add,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[evaluator]
        )

if __name__ == "__main__":
    main()
