from transformers import *
import tensorflow as tf

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
