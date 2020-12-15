class Biaffine_2(tf.keras.layers.Layer):
    def __init__(self, in_size, out_size,MAX_LEN):
        super(Biaffine_2, self).__init__()
        self.w1 = self.add_weight(
            name='weight1', 
            shape=(in_size, out_size, in_size),
            trainable=True)
        self.w2 = self.add_weight(
            name='weight2', 
            shape=(2*in_size + 1, out_size),
            trainable=True)
        self.MAX_LEN = MAX_LEN
        
    def call(self, input1, input2):
        f1 = tf.expand_dims(input1,2)
        f2 = tf.expand_dims(input2,1)
        f1 = tf.tile(f1,multiples=(1,1,self.MAX_LEN,1))
        f2 = tf.tile(f2,multiples=(1,self.MAX_LEN,1,1))
        concat_f1f2 = tf.concat((f1,f2),axis=-1)
        concat_f1f2 = tf.concat((concat_f1f2,tf.ones_like(concat_f1f2[..., :1])), axis=-1)
        # bxi,oij,byj->boxy
        logits_1 = tf.einsum('bxi,ioj,byj->bxyo', input1, self.w1, input2)
        logits_2 = tf.einsum('bijy,yo->bijo',concat_f1f2,self.w2)
        return logits_1+logits_2 

def build_model_3(pretrained_path,config,MAX_LEN,Cs_num,cs_em_size,R_num):
    ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    cs = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    
    config.output_hidden_states = True
    bert_model = TFBertModel.from_pretrained(pretrained_path,config=config,from_pt=True)
    x, _, hidden_states = bert_model(ids,attention_mask=att)
    layer_1 = hidden_states[-1]
    layer_2 = hidden_states[-2]
    
    
    start_logits = tf.keras.layers.Dense(256,activation = 'relu')(layer_1)
    start_logits = tf.keras.layers.Dense(Cs_num,activation = 'sigmoid')(start_logits)
    start_logits = tf.keras.layers.Lambda(lambda x: x**2,name='s_start')(start_logits)
    
    end_logits = tf.keras.layers.Dense(256,activation = 'relu')(layer_1)
    end_logits = tf.keras.layers.Dense(Cs_num,activation = 'sigmoid')(end_logits)
    end_logits = tf.keras.layers.Lambda(lambda x: x**2,name='s_end')(end_logits)
    
    cs_emb = tf.keras.layers.Embedding(Cs_num,cs_em_size)(cs)
    
    concat_cs = tf.keras.layers.Concatenate(axis=-1)([layer_1,layer_2])
    
    f1 = tf.keras.layers.Dropout(0.2)(concat_cs)
    f1 = tf.keras.layers.Dense(256,activation='relu')(f1)
    f1 = tf.keras.layers.Dense(128,activation='relu')(f1)
    f1 = tf.keras.layers.Concatenate(axis=-1)([f1,cs_emb])
    
    f2 = tf.keras.layers.Dropout(0.2)(concat_cs)
    f2 = tf.keras.layers.Dense(256,activation='relu')(f2)
    f2 = tf.keras.layers.Dense(128,activation='relu')(f2)
    f2 = tf.keras.layers.Concatenate(axis=-1)([f2,cs_emb])
    
    Biaffine_layer = Biaffine_2(128+cs_em_size,R_num,MAX_LEN)
    output_logist = Biaffine_layer(f1,f2)
    output_logist = tf.keras.layers.Activation('sigmoid')(output_logist)
    output_logist = tf.keras.layers.Lambda(lambda x: x**4,name='relation')(output_logist)
    
    model = tf.keras.models.Model(inputs=[ids,att,cs], outputs=[start_logits,end_logits,output_logist])
    model_2 = tf.keras.models.Model(inputs=[ids,att], outputs=[start_logits,end_logits])
    model_3 = tf.keras.models.Model(inputs=[ids,att,cs], outputs=[output_logist])
    return model,model_2,model_3
