def build_model(pretrained_path,config,MAX_LEN,Cs_num,cs_em_size,R_num):
    ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    cs = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    
#     R_embedding = tf.keras.layers.Embedding(R_num,R_em_size)
    
    config.output_hidden_states = True
    bert_model = TFBertModel.from_pretrained(pretrained_path,config=config,from_pt=True)
    x, _, hidden_states = bert_model(ids,attention_mask=att)
    layer_1 = hidden_states[-1]
    
    start_logits = tf.keras.layers.Dense(Cs_num,activation = 'sigmoid')(layer_1)
    start_logits = tf.keras.layers.Lambda(lambda x: x**2,name='s_start')(start_logits)
    
    end_logits = tf.keras.layers.Dense(Cs_num,activation = 'sigmoid')(layer_1)
    end_logits = tf.keras.layers.Lambda(lambda x: x**2,name='s_end')(end_logits)
    
    cs_emb = tf.keras.layers.Embedding(Cs_num,cs_em_size)(cs)
    concat_cs = tf.keras.layers.Concatenate(axis=-1)([layer_1,cs_emb])
    
    f1 = tf.keras.layers.Dense(128)(concat_cs)
    f2 = tf.keras.layers.Dense(128)(concat_cs)
    
    f1 = tf.expand_dims(f1,1)
    f2 = tf.expand_dims(f2,2)

    f1 = tf.tile(f1,multiples=(1,MAX_LEN,1,1))
    f2 = tf.tile(f2,multiples=(1,1,MAX_LEN,1))
    
    concat_f = tf.keras.layers.Concatenate(axis=-1)([f1,f2])
    output_logist = tf.keras.layers.Dense(128,activation='relu')(concat_f)
    output_logist = tf.keras.layers.Dense(R_num,activation='sigmoid')(output_logist)
    output_logist = tf.keras.layers.Lambda(lambda x: x**4,name='relation')(output_logist)

    model = tf.keras.models.Model(inputs=[ids,att,cs], outputs=[start_logits,end_logits,output_logist])
    model_2 = tf.keras.models.Model(inputs=[ids,att], outputs=[start_logits,end_logits])
    model_3 = tf.keras.models.Model(inputs=[ids,att,cs], outputs=[output_logist])
    return model,model_2,model_3
