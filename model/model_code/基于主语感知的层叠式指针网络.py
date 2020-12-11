class LayerNormalization(tf.keras.layers.Layer):
    """(Conditional) Layer Normalization
    hidden_*系列参数仅为有条件输入时(conditional=True)使用
    """
    def __init__(
        self,
        center=True,
        scale=True,
        epsilon=None,
        conditional=False,
        hidden_units=None,
        hidden_activation='linear',
        hidden_initializer='glorot_uniform',
        **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        self.hidden_activation = activations.get(hidden_activation)
        self.hidden_initializer = initializers.get(hidden_initializer)
        self.epsilon = epsilon or 1e-12
    def compute_mask(self, inputs, mask=None):
        if self.conditional:
            masks = mask if mask is not None else []
            masks = [m[None] for m in masks if m is not None]
            if len(masks) == 0:
                return None
            else:
                return K.all(K.concatenate(masks, axis=0), axis=0)
        else:
            return mask
        
    def build(self, input_shape):
        super(LayerNormalization, self).build(input_shape)
        if self.conditional:
            shape = (input_shape[0][-1],)
        else:
            shape = (input_shape[-1],)
        if self.center:
            self.beta = self.add_weight(
                shape=shape, initializer='zeros', name='beta')
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape, initializer='ones', name='gamma')
        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = tf.keras.layers.Dense(
                    units=self.hidden_units,
                    activation=self.hidden_activation,
                    use_bias=False,
                    kernel_initializer=self.hidden_initializer)
            if self.center:
                self.beta_dense = tf.keras.layers.Dense(
                    units=shape[0], use_bias=False, kernel_initializer='zeros')
            if self.scale:
                self.gamma_dense = tf.keras.layers.Dense(
                    units=shape[0], use_bias=False, kernel_initializer='zeros')

    def call(self, inputs):
        """如果是条件Layer Norm，则默认以list为输入，第二个是condition
        """
        if self.conditional:
            inputs, cond = inputs
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)
            for _ in range(K.ndim(inputs) - K.ndim(cond)):
                cond = K.expand_dims(cond, 1)
            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma
        outputs = inputs
        if self.center:
            mean = K.mean(outputs, axis=-1, keepdims=True)
            outputs = outputs - mean
        if self.scale:
            variance = K.mean(K.square(outputs), axis=-1, keepdims=True)
            std = K.sqrt(variance + self.epsilon)
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta
        return outputs

def get_initializer(initializer_range: float = 0.02) -> tf.initializers.TruncatedNormal:
    """
    Creates a :obj:`tf.initializers.TruncatedNormal` with the given range.

    Args:
        initializer_range (`float`, defaults to 0.02): Standard deviation of the initializer range.

    Returns:
        :obj:`tf.initializers.TruncatedNormal`: The truncated normal initializer.
    """
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)

def shape_list(tensor: tf.Tensor) -> List[int]:
    """
    Deal with dynamic shape in tensorflow cleanly.

    Args:
        tensor (:obj:`tf.Tensor`): The tensor we want the shape of.

    Returns:
        :obj:`List[int]`: The shape of the tensor as a list.
    """
    static = tensor.shape.as_list()
    dynamic = tf.shape(tensor)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

class TFBertSelfAttention(tf.keras.layers.Layer):
    def __init__(self,hidden_size,num_attention_heads,**kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (self.hidden_size, self.num_attention_heads)
            )
        assert self.hidden_size % self.num_attention_heads == 0
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = tf.keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(0.02), name="query"
        )
        self.key = tf.keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(0.02), name="key"
        )
        self.value = tf.keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(0.02), name="value"
        )
        self.dropout = tf.keras.layers.Dropout(0.1)

    def transpose_for_scores(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_attention_heads, self.attention_head_size))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, hidden_states, attention_mask, head_mask, output_attentions, training=False):
        batch_size = shape_list(hidden_states)[0]
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = tf.matmul(
            query_layer, key_layer, transpose_b=True
        )  # (batch size, num_heads, seq_len_q, seq_len_k)
        dk = tf.cast(shape_list(key_layer)[-1], attention_scores.dtype)  # scale attention_scores
        attention_scores = attention_scores / tf.math.sqrt(dk)
        if attention_mask is not None:
            extended_attention_mask = tf.cast(attention_mask,tf.float32)
            extended_attention_mask = extended_attention_mask[:, tf.newaxis, tf.newaxis, :]
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            # Apply the attention mask is (precomputed for all layers in TFBertModel call() function)
            attention_scores = attention_scores + extended_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = tf.nn.softmax(attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs, training=training)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = tf.matmul(attention_probs, value_layer)
        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])
        context_layer = tf.reshape(
            context_layer, (batch_size, -1, self.all_head_size)
        )  # (batch_size, seq_len_q, all_head_size)
        outputs = (context_layer, attention_probs) if output_attentions else context_layer
        return outputs


def extract_subject(inputs):
    """根据subject_ids从output中取出subject的向量表征
    """
    output, subject_ids = inputs
    start = tf.gather(output,subject_ids[:,0],axis=1,batch_dims=0)
    end = tf.gather(output,subject_ids[:,1],axis=1,batch_dims=0)
    subject = tf.keras.layers.Concatenate(axis=2)([start, end])
    return subject[:,0]

def build_model(pretrained_path,config,MAX_LEN,p2id):
    ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    s_po_index =  tf.keras.layers.Input((2,), dtype=tf.int32)
    
    config.output_hidden_states = True
    bert_model = TFBertModel.from_pretrained(pretrained_path,config=config,from_pt=True)
    x, _, hidden_states = bert_model(ids,attention_mask=att)

    layer_1 = hidden_states[-1]
    
    start_logits = tf.keras.layers.Dense(1,activation = 'sigmoid')(layer_1)
    start_logits = tf.keras.layers.Lambda(lambda x: x**2)(start_logits)
    
    end_logits = tf.keras.layers.Dense(1,activation = 'sigmoid')(layer_1)
    end_logits = tf.keras.layers.Lambda(lambda x: x**2)(end_logits)
    
    subject_1 = extract_subject([layer_1,s_po_index])
    Normalization_1 = LayerNormalization(conditional=True)([layer_1, subject_1])
    
    position_emb_s = bert_model.bert.get_input_embeddings().position_embeddings(s_po_index[:,0])
    position_emb_e = bert_model.bert.get_input_embeddings().position_embeddings(s_po_index[:,1])
    position_embedding = position_emb_s + position_emb_e
    position_embedding = position_embedding[:,tf.newaxis,:]
    add_position = Normalization_1 + position_embedding
    
    self_attenion = TFBertSelfAttention(768,1)(add_position,att,head_mask=None,output_attentions=False)
    dense = tf.keras.layers.Dense(768,activation='relu')(self_attenion)
    dense = tf.keras.layers.Dropout(0.2)(dense)
    dense = tf.keras.layers.Dense(512,activation='relu')(dense)
    
    op_out_put_start = tf.keras.layers.Dense(len(p2id),activation = 'sigmoid')(dense)
    op_out_put_start = tf.keras.layers.Lambda(lambda x: x**4)(op_out_put_start)
    
    op_out_put_end = tf.keras.layers.Dense(len(p2id),activation = 'sigmoid')(dense)
    op_out_put_end = tf.keras.layers.Lambda(lambda x: x**4)(op_out_put_end)

    
    model = tf.keras.models.Model(inputs=[ids,att,s_po_index], outputs=[start_logits,end_logits,op_out_put_start,op_out_put_end])
    model_2 = tf.keras.models.Model(inputs=[ids,att], outputs=[start_logits,end_logits])
    model_3 = tf.keras.models.Model(inputs=[ids,att,s_po_index], outputs=[op_out_put_start,op_out_put_end])
    return model,model_2,model_3
