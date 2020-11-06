import tensorflow as tf

def sparse_categorical_crossentropy(y_true, y_pred):
    y_true = tf.reshape(y_true, tf.shape(y_pred)[:-1])
    y_true = tf.cast(y_true, tf.int32)
    y_true = tf.one_hot(y_true, K.shape(y_pred)[-1])
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

def loss_with_gradient_penalty(model,epsilon=1):
    def loss_with_gradient_penalty_2(y_true, y_pred):
        loss = tf.math.reduce_mean(sparse_categorical_crossentropy(y_true, y_pred))
        embeddings = model.variables[0]
        gp = tf.math.reduce_sum(tf.gradients(loss, [embeddings])[0].values**2)
        return loss + 0.5 * epsilon * gp
    return loss_with_gradient_penalty_2
