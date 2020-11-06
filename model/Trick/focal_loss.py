from tensorflow.python.ops import array_ops
import tensorflow as tf

def binary_focal_loss(target_tensor,prediction_tensor, alpha=0.25, gamma=2):
    zeros = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)
    target_tensor = tf.cast(target_tensor,prediction_tensor.dtype)
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - prediction_tensor, zeros)
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, prediction_tensor)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.math.log(tf.clip_by_value(prediction_tensor, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.math.log(tf.clip_by_value(1.0 - prediction_tensor, 1e-8, 1.0))
    return tf.math.reduce_sum(per_entry_cross_ent)
    
def softmax_focal_loss(label,pred,class_num=6, gamma=2):
    label = tf.squeeze(tf.cast(tf.one_hot(tf.cast(label,tf.int32),class_num),pred.dtype)) 
    pred = tf.clip_by_value(pred, 1e-8, 1.0)
    w1 = tf.math.pow((1.0-pred),gamma)
    L =  - tf.math.reduce_sum(w1 * label * tf.math.log(pred))
    return L
