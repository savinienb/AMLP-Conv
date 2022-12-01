import tensorflow.compat.v1 as tf
import numpy as np

# This function to generate evidence is used for the first example
def relu_evidence(logits):
    return tf.nn.relu(logits)

# This one usually works better and used for the second and third examples
# For general settings and different datasets, you may try this one first
def exp_evidence(logits):
    return tf.exp(tf.clip_by_value(logits,-10,10))

# This one is another alternative and
# usually behaves better than the relu_evidence
def softplus_evidence(logits):
    return tf.nn.softplus(logits)


def KL(alpha, num_classes):
    beta = tf.constant(np.ones((1, num_classes)), dtype=tf.float32)
    S_alpha = tf.reduce_sum(alpha, axis=1, keepdims=True)
    S_beta = tf.reduce_sum(beta, axis=1, keepdims=True)
    lnB = tf.math.lgamma(S_alpha) - tf.reduce_sum(tf.math.lgamma(alpha), axis=1, keepdims=True)
    lnB_uni = tf.reduce_sum(tf.math.lgamma(beta), axis=1, keepdims=True) - tf.math.lgamma(S_beta)

    dg0 = tf.math.digamma(alpha)
    dg1 = tf.math.digamma(S_alpha)

    kl = tf.reduce_sum((alpha - beta) * (dg0 - dg1), axis=1, keepdims=True) + lnB + lnB_uni
    return kl


def mse_loss(one_hot_label, evidence, global_step, annealing_step, num_classes, correction_term=1):
    alpha = evidence + 1
    S = tf.reduce_sum((evidence * correction_term) + 1, axis=1, keepdims=True)
    p_k = alpha / S

    A = tf.reduce_sum((one_hot_label - p_k) ** 2, axis=1, keepdims=True)
    B = tf.reduce_sum(alpha * (S - alpha) / (S * S * (S + 1)), axis=1, keepdims=True) # equal to (p_k * (1-p_k))/(S+1)
    loss_data = A + B
    annealing_coef = tf.minimum(1.0, tf.cast(global_step, tf.float32) / tf.cast(annealing_step, tf.float32))

    alpha_wave = evidence * (1 - one_hot_label) + 1 # should be equal to one_hot_label + (1 - one_hot_label) * alpha

    loss_kl = annealing_coef * KL(alpha_wave, num_classes)
    return loss_data + loss_kl, loss_data, loss_kl

def mse_loss_multiple_labels(one_hot_label_birads, one_hot_label_diagnosis, evidence, global_step, annealing_step, num_classes):
    alpha = evidence + 1
    S = tf.reduce_sum(alpha, axis=1, keepdims=True)
    p_k = alpha / S

    A_birads = tf.reduce_sum((one_hot_label_birads - p_k) ** 2, axis=1, keepdims=True)
    A_diagnosis = tf.reduce_sum((one_hot_label_diagnosis - p_k) ** 2, axis=1, keepdims=True)
    B = tf.reduce_sum(alpha * (S - alpha) / (S * S * (S + 1)), axis=1, keepdims=True) # equal to (p_k * (1-p_k))/(S+1)
    loss_data = tf.reduce_mean([A_birads, A_diagnosis]) + B
    # loss_data = tf.compat.v1.Print(loss_data, [A_birads, A_diagnosis, loss_data])
    annealing_coef = tf.minimum(1.0, tf.cast(global_step, tf.float32) / tf.cast(annealing_step, tf.float32))

    alpha_wave_birads = evidence * (1 - one_hot_label_birads) + 1 # should be equal to one_hot_label + (1 - one_hot_label) * alpha
    alpha_wave_diag = evidence * (1 - one_hot_label_diagnosis) + 1 # should be equal to one_hot_label + (1 - one_hot_label) * alpha

    loss_kl_birads = annealing_coef * KL(alpha_wave_birads, num_classes)
    loss_kl_diagnosis = annealing_coef * KL(alpha_wave_diag, num_classes)
    loss_kl = tf.reduce_mean([loss_kl_birads, loss_kl_diagnosis])
    return loss_data + loss_kl, loss_data, loss_kl


def own_mess(prediction_train, label_one_hot, num_labels, global_step):
    # prediction_train is the evidence vector you get from the network after classification relu layer
    alpha_k = prediction_train + 1
    S = tf.reduce_sum(alpha_k, axis=1, keepdims=True)  # total evidence
    p_k = alpha_k / S  # probability vector expected_prob_vector

    loss_data = tf.reduce_sum(tf.math.square(label_one_hot - p_k) + (p_k * (1 - p_k)) / (S + 1), axis=1)

    alpha_wave = label_one_hot + (1 - label_one_hot) * alpha_k

    alpha_lg = tf.math.lgamma(alpha_wave)
    alpha_S_lg = tf.math.lgamma(tf.reduce_sum(alpha_wave, axis=1))
    alpha_lg_S = tf.reduce_sum(alpha_lg, axis=1, keepdims=True)
    K_lg = tf.math.lgamma(tf.constant(num_labels, tf.float32))
    alpha_dg = tf.math.digamma(alpha_wave)
    alpha_S_dg = tf.math.digamma(tf.reduce_sum(alpha_wave, axis=1))
    loss_KL = alpha_S_lg - (K_lg + alpha_lg_S) + \
                   tf.reduce_sum((alpha_wave - 1) * (alpha_dg - alpha_S_dg), axis=1)
    t = tf.cast(global_step / 1000, tf.float32)
    lamda = tf.math.minimum(1.0, t / 10.0)
    loss_net = tf.reduce_sum(loss_data) + lamda * tf.reduce_sum(loss_KL)

    return loss_net
