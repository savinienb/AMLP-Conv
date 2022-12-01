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
    beta_shape = [1] * alpha.get_shape().ndims
    beta_shape[1] = num_classes
    beta = tf.constant(np.ones(beta_shape), dtype=tf.float32)
    S_alpha = tf.reduce_sum(alpha, axis=1, keepdims=True)
    S_beta = tf.reduce_sum(beta, axis=1, keepdims=True)
    lnB = tf.math.lgamma(S_alpha) - tf.reduce_sum(tf.math.lgamma(alpha), axis=1, keepdims=True)
    lnB_uni = tf.reduce_sum(tf.math.lgamma(beta), axis=1, keepdims=True) - tf.math.lgamma(S_beta)

    dg0 = tf.math.digamma(alpha)
    dg1 = tf.math.digamma(S_alpha)

    kl = tf.reduce_sum((alpha - beta) * (dg0 - dg1), axis=1, keepdims=True) + lnB + lnB_uni
    return kl


def mse_loss(one_hot_label, evidence, annealing_coef, num_classes):
    alpha = evidence + 1
    S = tf.reduce_sum(alpha, axis=1, keepdims=True) + 1e-8
    p_k = alpha / S

    # A = tf.reduce_sum((one_hot_label - p_k) ** 2, axis=1, keepdims=True)
    # #B = tf.reduce_sum(alpha * (S - alpha) / (S * S * (S + 1)), axis=1, keepdims=True) # equal to (p_k * (1-p_k))/(S+1)
    # B = tf.reduce_sum(p_k * (1 - p_k) / (S + 1), axis=1, keepdims=True) # equal to (p_k * (1-p_k))/(S+1)
    # loss_data = A + B
    A = tf.reduce_sum((one_hot_label * S - alpha) ** 2, axis=1, keepdims=True)
    B = tf.reduce_sum(alpha * (S - alpha) / (S + 1), axis=1, keepdims=True) # equal to (p_k * (1-p_k))/(S+1)
    #B = tf.reduce_sum(p_k * (1 - p_k) / (S + 1), axis=1, keepdims=True) # equal to (p_k * (1-p_k))/(S+1)
    loss_data = (A + B) / (S * S)

    alpha_wave = evidence * (1 - one_hot_label) + 1 # should be equal to one_hot_label + (1 - one_hot_label) * alpha
    loss_kl = annealing_coef * KL(alpha_wave, num_classes)
    return loss_data, loss_kl

def loss_eq5(one_hot_label, evidence, annealing_coef, num_classes):
    alpha = evidence + 1
    S = tf.reduce_sum(alpha, axis=1, keepdims=True)
    loglikelihood = tf.reduce_sum((one_hot_label-(alpha/S))**2, axis=1, keepdims=True) + tf.reduce_sum(alpha*(S-alpha)/(S*S*(S+1)), axis=1, keepdims=True)
    alpha_wave = evidence * (1 - one_hot_label) + 1 # should be equal to one_hot_label + (1 - one_hot_label) * alpha
    loss_kl = annealing_coef * KL(alpha_wave, num_classes)
    return loglikelihood, loss_kl

def loss_eq4(one_hot_label, evidence, annealing_coef, num_classes):
    alpha = evidence + 1
    loglikelihood = tf.reduce_sum(one_hot_label * (tf.digamma(tf.reduce_sum(alpha, axis=1, keepdims=True)) - tf.digamma(alpha)), axis=1, keepdims=True)
    alpha_wave = evidence * (1 - one_hot_label) + 1 # should be equal to one_hot_label + (1 - one_hot_label) * alpha
    loss_kl = annealing_coef * KL(alpha_wave, num_classes)
    return loglikelihood, loss_kl

def loss_eq3(one_hot_label, evidence, annealing_coef, num_classes):
    alpha = evidence + 1
    loglikelihood = tf.reduce_sum(one_hot_label * (tf.log(tf.reduce_sum(alpha, axis=1, keepdims=True)) - tf.log(alpha)), axis=1, keepdims=True)
    alpha_wave = evidence * (1 - one_hot_label) + 1 # should be equal to one_hot_label + (1 - one_hot_label) * alpha
    loss_kl = annealing_coef * KL(alpha_wave, num_classes)
    return loglikelihood, loss_kl

def loss_eq3_with_exp(one_hot_label, evidence, annealing_coef, num_classes):
    shape = evidence.get_shape().as_list()
    shape[1] = 1
    alpha_concat_log_num_classes = tf.concat([evidence, tf.ones(shape, np.float32) * np.log(num_classes)], axis=1)
    loglikelihood = tf.reduce_sum(one_hot_label * (tf.math.reduce_logsumexp(alpha_concat_log_num_classes, axis=1, keepdims=True) - tf.nn.softplus(evidence)), axis=1, keepdims=True)

    alpha_wave = tf.exp(evidence) * (1 - one_hot_label) + 1 # should be equal to one_hot_label + (1 - one_hot_label) * alpha
    loss_kl = annealing_coef * KL(alpha_wave, num_classes)
    return loglikelihood, loss_kl
