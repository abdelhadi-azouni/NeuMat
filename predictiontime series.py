import math
import random
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

LSTM_SIZE = 64
LSTM_LAYERS = 2
BATCH_SIZE = 16
NUM_T_STEPS = 4
MAX_STEPS = 1000
LAMBDA_REG = 5e-4


def ground_truth_func(i, j, t):
    return i * math.pow(t, 2) + j


def get_batch(batch_size):
    seq = np.zeros([batch_size, NUM_T_STEPS, 1], dtype=np.float32)
    tgt = np.zeros([batch_size, NUM_T_STEPS], dtype=np.float32)

    for b in xrange(batch_size):
        i = float(random.randint(-25, 25))
        j = float(random.randint(-100, 100))
        for t in xrange(NUM_T_STEPS):
            value = ground_truth_func(i, j, t)
            seq[b, t, 0] = value

        for t in xrange(NUM_T_STEPS):
            tgt[b, t] = ground_truth_func(i, j, t + NUM_T_STEPS)
    return seq, tgt


# Placeholder for the inputs in a given iteration
sequence = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_T_STEPS, 1])
target = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_T_STEPS])

fc1_weight = tf.get_variable('w1', [LSTM_SIZE, 1], initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0))
fc1_bias = tf.get_variable('b1', [1], initializer=tf.constant_initializer(0.1))

# ENCODER
with tf.variable_scope('ENC_LSTM'):
    lstm = rnn.LSTMCell(LSTM_SIZE)
    multi_lstm = rnn.MultiRNNCell([lstm] * LSTM_LAYERS)
    initial_state = multi_lstm.zero_state(BATCH_SIZE, tf.float32)
    state = initial_state
    for t_step in xrange(NUM_T_STEPS):
        if t_step > 0:
            tf.get_variable_scope().reuse_variables()

        # state value is updated after processing each batch of sequences
        output, state = multi_lstm(sequence[:, t_step, :], state)

learned_representation = state

# DECODER
with tf.variable_scope('DEC_LSTM'):
    lstm = rnn.LSTMCell(LSTM_SIZE)
    multi_lstm = rnn.MultiRNNCell([lstm] * LSTM_LAYERS)
    state = learned_representation
    logits_stacked = None
    loss = 0.0
    for t_step in xrange(NUM_T_STEPS):
        if t_step > 0:
            tf.get_variable_scope().reuse_variables()

        # state value is updated after processing each batch of sequences
        output, state = multi_lstm(sequence[:, t_step, :], state)
        # output can be used to make next number prediction
        logits = tf.matmul(output, fc1_weight) + fc1_bias

        if logits_stacked is None:
            logits_stacked = logits
        else:
            logits_stacked = tf.concat(1, [logits_stacked, logits])

        loss += tf.reduce_sum(tf.square(logits - target[:, t_step])) / BATCH_SIZE

reg_loss = loss + LAMBDA_REG * (tf.nn.l2_loss(fc1_weight) + tf.nn.l2_loss(fc1_bias))

train = tf.train.AdamOptimizer().minimize(reg_loss)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    total_loss = 0.0
    for step in xrange(MAX_STEPS):
        seq_batch, target_batch = get_batch(BATCH_SIZE)

        feed = {sequence: seq_batch, target: target_batch}
        _, current_loss = sess.run([train, reg_loss], feed)
        if step % 10 == 0:
            print("@{}: {}".format(step, current_loss))
        total_loss += current_loss

    print('Total loss:', total_loss)

    print('### SIMPLE EVAL: ###')
    seq_batch, target_batch = get_batch(BATCH_SIZE)
    feed = {sequence: seq_batch, target: target_batch}
    prediction = sess.run([logits_stacked], feed)
    for b in xrange(BATCH_SIZE):
        print("{} -> {})".format(str(seq_batch[b, :, 0]), target_batch[b, :]))
        print(" `-> Prediction: {}".format(prediction[0][b]))
