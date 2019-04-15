# -*- coding: UTF-8 -*-
from flask import Flask, jsonify, request, abort
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

app = Flask(__name__)

rnn_unit = 20       # hidden layer units
input_size = 7
output_size = 1
lr = 1e-5        # learning rate

# ——————————————————import data——————————————————————
f = open('data/train.csv')
df = pd.read_csv(f)
data = df.iloc[:, 0:input_size+1].values
mean = np.mean(data, axis=0)
var = np.var(data, axis=0)
normalized_data = (data-mean)/var
# generate train data
def get_train_data(batch_size=60, time_step=10, train_begin=0, train_end=16000):
    batch_index = []
    # data_train = data[train_begin:train_end]
    # normalized_train_data = (data_train-mean)/var  # normalized
    normalized_train_data = normalized_data[train_begin:train_end]
    train_x, train_y = [], []   # train data
    for i in range(len(normalized_train_data)-time_step):
       if i % batch_size == 0:
           batch_index.append(i)
       x = normalized_train_data[i:i+time_step, :input_size]
       y = normalized_train_data[i:i+time_step, input_size, np.newaxis]
       train_x.append(x.tolist())
       train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data)-time_step))
    return batch_index, train_x, train_y


# generate test data
def get_test_data(time_step=10, test_begin=16000, test_end=17395):
    # data_test = data[test_begin:test_end]
    normalized_test_data = normalized_data[test_begin:test_end]  # normalize
    size = (len(normalized_test_data)+time_step-1)//time_step  # there are size sample
    test_x, test_y = [], []
    for i in range(size-1):
       x = normalized_test_data[i*time_step:(i+1)*time_step, :input_size]
       y = normalized_test_data[i*time_step:(i+1)*time_step, input_size]
       test_x.append(x.tolist())
       test_y.extend(y)
    # test_x.append((normalized_test_data[(i+1)*time_step:, :input_size]).tolist())
    # test_y.extend((normalized_test_data[(i+1)*time_step+1:, input_size]).tolist())
    return test_x, test_y


# ——————————————————Define the parameter of neural network——————————————————
weights = {
         'in':tf.Variable(tf.random_normal([input_size, rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit, 1]))
        }
biases = {
        'in':tf.Variable(tf.constant(0.1, shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1, shape=[1,]))
       }


# ——————————————————Define architecture of Long Short Term Memory network——————————————————
def lstm(X):
    batch_size = tf.shape(X)[0]
    time_step = tf.shape(X)[1]
    w_in = weights['in']
    b_in = biases['in']
    input = tf.reshape(X,[-1,input_size])
    input_rnn = tf.matmul(input,w_in)+b_in
    input_rnn = tf.reshape(input_rnn,[-1,time_step,rnn_unit])
    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state = cell.zero_state(batch_size,dtype=tf.float32)
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
    # output_rnn is the result of each lstm cell，final_states is the result of last cell
    output = tf.reshape(output_rnn, [-1, rnn_unit])
    w_out = weights['out']
    b_out = biases['out']
    pred = tf.matmul(output, w_out)+b_out
    return pred, final_states


# ——————————————————train this model——————————————————
def train_lstm(batch_size=60,time_step=10,train_begin=0,train_end=17300):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    batch_index,train_x,train_y=get_train_data(batch_size, time_step,train_begin,train_end)
    pred,_=lstm(X)
    #loss function
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    tf.summary.scalar('loss',loss)
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    saver=tf.train.Saver(tf.global_variables(), max_to_keep=15)
    merged = tf.summary.merge_all()

    module_file = tf.train.latest_checkpoint('./train3')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, module_file)
        train_writer = tf.summary.FileWriter('train3/log', sess.graph)
        for i in range(10000):
            for step in range(len(batch_index)-1):
                _,summary,loss_=sess.run([train_op,merged,loss], feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]]})
                train_writer.add_summary(summary, i)
            print(i,loss_)

            if i % 1000==0:
                print("save model：", saver.save(sess, os.path.join(os.getcwd()+"\\train3", 'stock.model'), global_step=i))
    train_writer.close()
# train_lstm()


# ————————————————prediction————————————————————
def prediction_test(time_step=10):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    #Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    test_x, test_y = get_test_data(time_step)
    pred,_=lstm(X)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # load parameter
        module_file = tf.train.latest_checkpoint('./train3')
        saver.restore(sess, module_file)
        test_predict = []
        for step in range(len(test_x)-1):
            prob = sess.run(pred, feed_dict={X: [test_x[step]]})
            predict = prob.reshape((-1))
            test_predict.extend(predict)
        test_y = np.array(test_y) * var[input_size] + mean[input_size]
        test_predict = np.array(test_predict)*var[input_size]+mean[input_size]

        # # show the result
        plt.figure()
        for i in range(len(test_predict)):
            print("predict\t" + str(test_predict[i]) + "\t true\t" + str(test_y[i]))
        plt.plot(list(range(len(test_predict))), test_predict, color='b')
        plt.plot(list(range(len(test_y))), test_y,  color='r')
        plt.show()

prediction_test()
