import tensorflow as tf
import pandas as pd
import numpy as np
import sys
sys.path.append('D:\\PythonSpace\\TensorFlow')
import factorMat as FM

startDate = '2017-01-01'
endDate = '2017-12-31'
factorList = ['rtn_5', 'volume']
SAMPLE_NUM = 5000
NODES_IN = len(factorList)
NODES_OUT = 21
HIDDEN_NODES = 5
LEARNING_RATE = 0.5
STEP_NUM = 2000


def prepareData_StockFactor():
    fobj = FM.factorMat(factorList,startDate,endDate)
    x_data,y_data = fobj.loadStockFactor_NeutualNetwork(0,100)
    return x_data, y_data

def prepareData_test():
    x_data = np.linspace(-1,1,300)[:,np.newaxis]
    noise = np.random.normal(0,0.05,x_data.shape)
    y_data = np.square(x_data) + noise
    return x_data, y_data

def nn_layer(input_data,input_size,output_size,activation_function=None):
    with tf.name_scope('layer'):
        Weight = tf.Variable(tf.random_uniform([input_size,output_size],0,1.0))
        Bias = tf.Variable(tf.zeros([1,output_size]) + 0.1)
        output = tf.matmul(input_data, Weight) + Bias
        #dropout
        output = tf.nn.dropout(output,keep_prob=0.6)
        if activation_function == None:
            return output
        else:
            return activation_function(output)

x_data, y_data = prepareData_StockFactor()
x_data = x_data[:min(SAMPLE_NUM,len(x_data)-1)]
y_data = y_data[:min(SAMPLE_NUM,len(y_data)-1)]

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32,[None,NODES_IN],name='x_input')
    ys = tf.placeholder(tf.float32,[None,NODES_OUT],name='y_input')

layer_h1 = nn_layer(xs,NODES_IN,HIDDEN_NODES,activation_function=tf.nn.relu)
layer_h2 = nn_layer(layer_h1,HIDDEN_NODES,HIDDEN_NODES,activation_function=tf.nn.relu)
y_pred = nn_layer(layer_h2,HIDDEN_NODES,NODES_OUT)

# 均方误差--回归问题
# loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - y_pred),reduction_indices=[1]))
# 交叉熵--分类问题
with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=ys,logits=y_pred)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean

with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    writer = tf.summary.FileWriter("logs/",sess.graph)
    sess.run(init)
    for step in range(STEP_NUM):
        sess.run(train,feed_dict={xs:x_data,ys:y_data})
        if step % 50 == 0:
            print(step,' Loss-',sess.run(loss,feed_dict={xs:x_data,ys:y_data}))