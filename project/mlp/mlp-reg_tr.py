#!/usr/bin/python2.7
import numpy as np
import tensorflow as tf
from modules.mlp_class import ANN_Regression
from modules.load_data import psi_reader

nodeList = [1, 10, 1001]; learning_rate = 1e-3; l2 = 0;
machine_name = 'M-%d'%(nodeList[1]); ver = 1; model_name = machine_name + '-' + str(ver)
#----------------------------------------------------------------------
nepoch = 1000; drop_ratio = 1; batchR = 1e-1
#----------------------------------------------------------------------
reg = ANN_Regression(nodeList, tf.sigmoid, learning_rate, l2)
x, y = psi_reader("../raw-data")
x = np.array(x); x = x.reshape([-1, 1])
y = np.array(y); y = y[:, :, 1]; y = y.reshape([-1, 1001])

reg.insert_data(x, y)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

try:
    reg.load(sess, model_name, ver, './save-file1/')
    print "\t-----------"
    print "\t!load data:",model_name
    print "\t-----------\n"
except:
    print ' There are no data for \'%s\''%model_name
    pass

reg.insert_data(x, y)
reg.run(sess, nepoch, drop_ratio, batchR)
reg.save(sess, model_name, ver, './save-file1/')
