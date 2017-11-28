#!/usr/bin/python2.7
import numpy as np
import tensorflow as tf
from modules.mlp_class import ANN_Regression
from modules.load_data import psi_reader

nodeList = [1, 100, 1001];
machine_name = 'M-%d'%(nodeList[1]); ver = 1; model_name = machine_name + '-' + str(ver)
#----------------------------------------------------------------------
nepoch = 1000; drop_ratio = 0.5; batchR = 1e-1
#----------------------------------------------------------------------
psis = np.linspace(-0.2, 0.2, 5)
#----------------------------------------------------------------------
reg = ANN_Regression(nodeList, tf.sigmoid)
x, y = psi_reader("../raw-data")
y = np.array(y); y = y[0, :, 0]; x_axis = y.reshape([-1])

sess = tf.Session()
sess.run(tf.global_variables_initializer())

try:
    reg.load(sess, model_name, ver, './save-file1/')
    print "\t-----------"
    print "\t!load data:",model_name
    print "\t-----------\n"
except:
    raise IOError(' There are no data for \'%s\''%model_name)


wdata = np.zeros([len(x_axis), len(psis)+1])
wdata[:, 0] = x_axis
for i, psi in enumerate(psis):
    y = reg(sess, np.array([[psi]])); y.reshape([-1])
    wdata[:, i+1] = y

print psis
np.savetxt('m-reg-%s.dat'%nodeList[1], wdata)
