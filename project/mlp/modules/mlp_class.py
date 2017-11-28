#!/usr/bin/python2.7
import numpy as np
import tensorflow as tf


class ANN_MLP_Base:
    def __init__(self):
        self.inData = []
        self.outData = []
        self.numTail = -1
        self._wStorage = []
        self._bStorage = []


    def insert_data(self, inData, outData):
        size_col = len(inData)

        assert(size_col == len(outData))

        for i in xrange(size_col):
            self.inData.append([])
            self.outData.append([])
            self.numTail += 1

            for j,datum in enumerate(inData[i]):
                self.inData[self.numTail].append(datum)

            for j,datum in enumerate(outData[i]):
                self.outData[self.numTail].append(datum)


    def save(self, sess, model_name, ver, location = './'):
        saver = tf.train.Saver()
        file_name = location + model_name
        try:
            saver.save(sess, file_name, ver)
        except:
            raise IOError(' -- Is the (%s) directory exist?'%(model_name + str(ver)))


    def load(self, sess, model_name, ver, location = './'):
        saver = tf.train.Saver()
        file_name = location + model_name + '-' + str(ver)
        try:
            saver.restore(sess, file_name)
        except:
            raise IOError(' -- There is no name for %s'%(model_name + str(ver)))


    def get_structure(self, sess, ind):
        W = sess.run(self._wStorage[ind])
        b = sess.run(self._bStorage[ind])

        return W,b


    def _inLayerGenerator(self, outLayer, outNumDim, inNumDim):
        # Generates next layer with normalized signal.

        """
        neuron  out   xW     in    neuron
           0    ---    X    ---     0
           0    ---    X    ---     0
           ..          X    ---     ..
           0    ---    X    ---     0
           0    ---    X    ---     0
                      +b
        """
        W = tf.Variable(tf.random_normal([outNumDim,inNumDim]),dtype = 'float32')
        b = tf.Variable(tf.random_normal([inNumDim]),dtype = 'float32')

        self._wStorage.append(W)
        self._bStorage.append(b)

        inLayer = tf.matmul(tf.cast(outLayer, 'float32'), W) + b

        return inLayer


    def _multiLayerGenerator(self, inputLayer, numNodeList, active = tf.nn.relu):
        # The # of hidden layer should be larger than 1.
        assert(len(numNodeList) >= 3)    

        _outLayer = inputLayer
        size = len(numNodeList)

        # Connect hidden layer
        for i in range(size-2):
            outNumDim = numNodeList[i]
            inNumDim = numNodeList[i+1]
            _inLayer = self._inLayerGenerator(_outLayer,outNumDim,inNumDim)
            _outLayer = active(_inLayer)
            _outLayer = tf.nn.dropout(_outLayer, self.dropout)

        finOutNumDim = numNodeList[-2];
        finInNumDim = numNodeList[-1];

        totLayer = self._inLayerGenerator(_outLayer, finOutNumDim, finInNumDim)

        return totLayer





class ANN_Classification(ANN_MLP_Base):
    def __init__(self, numNodeList, active = tf.nn.relu, learning_rate = 1e-2, l2lambda = 0):
        ANN_MLP_Base.__init__(self)
        inNumDim = numNodeList[0]
        outNumDim = numNodeList[-1]

        self.__numNodeList = numNodeList
        self.dropout = tf.placeholder(tf.float32)
        self.batch_size = tf.placeholder(tf.float32, shape = ())
        self.inLayer = tf.placeholder(tf.float32, [None,inNumDim])
        self.outLayer = tf.placeholder(tf.float32, [None,outNumDim])
        self.totLayer = self._multiLayerGenerator(self.inLayer, numNodeList, active)
        self.pred = tf.nn.softmax(self.totLayer)

        l2Reg = 0
        for wi in self._wStorage:
            l2Reg += tf.norm(wi)
        l2Reg *= l2lambda

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.totLayer, labels = self.outLayer))\
                    + l2Reg

        self.train = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)


    # Stochastic gradient descent
    def run(self, sess, numEpochs, dropout, batchRatio = 1.):
        assert(isinstance(sess,tf.Session))
        
        inData = np.array(self.inData)
        outData = np.array(self.outData)

        totSize = self.numTail + 1

        batchSize = int(totSize * batchRatio)

        print '------------------'
        print "#Total size:",totSize
        print "#batch size:",batchSize
        print '------------------'

        for numEpoch in xrange(numEpochs):
            idx = range(totSize)
            idx = np.random.permutation(idx)
            mini_batchs = [[ inData[ idx[k:k+batchSize] ],
                            outData[ idx[k:k+batchSize] ] ]
                                                  for k in xrange(0, totSize, batchSize)]
            for mini_batch in mini_batchs:
                sess.run( self.train, feed_dict = {  self.inLayer : mini_batch[0], 
                                                     self.outLayer : mini_batch[1], 
                                                     self.batch_size : batchSize,
                                                     self.dropout : dropout       } )

            cost = sess.run( self.cost, 
                             feed_dict = { self.inLayer : inData, 
                                           self.outLayer : outData, 
                                           self.batch_size : float(batchSize),
                                           self.dropout : 1.  } )

            print "epoch:", numEpoch, "cost :", cost,

            prediction_tag_dat = self.predict(sess, inData)
            numSize = len(prediction_tag_dat)

            accum = 0.

            for i,prediction_vector in enumerate(prediction_tag_dat):
                pred_index = np.argmax(prediction_vector)
                if(1 == outData[i,pred_index]):
                    accum += 1.

            accum /= float(numSize)

            print 'acc:', accum

        print "cost :", sess.run( self.cost, feed_dict = { self.inLayer : inData, 
                                                           self.outLayer : outData, 
                                                           self.batch_size : float(totSize),
                                                           self.dropout : 1. } )


    def estimation(self, sess, te_indata, te_tag):
        assert(len(te_indata) == len(te_tag))

        num_count = 0.
        pred = sess.run(self.pred,\
                       feed_dict = {self.inLayer : te_indata,
                                    self.dropout : 1.})

        for i, test_data in enumerate(te_indata):
            if np.argmax(pred[i]) == np.argmax(te_tag[i]):
                num_count += 1.

        num_count /= len(te_tag)

        return num_count


    def predict(self, sess, x):
        return sess.run(self.pred, feed_dict = {self.inLayer : x,
                                                self.dropout : 1})




class ANN_Regression(ANN_MLP_Base):
    def __init__(self, numNodeList, active = tf.sigmoid, learning_rate = 1e-2, l2lambda = 0):
        ANN_MLP_Base.__init__(self)
        inNumDim = numNodeList[0]
        outNumDim = numNodeList[-1]

        self.__numNodeList = numNodeList
        self.dropout = tf.placeholder(tf.float32)
        self.batch_size = tf.placeholder(tf.float32, shape = ())
        self.inLayer = tf.placeholder(tf.float32, [None, inNumDim])
        self.outLayer = tf.placeholder(tf.float32, [None, outNumDim])
        self.totLayer = self._multiLayerGenerator(self.inLayer, numNodeList, active)

        l2Reg = 0
        for wi in self._wStorage:
            l2Reg += tf.norm(wi)
        l2Reg *= l2lambda

        self.cost = tf.reduce_mean((self.totLayer - self.outLayer)**2) + l2Reg

        self.train = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)


    def __call__(self, sess, x):
        #x = x.reshape([-1, self.__numNodeList[0]])
        return sess.run(self.totLayer, feed_dict = {self.inLayer : x,
                                                    self.dropout : 1})


    def run(self, sess, numEpochs, dropout, batchRatio = 1.):
        assert(isinstance(sess,tf.Session))
        
        inData = np.array(self.inData)
        outData = np.array(self.outData)

        totSize = self.numTail + 1

        batchSize = int(totSize * batchRatio)

        print '------------------'
        print "#Total size:",totSize
        print "#batch size:",batchSize
        print '------------------'

        for numEpoch in xrange(numEpochs):
            idx = range(totSize)
            idx = np.random.permutation(idx)
            mini_batchs = [[ inData[ idx[k:k+batchSize] ],
                            outData[ idx[k:k+batchSize] ] ]
                                                  for k in xrange(0, totSize, batchSize)]
            for mini_batch in mini_batchs:
                sess.run( self.train, feed_dict = {  self.inLayer : mini_batch[0], 
                                                     self.outLayer : mini_batch[1], 
                                                     self.batch_size : batchSize,
                                                     self.dropout : dropout       } )

            cost = sess.run( self.cost, 
                             feed_dict = { self.inLayer : inData, 
                                           self.outLayer : outData, 
                                           self.batch_size : float(batchSize),
                                           self.dropout : 1.  } )

            print "epoch:", numEpoch, "cost :", cost

        print "cost :", sess.run( self.cost, feed_dict = { self.inLayer : inData, 
                                                           self.outLayer : outData, 
                                                           self.batch_size : float(totSize),
                                                           self.dropout : 1. } )
