'''
Tensorflow application to Deep learning in High Energy Physics

ref.: goo.gl/zB6rCo

Created by Lijie Tu 01/18/2017

Contact: LT374@cornell.edu 

1. normalization
2. iloc[0] label correct? 
3. regularization
4. shuffle the data
5. drop some nodes
'''

import DLEventClassificationTF as DLTF
import modelPool as model
import tensorflow as tf
import pandas as pd
import numpy as np
import scipy
import subprocess
import sys
import time

start = time.time()
start00 = time.time()
num_features = 5
length = int(sys.argv[1])
complexity = int(sys.argv[2])
batch = 500
periods = 200
start0 = time.time()

n_classes,data = DLTF.read_paths('/mnt/scratch/lt374/data2/')
x_data, y_data = DLTF.process_data(data, num_features, length)

split_rate = 0.70
x_train, x_test, y_train, y_test = DLTF.split_data(x_data, y_data, split_rate)
data_size = len(x_train)
n_classes = max(max(y_train),max(y_test)) + 1

start1 = time.time()

num_input = length * num_features

y_train = DLTF.convert_one_hot(y_train, n_classes, num_input)
y_test = DLTF.convert_one_hot(y_test, n_classes, num_input)
    
x = tf.placeholder('float64', [None, num_input])
y = tf.placeholder('float64')

prediction = model.model_x(x, num_input, n_classes, complexity)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
 
optimizer = tf.train.AdamOptimizer().minimize(cost)

lossFile = open("loss_"+str(length)+"_"+str(complexity)+".txt","w")
accFile  = open("accuracy_"+str(length)+"_"+str(complexity)+".txt","w")
repoFile = open("report_"+str(length)+"_"+str(complexity)+".txt","w")

middle = time.time()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    memory = subprocess.check_output("bash gpu_monitor.sh", shell=True)
    memory = memory.rstrip()
    repoFile.write(str(memory) + '\n')

    for period in range(periods):
        loss = 0
        looper = 0 
        idx = np.arange(0,len(x_train))
        while looper < len(x_train):
            np.random.shuffle(idx)
            x_train_sf = x_train[idx]
            y_train_sf = y_train[idx]
            start = looper
            end = looper + batch
            xp = np.array(x_train_sf[start:end])
            yp = np.array(y_train_sf[start:end]) 
            _, l = sess.run([optimizer,cost],feed_dict={x:xp, y:yp}) 	
            loss += l
            looper += batch

       # print 'Period: ', period, ' completed out of ', periods, ' loss: ', loss

	lossFile.write(str(loss) + '\n')
	
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))

        accuracy = tf.reduce_mean(tf.cast(correct,'float64'))

        accPer = accuracy.eval({x:x_test, y:y_test})
       # print 'Period: ', period, ' completed out of ', periods, 'Accuracy: ', accPer

	accFile.write(str(accPer) + '\n')
	

end = time.time()
duration = end-start00
#print start
#print start00
#print start0
#print start1
#print middle
#print end
#print duration
repoFile.write(str(duration) + '\n')


