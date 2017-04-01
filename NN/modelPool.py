import DLEventClassificationTF as DLTF
import tensorflow as tf
import pandas as pd
import numpy as np

'''

Model functions 
ref.: goo.gl/Ic8GFt

'''

# lengthy way of defining NN

def model_5(data, num_input, n_classes):
    nodes_layer1 = 500
    nodes_layer2 = 1000
    nodes_layer3 = 1000
    nodes_layer4 = 1000
    nodes_layer5 = 500

    layer1 = {'weights': tf.Variable(tf.random_normal([num_input,nodes_layer1], dtype=tf.float64)),
               'biases': tf.Variable(tf.random_normal([nodes_layer1], dtype=tf.float64))}

    layer2 = {'weights': tf.Variable(tf.random_normal([nodes_layer1,nodes_layer2], dtype=tf.float64)),
               'biases': tf.Variable(tf.random_normal([nodes_layer2], dtype=tf.float64))}

    layer3 = {'weights': tf.Variable(tf.random_normal([nodes_layer2,nodes_layer3], dtype=tf.float64)),
               'biases': tf.Variable(tf.random_normal([nodes_layer3], dtype=tf.float64))}

    layer4 = {'weights': tf.Variable(tf.random_normal([nodes_layer3,nodes_layer4], dtype=tf.float64)),
               'biases': tf.Variable(tf.random_normal([nodes_layer4], dtype=tf.float64))}

    layer5 = {'weights': tf.Variable(tf.random_normal([nodes_layer4,nodes_layer5], dtype=tf.float64)),
               'biases': tf.Variable(tf.random_normal([nodes_layer5], dtype=tf.float64))}

    layer6 = {'weights': tf.Variable(tf.random_normal([nodes_layer5,n_classes], dtype=tf.float64)),
               'biases': tf.Variable(tf.random_normal([n_classes], dtype=tf.float64))}


    l1 = tf.add(tf.matmul(data, layer1['weights']),layer1['biases']) 
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, layer2['weights']),layer2['biases']) 
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, layer3['weights']),layer3['biases']) 
    l3 = tf.nn.relu(l3)

    l4 = tf.add(tf.matmul(l3, layer4['weights']),layer4['biases']) 
    l4 = tf.nn.relu(l4)

    l5 = tf.add(tf.matmul(l4, layer5['weights']),layer5['biases']) 
    l5 = tf.nn.relu(l5)

#    regloss = (tf.nn.l2_loss(layer1['weights']) + tf.nn.l2_loss(layer2['weights']) + 
#              tf.nn.l2_loss(layer3['weights']) + tf.nn.l2_loss(layer4['weights']) +
#              tf.nn.l2_loss(layer5['weights']))

    # keep_prob = tf.placeholder(tf.float64)
    # layer_drop = tf.nn.dropout(l4, keep_prob)

    output = tf.matmul(l5,layer6['weights']) + layer6['biases']
    return output
#    return output, regloss



# lengthy way of defining NN
def model_10(data, num_input, n_classes):
    nodes_layer1 = 500
    nodes_layer2 = 1000
    nodes_layer3 = 1000
    nodes_layer4 = 1000
    nodes_layer5 = 1000
    nodes_layer6 = 1000
    nodes_layer7 = 1000
    nodes_layer8 = 1000
    nodes_layer9 = 1000
    nodes_layer10 = 500

    layer1 = {'weights': tf.Variable(tf.random_normal([num_input,nodes_layer1], dtype=tf.float64)),
               'biases': tf.Variable(tf.random_normal([nodes_layer1], dtype=tf.float64))}

    layer2 = {'weights': tf.Variable(tf.random_normal([nodes_layer1,nodes_layer2], dtype=tf.float64)),
               'biases': tf.Variable(tf.random_normal([nodes_layer2], dtype=tf.float64))}

    layer3 = {'weights': tf.Variable(tf.random_normal([nodes_layer2,nodes_layer3], dtype=tf.float64)),
               'biases': tf.Variable(tf.random_normal([nodes_layer3], dtype=tf.float64))}

    layer4 = {'weights': tf.Variable(tf.random_normal([nodes_layer3,nodes_layer4], dtype=tf.float64)),
               'biases': tf.Variable(tf.random_normal([nodes_layer4], dtype=tf.float64))}

    layer5 = {'weights': tf.Variable(tf.random_normal([nodes_layer4,nodes_layer5], dtype=tf.float64)),
               'biases': tf.Variable(tf.random_normal([nodes_layer5], dtype=tf.float64))}

    layer6 = {'weights': tf.Variable(tf.random_normal([nodes_layer5,nodes_layer6], dtype=tf.float64)),
               'biases': tf.Variable(tf.random_normal([nodes_layer6], dtype=tf.float64))}

    layer7 = {'weights': tf.Variable(tf.random_normal([nodes_layer6,nodes_layer7], dtype=tf.float64)),
               'biases': tf.Variable(tf.random_normal([nodes_layer7], dtype=tf.float64))}
    
    layer8 = {'weights': tf.Variable(tf.random_normal([nodes_layer7,nodes_layer8], dtype=tf.float64)),
               'biases': tf.Variable(tf.random_normal([nodes_layer8], dtype=tf.float64))}

    layer9 = {'weights': tf.Variable(tf.random_normal([nodes_layer8,nodes_layer9], dtype=tf.float64)),
               'biases': tf.Variable(tf.random_normal([nodes_layer9], dtype=tf.float64))}

    layer10 = {'weights': tf.Variable(tf.random_normal([nodes_layer9,nodes_layer10], dtype=tf.float64)),
               'biases': tf.Variable(tf.random_normal([nodes_layer10], dtype=tf.float64))}

    layer11 = {'weights': tf.Variable(tf.random_normal([nodes_layer10,n_classes], dtype=tf.float64)),
               'biases': tf.Variable(tf.random_normal([n_classes], dtype=tf.float64))}


    l1 = tf.add(tf.matmul(data, layer1['weights']),layer1['biases']) 
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, layer2['weights']),layer2['biases']) 
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, layer3['weights']),layer3['biases']) 
    l3 = tf.nn.relu(l3)

    l4 = tf.add(tf.matmul(l3, layer4['weights']),layer4['biases']) 
    l4 = tf.nn.relu(l4)

    l5 = tf.add(tf.matmul(l4, layer5['weights']),layer5['biases']) 
    l5 = tf.nn.relu(l4)

    l6 = tf.add(tf.matmul(l5, layer6['weights']),layer6['biases']) 
    l6 = tf.nn.relu(l6)

    l7 = tf.add(tf.matmul(l6, layer7['weights']),layer7['biases']) 
    l7 = tf.nn.relu(l7)

    l8 = tf.add(tf.matmul(l7, layer8['weights']),layer8['biases']) 
    l8 = tf.nn.relu(l8)

    l9 = tf.add(tf.matmul(l8, layer9['weights']),layer9['biases']) 
    l9 = tf.nn.relu(l9)

    l10 = tf.add(tf.matmul(l9, layer10['weights']),layer10['biases']) 
    l10 = tf.nn.relu(l10)

    output = tf.matmul(l10,layer11['weights']) + layer11['biases']
    return output

def model_x(data, num_input, n_classes, complexity):
    nodes_layer = [1000] * (complexity+1)
    nodes_layer[0] = 0
    nodes_layer[1] = 500
    nodes_layer[-1] = 500

    layer1 = {'weights': tf.Variable(tf.random_normal([num_input,nodes_layer[1]], dtype=tf.float64)),
               'biases': tf.Variable(tf.random_normal([nodes_layer[1]], dtype=tf.float64))}
    layers = {}
    layers[1]=layer1

    for layerNum in range(1,complexity):
        layers[layerNum + 1] = {'weights': tf.Variable(tf.random_normal([nodes_layer[layerNum],nodes_layer[layerNum + 1]], dtype=tf.float64)),
               'biases': tf.Variable(tf.random_normal([nodes_layer[layerNum + 1]], dtype=tf.float64))} 
    
    layers[complexity+1] = {'weights': tf.Variable(tf.random_normal([nodes_layer[complexity],n_classes], dtype=tf.float64)),
               'biases': tf.Variable(tf.random_normal([n_classes], dtype=tf.float64))}

    l = {}
    l[1] = tf.add(tf.matmul(data, layers[1]['weights']),layers[1]['biases']) 
    l[1] = tf.nn.relu(l[1])
    for ll in range(2,len(nodes_layer)):
        l0 = tf.add(tf.matmul(l[ll-1], layers[ll]['weights']),layers[ll]['biases'])  
        l0 = tf.nn.relu(l0)
        l[ll] = l0     

    output = tf.matmul(l[complexity],layers[complexity+1]['weights']) + layers[complexity+1]['biases']
    return output
