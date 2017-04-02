'''

Tensorflow Application @ Deep learning in High Energy Physics

Created by Lijie Tu 01/18/2017

Contact: LT374@cornell.edu

'''

import numpy as np
import pandas as pd
import os

'''

Modified from preprocess.py to fit current needs
This function reads dirs of files

'''
def read_paths(dir):

    paths = os.listdir(dir)
#    paths.pop(0) # MacOS delete
    data = []
    label = 0

    for path in paths:
        data.append(read_files(dir + path, label))
        # print dir + path
        # print len(data)
        # print len(data[label])
        label += 1

    result = pd.concat(data)

    #print 'label', label

    return label,result

'''

Modified from preprocess.py to fit current needs
This function reads and add labels to files

'''
def read_files(dir, label):
    file_names = os.listdir(dir)
 #   file_names.pop(0) # MacOS delete
    data = []
    
    for name in file_names:
        abs_path = dir + '/' + name
        dataOneFile = pd.DataFrame(np.load(abs_path))
        dataOneFile['label'] = [label] * len(dataOneFile.index)
        data.append(dataOneFile)

    result = pd.concat(data)

    return result

'''

Modified from preprocess.py to fit current needs
This function sort and assemble the data based on groups

'''
def process_data(data, num_features, length):
    # copy the data to avoid messing up the original
    data_sorted = data.copy()
    data_sorted.sort_values(['lumi','evt','run'])
    data_grouped = data_sorted.groupby(['lumi','evt','run'])

    train_x = []
    train_y = []
    dataGroupExtract = []

    for ele in data_grouped:
        dataGroup = ele[1].sort_values('pt', ascending=0)
        if len(dataGroup.index) >= length:
            dataGroup = dataGroup.head(length)
            dataGroupExtract = pd.concat([dataGroup['pt'],dataGroup['eta'],dataGroup['phi'],dataGroup['dxy'],dataGroup['dz']])

        else:
            dataGroupExtract = pd.concat([dataGroup['pt'],dataGroup['eta'],dataGroup['phi'],dataGroup['dxy'],dataGroup['dz']])
            makeup = length * num_features - len(dataGroupExtract)
            dataGroupExtract = dataGroupExtract.append(pd.Series([0] * makeup))


        train_x.append(dataGroupExtract)
        train_y.append(dataGroup['label'].iloc[0])

    return train_x, train_y

'''

Split the data into training and testing sets

'''

def split_data(x_data,y_data, eta):
    x_data_np = pd.np.array(x_data)
    y_data_np = pd.np.array(y_data)
    mask = np.random.rand(len(x_data_np)) < eta
    x_train = x_data_np[mask]
    x_test = x_data_np[~mask]
    y_train = y_data_np[mask]
    y_test = y_data_np[~mask]

    return x_train, x_test, y_train, y_test


'''

Convert the single integer label into one-hot array

'''
def convert_one_hot(label, n_classes, num_input):
    one_hot = np.zeros((len(label),n_classes))
    one_hot[np.arange(len(label)),label] = 1

    return one_hot





