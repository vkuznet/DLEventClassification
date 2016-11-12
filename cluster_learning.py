# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 07:12:02 2016

@author: mattwallingford
"""


import preprocess
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AffinityPropagation
from matplotlib import colors
import collections
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import time
import preprocess
CLUSTERS = 50
PROPC = .1

def bag_clusters(df, clusters = 50, proportion = .4):
    x_data = []
    y_data = []
    samples = np.random.choice([False, True],len(df.iloc[:,1]), p = [1-proportion,proportion]);
    #df = df.loc[:, 'charge':'sis_49_z']
    model = KMeans(n_clusters = clusters, random_state = 3).fit(df.loc[:,'pt':'evt'][samples])
    predictions = model.predict(df.loc[:,'pt':'evt'])
    df['cluster']= predictions
    gb = df.groupby(['lumi','evt','run'])
    for gp in gb:
        grouped_df = gp[1]
        dict_predictions = Counter(grouped_df['cluster'])
        vector = predict_to_vector(dict_predictions,clusters)
        #vector = normalize_vector(vector)
        x_data.append(vector)
        y_data.append(grouped_df['label'].iloc[0])
    return x_data, y_data

def predict_to_vector(dict_predictions, cluster_num):
    vector = []
    for i in range(0,cluster_num):
        vector.append(dict_predictions[i])
    return vector

def normalize_vector(vector):
    norm = sum(vector)
    vector_norm = []
    for v in vector:
        vector_norm.append(v/norm)
    return vector_norm
    
if __name__ == "__main__":
    df = preprocess.process_container_folder('cmsdata-2', '5-param')
    print('done preprocessing...')
    x_data, labels = bag_clusters(df)
    print('done clustering...')
    x_tr, x_tst, y_tr, y_tst = train_test_split(x_data, labels, test_size=0.33, random_state=3)
    #clf = svm.SVC(kernel = 'rbf', gamma = .001)
    clf = RandomForestClassifier(n_estimators = 500)
    clf.fit(x_tr, y_tr)
    print(clf.score(x_tst,y_tst))
        
    
    
    