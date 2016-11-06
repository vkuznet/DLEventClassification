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
CLUSTERS = 150


def bag_clusters(df):
    x_data = []
    y_data = []
    start = time.time()
    predictions = KMeans(n_clusters = CLUSTERS, random_state = 3).fit_predict(df)
    end = time.time()
    print(end - start())
    df['cluster']= predictions
    gb = df.groupby(['lumi','evt','run'])
    for gp in gb:
        grouped_df = gp[1]
        dict_predictions = Counter(grouped_df['cluster'])
        vector = predict_to_vector(dict_predictions,CLUSTERS)
        x_data.append(vector)
        y_data.append(grouped_df['label'].iloc[0])
    return x_data, y_data

def predict_to_vector(dict_predictions, cluster_num):
    vector = []
    for i in range(0,cluster_num):
        vector.append(dict_predictions[i])
    return vector
    
    
if __name__ == "__main__":
    df = preprocess.process_container_folder('cmsdata-2', '5-param')
    x_data, labels = bag_clusters(df)
    x_tr, x_tst, y_tr, y_tst = train_test_split(x_data, labels, test_size=0.33, random_state=3)
    #clf = svm.SVC(kernel = 'rbf', gamma = .001)
    clf = RandomForestClassifier(n_estimators = 500)
    clf.fit(x_tr, y_tr)
    print(clf.score(x_tst,y_tst))
        
    
    
    