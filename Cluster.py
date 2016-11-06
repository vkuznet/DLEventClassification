# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 13:32:24 2016

@author: mattwallingford
"""
import preprocess
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AffinityPropagation
import random
from matplotlib import colors

MAX_CLUSTER = 3

def remove_labels(df):
    del df['evt']
    del df['lumi']
    del df['run']
    del df['label']
    return df
    
    

def find_best_cluster(df):
    K = range(1,MAX_CLUSTER)
    models = [KMeans(n_clusters=k, random_state = 0).fit(df) for k in K]
    print('Completed: training...')
    centroids = [m.cluster_centers_ for m in models]
    print('Computed: centroids...')
    all_distances = [cdist(df,C, 'euclidean') for C in centroids]
    dist = [np.min(d,axis=1) for d in all_distances]
    print('Computed: distances...')
    avg_distances = [sum(distances)/len(df.iloc[:,1]) for distances in dist]
    print(avg_distances)
    #model = cl.k_means(clusters = k)
    #cluster_label = model.fit_predict(df)
    
    return avg_distances
    
#Try more clusters, 
def plot_elbow(df):
    y = find_best_cluster(df)
    x = range(1,MAX_CLUSTER)
    plt.plot(x,y)
    plt.axis([1,MAX_CLUSTER,2,8])
    plt.show()
    
def plot_clusters_k_means(df):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    pca = PCA(n_components = 3)
    trans_df = pca.fit_transform(df)
    trans_df = pd.DataFrame(trans_df)
    print('Transformed data...')
    predictions = KMeans(n_clusters = 10, random_state = 3).fit_predict(df)
    colors = convert_cluster_to_color(predictions)
    print('Predicted Colors...')
    
    x = np.array(trans_df.iloc[:,0])
    y = np.array(trans_df.iloc[:,1])
    z = np.array(trans_df.iloc[:,2])
    ax.scatter(xs = x,ys = y,zs = z, c = colors)
    #plt.axis([-1000,1000,-1000,1000])
    plt.show()
    
def plot_clusters_spectral(df):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    samples = np.random.choice([False, True],len(df.iloc[:,1]), p = [.99,.01])
    
    df1 = df[samples]
    print(len(df1))
    pca = PCA(n_components = 3)
    trans_df = pca.fit_transform(df)
    trans_df = pd.DataFrame(df)
    print('Transformed data...')
    
    model = AffinityPropagation().fit(df1)
    print('Constructed: Model')
    predictions = model.predict(df)
    print(len(model.cluster_centers_indices_))
    colors = convert_cluster_to_color(predictions)
    print('Predicted Colors...')
    
    x = np.array(trans_df.iloc[:,0])
    y = np.array(trans_df.iloc[:,1])
    z = np.array(trans_df.iloc[:,2])
    ax.scatter(xs = x,ys = y,zs = z, c = colors)
    #plt.axis([-100,100,-100,100, -100,100])
    ax = Axes3D(fig)
    ax.set_xlim3d(-1000, 1000)
    ax.set_ylim3d(-1000,1000)
    ax.set_zlim3d(-1000,1000)
    plt.show()

def convert_cluster_to_color(predictions):
    color = []
    for pred in predictions:
        if pred < 150:
            color.append(list(colors.cnames.keys())[pred])
        else:
            color.append('red')
    return color
    
    
    
if __name__ == "__main__":
    #df = remove_labels(preprocess.process_container_folder('cmsdata-2', '5-param'))
    df = preprocess.process_container_folder('cmsdata-2', '5-param')
    #cl.k_meanscluster(df)
    pass