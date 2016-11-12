# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 16:53:45 2016

@author: mattwallingford
"""

import os
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import math
 
FEATURES = 5
MINNUMPARTICLES = 10

"""create dataframe from all folders in folder"""
def process_container_folder(folder, tag):

    if os.path.isfile(tag + 'processed_df.csv'):
        final_df = pd.DataFrame.from_csv(tag+'processed_df.csv')
        return final_df
    
    folder_paths = os.listdir(folder)
    folder_paths.pop(0)#Remove formatting file
    df = []
    index = 0
    for path in folder_paths:
        df.append(process_df_folder(folder + '/' + path, index))
        index += 1
    final_df = pd.concat(df)
    final_df.to_csv(tag + 'processed_df.csv')
    print(final_df)
    return final_df
    

"""append all contents of folder to df and attach label"""
def process_df_folder(folder, label):
    file_paths = os.listdir(folder)
    file_paths.pop(0)
    try:
        file_paths.remove('log')
    except:
        pass
    list_df = []
    for path in file_paths:
        total_path = folder + '/' + path
        df = pd.DataFrame(np.load(total_path))
        df['label'] = [label]*len(df)
        list_df.append(df) 
    final_df = pd.concat(list_df)
    
    return final_df
    
"""sort dataframe according to lumi, evt, and run. Uses up to MINNUMPARTICLES rows
and concatenates into a single vector per event"""
def sort_df_by_event(df):
    df.sort_values(['lumi','evt','run'])
    gb = df.groupby(['lumi','evt','run'])
    train_x = []
    train_y = []
    vector = []
    #metric = 'dz'
    for gp in gb:
        df = gp[1].sort_values('pt', ascending = 0)
        train_y.append(df['label'].iloc[0])
        remove_labels(df)
        """mean = df[metric].mean()
        std = df[metric].std()
        max_pt = df[metric].max()"""
        x = []
        if len(df.index) >= MINNUMPARTICLES:
            df = df.head(MINNUMPARTICLES)
            x = []
            for i in range(0,len(df.iloc[1,:])):
                x.append(df.iloc[:,i])
            vector = pd.concat(x)

        else: 
            for i in range(0,len(df.iloc[1,:])):
                x.append(df.iloc[:,i])
            vector = pd.concat(x)
            num_missing = MINNUMPARTICLES*len(df.iloc[1,:]) - len(vector)
            vector = vector.append(pd.Series([0]*num_missing))
        

        #vector = vector.append(pd.Series([mean, std, max_pt]))
        train_x.append(vector)
    return train_x, train_y
    
def examine_events(df):
    df.sort_values(['lumi','evt','run'])
    gb = df.groupby(['lumi','evt','run'])
    avg_pt0 = []
    avg_pt1 = []
    avg_pt2 = []
    avg_pt3 = []
    metric = 'pt'
    for gp in gb:
        event = gp[1]
        if event['label'].iloc[0] == 0:
            avg_pt0.append(event[metric].mean())
        if event['label'].iloc[0] == 1:
            avg_pt1.append(event[metric].mean())
        if event['label'].iloc[0] == 2:
            avg_pt2.append(event[metric].mean())
        if event['label'].iloc[0] == 3:
            avg_pt3.append(event[metric].mean())
    print('Higgs')
    print(np.array(avg_pt0).mean())
    print(np.array(avg_pt0).std())
    print('MuMu')
    print(np.array(avg_pt1).mean())
    print(np.array(avg_pt1).std())
    print('QCD')
    print(np.array(avg_pt2).mean())
    print(np.array(avg_pt2).std())
    print('Lepton')
    print(np.array(avg_pt3).mean())
    print(np.array(avg_pt3).std())
    plt.scatter(avg_pt0, range(0,len(avg_pt0)))
    plt.axis([0,10,0,1000])
    plt.show()
    plt.scatter(avg_pt1, range(0,len(avg_pt1)))
    plt.axis([0,10,0,1000])
    plt.show()
    plt.scatter(avg_pt2, range(0,len(avg_pt2)))
    plt.axis([0,10,0,1000])
    plt.show()
    plt.scatter(avg_pt3, range(0,len(avg_pt3)))
    plt.axis([0,10,0,1000])
    plt.show()
    
def remove_labels(df):
    del df['evt']
    del df['lumi']
    del df['run']
    del df['label']
    try:
        del df['TrackId']
    except:
        return df
    return df
    
if __name__ == "__main__":
    train_x, labels = sort_df_by_event(process_container_folder('cmsdata_hits_4', '20-param'))
    print("Finished Processing...")
    #print(train_x)
    #pca = PCA(n_components = math.ceil(math.sqrt(MINNUMPARTICLES*5)))
    #trans_x = pca.fit_transform(train_x)
    x_tr, x_tst, y_tr, y_tst = train_test_split(train_x, labels, test_size=0.33, random_state=3)
    #trans1_x = pca.fit_transform(x_tr)
    #trans2_x = pca.fit_transform(x_tst)
    #clf = svm.SVC(kernel = 'rbf', gamma = .001)
    clf = RandomForestClassifier(n_estimators = 500)
    #param_grid = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},{'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}]
    #search = GridSearchCV(clf, param_grid)
    #search.fit(train_x, labels)
    clf.fit(x_tr, y_tr)
    print(clf.score(x_tst,y_tst))
    pass
    
    
    