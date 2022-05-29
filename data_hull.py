#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 28 22:17:21 2022

@author: lucas
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
import argparse

### Verify which predictions are in within the hull
def point_in_hull(point, hull, tolerance=1e-12):
    return all(
        (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
        for eq in hull.equations)

def calc_pred_domain(set1, set2, npcs) :
    
    ### set1 -> firts dataset
    ### set2 -> second dataset
    ### npcs -> number of PCs to be considered for the convex hull
    
    train = pd.read_csv(set1, sep=",")
    pred = pd.read_csv(set2, sep=",")


### Stack both datasets

    stack=np.vstack((train, pred))
    
    ## calculate PCs
    pca = PCA(n_components=npcs)
    d_pca = pca.fit_transform(stack)
    
    
    train_pc=d_pca[0:(train.shape[0])]
    pred_pc=d_pca[(train.shape[0]):]
    
    
    ### create Hull
    hull = ConvexHull(train_pc)
    
    pred_index=[]
    for i in range(pred_pc.shape[0]):
        if point_in_hull(pred_pc[i,], hull) == True:
            pred_index.append(i)
    
    return pred_index

def main():
    
    parser = argparse.ArgumentParser(description='')
    
    parser.add_argument('--training', type=str, required=True)
    parser.add_argument('--prediction', type=str, required=True)
    parser.add_argument('--npcs', type=int, required=True)
    args = parser.parse_args()
    
    
    pred_index = calc_pred_domain(args.training, args.prediction, args.npcs)

    
    pred_index = pd.DataFrame(pred_index)
    pred_index = pd.DataFrame(pred_index)
    pred_index.to_csv("result.csv",index=False, header=False)    
                    
if __name__=='__main__':
    main()

