
### Clustering Algorithm - HDBSCAN ###

import math
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import cluster
import time

'''
HDBSCAN Clustering Algorithm 
    - Performs clustering using HDBSCAN 
    - Predicts cone centers 
    - Outputs (x,y,z) center position of each cone
'''

'''
    Runs HDBSCAN on points
    clusterer.labels_ - each element is the cluster label. Noisy points assigned -1
'''
def run_dbscan(points, eps=0.5, min_samples=1):
    clusterer = cluster.DBSCAN(eps=eps, min_samples=min_samples)
    clusterer.fit(points)

    return clusterer

'''
    Returns the centroid for each cluster 
    Note: no additional filtering performed on size of cluster - i.e. using radial distance or ground plane
'''
def get_centroids_z(points, labels, scalar=1):
  
    points = np.zeros(points.shape) + points[:, :3]
    # Scales z-axis by scalar 
    points[:, 2] *= scalar
    
    
    n_clusters = np.max(labels) + 1
    centroids = []

    # Default probability of 1 to each point 
    probs = np.ones((points.shape[0], 1))

    # Iterate over each cluster 
    for i in range(n_clusters):
      
        # Extract points that belong to n_clusters[i]
        idxs = np.where(labels == i)[0]
        cluster_points = points[idxs]

        # Weighted average center for each cluster of points 
        cluster_probs = probs[idxs]
        scale = np.sum(cluster_probs)
        center = np.sum(cluster_points * cluster_probs, axis=0) / scale

        centroids.append(center)
        
        # NOTE: No additional filtering performed on size of cluster - i.e. using radial distance or ground plane
       

    return np.array(centroids)

'''
    Main code for cone clustering
'''s
def predict_cones_z(points): 
    
    if points.shape[0] == 0:
        return np.zeros((0, 3))

    points = points[:, :3]
    zmax = (points[:, 2] + 1).max(axis=0)
    endscal = (abs(zmax))
    points[:, 2] /= endscal

    # Run DBSCAN - returns probabilities 
    clusterer = run_dbscan(points, min_samples=2, eps=0.3)
    labels = clusterer.labels_.reshape((-1, 1))

    # Extracts centroids for each cluster 
    centroids = get_centroids_z(points, labels)

    return centroids.reshape((-1, 3))

