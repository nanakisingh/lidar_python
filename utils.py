### Utils ###

import math
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import cluster
import time

'''
Timer class
'''
class Timer:
    def __init__(self):
        self.data = dict()

    def start(self, timer_name):
        self.data[timer_name] = [time.time(), 0]

    def end(self, timer_name, msg="", ret=False):
        self.data[timer_name][1] = time.time()
        

        if not ret:
            print(f"{timer_name}: {(self.data[timer_name][1] - self.data[timer_name][0]) * 1000:.2f} ms {msg}")
        else:
            return round((self.data[timer_name][1] - self.data[timer_name][0]) * 1000, 3)
    
    def total(self):
        total = 0.0
        for section in self.data: 
            total += (self.data[section][1] - self.data[section][0])
        total = round(total*1000, 3)
        print(f"Total time: {total} ms")
        return total
    
    def total_specific(self, data):
        total = 0.0
        for section in data: 
            total += (self.data[section][1] - self.data[section][0])
        total = round(total*1000, 3)
        print(f"Total time: {total} ms")
        return total

'''
Visualisation Code
'''
def visualise(points_unfiltered, point_filtered):
    fig = plt.figure()

    # First frame
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_xlim([0, 40])
    ax1.set_ylim([-20, 20])
    ax1.set_zlim([-3, 1])
    ax1.scatter(points_unfiltered[:, 0], points_unfiltered[:, 1], points_unfiltered[:, 2], s=1)
    # ax1.view_init(elev=90, azim=0)
    ax1.set_title("Points Pre-processed")

    # Second frame
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_xlim([0, 40])
    ax2.set_ylim([-20, 20])
    ax2.set_zlim([-3, 1])
    ax2.scatter(point_filtered[:, 0], point_filtered[:, 1], point_filtered[:, 2], s=1)
    # ax2.view_init(elev=90, azim=0)
    ax2.set_title("Points Post-processed")

    plt.show()

def visualise_2(true_points, predicted_centers):
        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([0, 40])
    ax.set_ylim([-20, 20])
    ax.set_zlim([-15, 15])
    ax.scatter(true_points[:, 0], true_points[:, 1], true_points[:, 2], s=1)
    ax.scatter(predicted_centers[:, 0], predicted_centers[:, 1], predicted_centers[:, 2], s=1, c= 'red')

    plt.show()
    
def visualise_3(true_points):
        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([0, 40])
    ax.set_ylim([-20, 20])
    ax.set_zlim([-15, 15])
    ax.scatter(true_points[:, 0], true_points[:, 1], true_points[:, 2], s=1)

    plt.show()


'''
    Determine accuracy of predictions
'''

def evaluate_clusters(clusters, predicted_points, MoE=0.5):
    results = []
    successful_predictions = 0

    # Iterate over each predicted point
    for pred_point in predicted_points:
        # Calculate the Euclidean distance between the predicted point and each cluster center
        distances = np.linalg.norm(clusters - pred_point, axis=1)
                
        # Check if any of the distances are within the margin of error
        within_moe = distances <= MoE
        if np.any(distances <= MoE):
            successful_predictions += 1
        
        # Append the result (True if within 0.5 for any cluster, False otherwise)
        results.append(np.any(within_moe))

    accuracy = (successful_predictions / len(predicted_points)) * 100
    return accuracy, results

