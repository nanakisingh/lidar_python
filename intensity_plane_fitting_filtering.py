'''
Evaluation code

    Filtering: FOV range, Intensity based post 10 metres, random plane fitting for 0-10 metres

'''
import math
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import cluster
import time
import os 

from utils import *
from clustering_fns import *

### Filtering Algorithm - G&C ###

'''
Data preprocessing - remove points clouds outside given FOV range
    - PC is only PC[:, :3] 
    - Remove radial point 0 - 40 
'''

def fov_range(pointcloud, minradius=0, maxradius=40):
    # Calculate radial distance of each point (straight line distance from origin) and removes if outside range
    # made 4 for intensity based 
    pointcloud = pointcloud[:,:4]
    points_radius = np.sqrt(np.sum(pointcloud[:,:2] ** 2, axis=1))

    # Uses mask to remove points to only store in-range points
    radius_mask = np.logical_and(
        minradius <= points_radius, 
        points_radius <= maxradius
    )
    pointcloud = pointcloud[radius_mask]
    return pointcloud

'''
G&C Filtering Algorithm (i.e. Ground Truth)
    - Split point into M segments based on angle 
    - Split each segment into rbins based on radial distance of each point 
    - Fit a line to each segment 
    - Any point below the line is the 'ground' and is filtered
'''

alpha = 0.1
num_bins = 10
height_threshold = 0.13

def grace_and_conrad_filtering(points, alpha, num_bins, height_threshold):
    
    # change so that take lowest x points - averga eand fit a plane across all segments 

    angles = np.arctan2(points[:, 1], points[:, 0])  # Calculate angle for each point
    bangles = np.where(angles < 0, angles + 2 * np.pi, angles)

    # NOTE: making gangles from min to max to avoid iterating over empty regions
    if (bangles.size > 0): 
        
        gangles = np.arange(np.min(bangles), np.max(bangles), alpha)

        # Map angles to segments
        segments = np.digitize(bangles, gangles) - 1 
        # Calculate range for each point
        ranges = np.sqrt(points[:, 0]**2 + points[:, 1]**2) 

        rmax = np.max(ranges)
        rmin = np.min(ranges)
        bin_size = (rmax - rmin) / num_bins
        rbins = np.arange(rmin, rmax, bin_size)
        regments = np.digitize(ranges, rbins) - 1

        M, N = len(gangles), len(rbins)
        grid_cell_indices = segments * N + regments

        gracebrace = []
        for seg_idx in range(M):
            Bines = []
            min_zs = []
            prev_z = None
            for range_idx in range(N):
                bin_idx = seg_idx * N + range_idx
                idxs = np.where(grid_cell_indices == bin_idx)
                bin = points[idxs, :][0]
                if bin.size > 0:
                    min_z = np.min(bin[:, 2])
                    binLP = bin[bin[:, 2] == min_z][0].tolist()
                    min_zs.append(min_z)
                    Bines.append([np.sqrt(binLP[0]**2 + binLP[1]**2), binLP[2]])
                    prev_z = min_z

            if Bines:
                i = 0
                while i < len(min_zs):
                    good_before = i == 0 or min_zs[i] - min_zs[i - 1] < 0.1
                    good_after = i == len(min_zs) - 1 or min_zs[i] - min_zs[i + 1] < 0.1
                    if not (good_before and good_after):
                        Bines.pop(i)
                        min_zs.pop(i)
                        i -= 1
                    i += 1

                seg = segments == seg_idx
                X = [p[0] for p in Bines]
                Y = [p[1] for p in Bines]
                
                X = np.array(X)
                Y = np.array(Y)

                x_bar = np.mean(X)
                y_bar = np.mean(Y)
                x_dev = X - x_bar
                y_dev = Y - y_bar
                ss = np.sum(x_dev * x_dev)

                slope = np.sum(x_dev * y_dev) / np.sum(x_dev * x_dev) if ss != 0 else 0
                intercept = y_bar - slope * x_bar
                
                points_seg = points[seg]
                pc_compare = slope * np.sqrt(points_seg[:, 0]**2 + points_seg[:, 1]**2) + intercept
                pc_mask = (pc_compare + height_threshold) < points_seg[:, 2]
                conradbonrad = points_seg[pc_mask]
                if conradbonrad.tolist(): gracebrace.extend(conradbonrad.tolist())
     
        gracebrace = np.array(gracebrace)
        return gracebrace.reshape((-1, 3))


def intensity_filtering(points, max_dist=40,  g_and_c=True):
    

    points = points[~np.all(points[:, :3] == [0, 0, 0], axis=1)]

    #array of r
    r = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    #range mask
    mask_close = (r < 10)
    mask_far = (r > 10) & (r < 30)

    # Split the data into 2 arrays based on the masks
    close_points = points[mask_close]     # Points with 0 < r < 10
    far_points = points[mask_far]   # Points with 10 < r < 50
    far_r = np.sqrt(far_points[:, 0]**2 + far_points[:, 1]**2)

    #far points ground removal & transformation
    # logic: np.log(far_r) - scaling r by this val makes up for rapid fall of intensity with distance
    # if far_points[:, 3] * far_r * np.log(far_r) is high, the reflecitivity of the object is high 
    # so it is more likeley to be a non-ground point  
    r_mask = (far_points[:, 3] * far_r * np.log(far_r) >= 200)
    
    far_points[~r_mask, :3] = 0
    far_points = far_points[:, :3]
    far_points = far_points[:, [1, 0, 2]] 
    far_points[:, 0] = -far_points[:, 0] 
    far_points = far_points[~np.all(far_points == 0, axis=1)]
    
    if g_and_c: 
        
        # print("CLOSE ", close_points.shape)
        close_points_t = grace_and_conrad_filtering(close_points[:, :3], alpha, num_bins, height_threshold)
        
        
        close_points_t = close_points_t[:, [1, 0, 2]]
        close_points_t[:, 0] = -close_points_t[:, 0] 
        points = np.concatenate((close_points_t , far_points), axis=0)
        points = points[:, [1, 0, 2]]
        points[:, 1] = -points[:, 1]
        return points 
    
    #close points removal & transformation
    threshold = .1
    close_points = close_points[:, :3]
    close_points = close_points[:, [1, 0, 2]] 
    close_points[:, 0] = -close_points[:, 0] 
    
    # logic: randomly selects 300 points and sorts based on z and removes the top 5 values (outliers)
    random_selection = close_points[np.random.choice(close_points.shape[0], 300, replace=False)]
    sorted_selection = random_selection[random_selection[:, 2].argsort()]
    remaining_points = sorted_selection[:-5]
    lowest_z_points = remaining_points
    # logic: solves least square error problem for 295 values and stores coeff for a plane 
    X = lowest_z_points[:, 0]  
    Y = lowest_z_points[:, 1]  
    Z = lowest_z_points[:, 2] 
    A = np.vstack([X, Y, np.ones_like(X)]).T  
    B = Z  
    coeffs, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    a, b, d = coeffs

    # logic: z = a*x + y*b + d - for each point and then keep those greater than plane_z_values + threshold
    plane_z_values = close_points[:, 0] * a + close_points[:, 1] * b + d
    plane_mask = close_points[:, 2] >= plane_z_values + threshold #
    close_points = close_points[plane_mask]

    points = np.concatenate((close_points, far_points), axis=0)
    points = points[:, [1, 0, 2]]
    points[:, 1] = -points[:, 1]
    return points  

### MAIN RUN ###

'''
Base Algorithm - G&C
'''
    
def run_g_and_c(processed_data_frame): 

    # Ground Filtering 
    timer.start("Grace and Conrad Ground Filtering")
    ground_filtered_points = intensity_filtering(processed_data_frame, g_and_c = True)
    # ground_filtered_points = grace_and_conrad_filtering(processed_data_frame, alpha, num_bins, height_threshold)
    timer.end("Grace and Conrad Ground Filtering", "")
    print("Ground Filtered - Num of points:", len(ground_filtered_points))
    print()
    # visualise(single_frame, ground_filtered_points)

    # Clustering 
    timer.start("Clustering")
    clusters = predict_cones_z(ground_filtered_points)
    timer.end("Clustering", "")
    print("Clustered - Num of points:", len(clusters))
    print()
    # visualise(ground_filtered_points, clusters)
    timer.total_specific(["Data pre-processing","Grace and Conrad Ground Filtering","Clustering"])
    print()
    
    return clusters

'''
Comparative Algorithm - Intensity Based
'''
def run_prediction(processed_data_frame):

    # Ground Filtering 
    timer.start("Intensity Ground Filtering")
    ground_filtered_points = intensity_filtering(processed_data_frame, g_and_c = False)
    # ground_filtered_points = grace_and_conrad_filtering(processed_data_frame, alpha, num_bins, height_threshold)
    timer.end("Intensity Ground Filtering", "Completed Ground Filtering")
    print("Ground Filtered - Num of points:", len(ground_filtered_points))
    print()
    # visualise(single_frame, ground_filtered_points)

    # Clustering 
    timer.start("Clustering")
    clusters = predict_cones_z(ground_filtered_points)
    timer.end("Clustering", "Completed Clustering")
    print("Clustered - Num of points:", len(clusters))
    print()
    # visualise(ground_filtered_points, clusters)
    timer.total_specific(["Data pre-processing","Intensity Ground Filtering","Clustering"])
    print()
    
    return clusters




'''
    Single Data point 
'''

['/Users/nanaki/Downloads/tmp/instance-12.npz', '/Users/nanaki/Downloads/tmp/instance-13.npz', '/Users/nanaki/Downloads/tmp/instance-160.npz', '/Users/nanaki/Downloads/tmp/instance-161.npz', '/Users/nanaki/Downloads/tmp/instance-163.npz', '/Users/nanaki/Downloads/tmp/instance-162.npz']

    
    

folder_path = '/Users/nanaki/Downloads/tmp/instance-'+ str((12))+'.npz'

with np.load(folder_path, allow_pickle=True) as data:
    single_frame = data['points']
    # visualise_3(single_frame)

print()
timer = Timer()
# Process data 
timer.start("Data pre-processing")
processed_data_frame = fov_range(single_frame, 0, 40)
visualise_3(processed_data_frame)

timer.end("Data pre-processing", "")
print()
print("---------------")
ground_truth = run_g_and_c(processed_data_frame)
visualise_3(ground_truth)

print("---------------")
predictions = run_prediction(processed_data_frame)
print("---------------")
visualise_3(predictions)


# Determine accuracy
accuracy, results = evaluate_clusters(ground_truth, predictions)
print(f"True results: {results}")
print(f"Accuracy: {accuracy}%")



'''
    Multiple Data Points 
'''


# folder_path = '/Users/nanaki/Downloads/tmp/'
# paths = os.listdir(folder_path)
# errors = []

# for i in range(len(paths)):
#     file_path = os.path.join(folder_path, paths[i])
    
#     # Load in point cloud per frame 
#     with np.load(file_path, allow_pickle=True) as data:
#         single_frame = data['points']
        
#     # Process data point 
#     processed_data_frame = fov_range(single_frame, 0, 40)
#     # G and C 
#     ground_truth = intensity_filtering(processed_data_frame, g_and_c = True)
#     # Intensity 
#     predictions = intensity_filtering(processed_data_frame, g_and_c = False)
    
#     # Calculate accuracy 
#     accuracy, results = evaluate_clusters(ground_truth, predictions)
    
#     print(f"Filename: {file_path}")
#     print(f"Accuracy: {accuracy}%")
    
#     if accuracy < 95:
#         print(f"Error with file {file_path}")
#         errors.append(file_path)

# # Instance 2 
# print("All errors", errors)




    
    

