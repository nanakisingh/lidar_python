import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animations
# Path to the folder containing .npz files

# TODO: CHANGE PATH NAME
folder_path = "/Users/nanaki/Downloads/tt-4-18-eleventh"


# Filtering algorithm 

def convert_to_cones(points):
    length = len(points)
    cones = []
    look_ahead = 30
    for i in range(length):
            # get z - x and y 
            r = ((points[i][0]**2)+points[i][1]**2)**(1/2)
            # z x r^2 > 800 and r < 300 - within range 
            # 
            if points[i][3]*r**2 > 800 and (r) < look_ahead:
                # Iterates through the cones 
                
                cones.append(points[i])
    return cones

# Load points data from each .npz file
frames = []
for file_name in sorted(os.listdir(folder_path)):
    if file_name.endswith(".npz"):
        file_path = os.path.join(folder_path, file_name)
        with np.load(file_path, allow_pickle=True) as data:
            # print("Num of points", len(data['points']))
            # print("Size", data['points'][0])
        
            frames.append(data['points'])
        

# Set up the figure and 3D axis for animation
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-15, 15])
ax.set_ylim([-15, 15])
ax.set_zlim([-15, 15])

# Update function for animation
def update(num, frames, ax):
    ax.clear()
    ax.set_xlim([-15, 15])
    ax.set_ylim([-15, 15])
    ax.set_zlim([-15, 15])
    ax.scatter(frames[num][:, 0], frames[num][:, 1], frames[num][:, 2], s=1)
    ax.set_title(f"Frame {num + 1}")
    
    # ax.view_init(elev=20, azim=180)

# Create animation
ani = animations.FuncAnimation(fig, update, frames=len(frames), fargs=(frames, ax), interval=200)
plt.show()
