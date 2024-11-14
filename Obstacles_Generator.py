import csv
import matplotlib.pyplot as plt
import numpy as np
import random

# Set map size and initialize with all 0s (paths)
rows, cols = 600, 600
map_data = [[0 for _ in range(cols)] for _ in range(rows)]

# Define a function to create obstacle clusters without overlap
def create_obstacle_cluster(map_data, center_x, center_y, cluster_size):
    for i in range(-cluster_size, cluster_size):
        for j in range(-cluster_size, cluster_size):
            x, y = center_x + i, center_y + j
            # Check if within bounds and assign obstacle
            if 0 <= x < rows and 0 <= y < cols:
                map_data[x][y] = 1

# Check if a new cluster will overlap with any existing obstacles
def is_overlap(map_data, center_x, center_y, cluster_size):
    for i in range(-cluster_size, cluster_size):
        for j in range(-cluster_size, cluster_size):
            x, y = center_x + i, center_y + j
            if 0 <= x < rows and 0 <= y < cols and map_data[x][y] == 1:
                return True
    return False



# Generate clusters with dynamic sizes without overlap
def Generate_Map():
    num_clusters = 20
    min_cluster_size = 10  # Minimum size of a cluster
    max_cluster_size = 45  # Maximum size of a cluster
    attempts = 0

    for _ in range(num_clusters):
        placed = False
        while not placed and attempts < 100:  # Limit attempts to avoid infinite loop
            center_x = random.randint(max_cluster_size, rows - max_cluster_size - 1)
            center_y = random.randint(max_cluster_size, cols - max_cluster_size - 1)

            # Randomly choose a size for this cluster within the specified range
            cluster_size = random.randint(min_cluster_size, max_cluster_size)

            # Place cluster if there's no overlap
            if not is_overlap(map_data, center_x, center_y, cluster_size):
                create_obstacle_cluster(map_data, center_x, center_y, cluster_size)
                placed = True
            attempts += 1
            # Save to CSV
            with open('map.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(map_data)
    return map_data

# Load and visualize the map
def VisualizeMap(map_data):
    map_array = np.array(map_data)
    plt.figure(figsize=(10, 10))
    plt.imshow(map_array, cmap='Greys', interpolation='nearest')
    plt.title("Map Visualization")
    plt.show()

