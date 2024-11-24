from Obstacles_Generator import Generate_Map,VisualizeMap
import numpy as np
import pandas as pd
import csv
# map_data = Generate_Map()
# Load the map from the CSV file

with open('map.csv', 'r') as file:
    reader = csv.reader(file)
    map_data = [[int(cell) for cell in row] for row in reader]

# Convert map data to a numpy array for easier manipulation and visualization
map_grid =np.array(map_data)

grid_size = 600

data = []

# Helper function to calculate distance to nearest obstacle in a given direction
def find_distance_to_obstacle(x, y, dx, dy):
    distance = 0
    while 0 <= x < grid_size and 0 <= y < grid_size:
        x += dx
        y += dy
        distance += 1
        if not (0 <= x < grid_size and 0 <= y < grid_size) or map_grid[x, y] == 1:
            break
    return distance if 0 <= x < grid_size and 0 <= y < grid_size else grid_size  # Max if no obstacle found

# Populate data with distances and obstacle status
for x in range(grid_size):
    for y in range(grid_size):
        obstacle_status = map_grid[x, y]
        dist_above = find_distance_to_obstacle(x, y, -1, 0)  # Up
        dist_below = find_distance_to_obstacle(x, y, 1, 0)   # Down
        dist_left = find_distance_to_obstacle(x, y, 0, -1)   # Left
        dist_right = find_distance_to_obstacle(x, y, 0, 1)   # Right
        data.append([obstacle_status, dist_above, dist_below, dist_left, dist_right])

# Convert data to a DataFrame and save as CSV
df = pd.DataFrame(data)
df.to_csv('map_data.csv', index=False, header=False)

# Add noise to the distances
def add_noise_to_distances(data, noise_range=(-1, 1), valid_range=(0, 600)):
    noisy_data = []
    for row in data:
        obstacle_status, dist_above, dist_below, dist_left, dist_right = row
        # Generate noise for each distance measurement
        noise = np.random.uniform(noise_range[0], noise_range[1], size=4)
        # Add noise to distances and clip to valid range
        noisy_distances = np.clip(
            [dist_above + noise[0], dist_below + noise[1], dist_left + noise[2], dist_right + noise[3]],
            valid_range[0],
            valid_range[1]
        )
        noisy_data.append([obstacle_status] + noisy_distances.tolist())
    return noisy_data

# Apply noise to the dataset
noisy_data = add_noise_to_distances(data)

# Convert noisy data to a DataFrame and save as CSV
df_noisy = pd.DataFrame(noisy_data)
df_noisy.to_csv('map_data_noisy.csv', index=False, header=False)

# Visualize noisy data (if needed, by processing the noisy distances back into a map visualization)
VisualizeMap(noisy_data)
VisualizeMap(map_data)