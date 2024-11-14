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

VisualizeMap(map_data)