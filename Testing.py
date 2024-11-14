import numpy as np

# Initialize a 600x600 map with obstacle and path indicators
rows, cols = 600, 600
map_data = np.random.choice([0, 1], (rows, cols), p=[0.6, 0.4])  # 80% paths, 20% obstacles

# Initialize a 3D array to store each pixel's data: [obstacle, dist_up, dist_down, dist_right, dist_left]
map_with_info = np.zeros((rows, cols, 5), dtype=int)
map_with_info[:, :, 0] = map_data  # Set obstacle data from map_data

# Fill distances with a large number initially
max_distance = rows * cols  # Maximum possible distance
map_with_info[:, :, 1:] = max_distance

# First pass: Top-left to Bottom-right
for i in range(rows):
    for j in range(cols):
        if map_data[i, j] == 1:  # Obstacle
            map_with_info[i, j, 1:] = 0  # No distance to itself
        else:
            # Check above
            if i > 0:
                map_with_info[i, j, 1] = map_with_info[i - 1, j, 1] + 1
            # Check left
            if j > 0:
                map_with_info[i, j, 4] = map_with_info[i, j - 1, 4] + 1

# Second pass: Bottom-right to Top-left
for i in range(rows - 1, -1, -1):
    for j in range(cols - 1, -1, -1):
        if map_data[i, j] == 0:  # Only calculate for free path
            # Check below
            if i < rows - 1:
                map_with_info[i, j, 2] = min(map_with_info[i, j, 2], map_with_info[i + 1, j, 2] + 1)
            # Check right
            if j < cols - 1:
                map_with_info[i, j, 3] = min(map_with_info[i, j, 3], map_with_info[i, j + 1, 3] + 1)

# map_with_info now contains the data in the form [obstacle, dist_up, dist_down, dist_right, dist_left] for each pixel
print("Map with additional information is ready.")


np.savetxt('obstacles.csv', map_with_info[:, :, 0], delimiter=',', fmt='%d')    # Obstacle layer
np.savetxt('dist_up.csv', map_with_info[:, :, 1], delimiter=',', fmt='%d')      # Distance up layer
np.savetxt('dist_down.csv', map_with_info[:, :, 2], delimiter=',', fmt='%d')    # Distance down layer
np.savetxt('dist_right.csv', map_with_info[:, :, 3], delimiter=',', fmt='%d')   # Distance right layer
np.savetxt('dist_left.csv', map_with_info[:, :, 4], delimiter=',', fmt='%d')    # Distance left layer
