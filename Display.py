import csv
import matplotlib.pyplot as plt
import numpy as np

# Load the map from the CSV file
with open('map.csv', 'r') as file:
    reader = csv.reader(file)
    map_data = [[int(cell) for cell in row] for row in reader]

# Convert map_data to a NumPy array for easier manipulation
map_array = np.array(map_data)

# Plot the map
plt.figure(figsize=(10, 10))
plt.imshow(map_array, cmap='Greys', interpolation='nearest')
#plt.colorbar(label="0 = Path, 1 = Obstacle")  # Adds a legend
plt.title("Map Visualization")
plt.show()
