import csv
import random

# Set the size of the map
rows, cols = 600, 600

# Generate the map with 1s and 0s randomly (you can adjust the probability as needed)
map_data = [[random.choice([0, 1]) for _ in range(cols)] for _ in range(rows)]

# Save the map to a CSV file
with open('map.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(map_data)

print("Map saved to map.csv")
