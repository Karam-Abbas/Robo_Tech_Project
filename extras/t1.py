import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter
import random
import csv
import pandas as pd
from tqdm import tqdm

# Combine core functionalities from both files

class MapHandler:
    def __init__(self, rows=600, cols=600):
        self.rows = rows
        self.cols = cols
        self.map_data = [[0 for _ in range(cols)] for _ in range(rows)]

    def initialize_from_csv(self, filename):
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            self.map_data = [list(map(int, row)) for row in reader]
        self.rows = len(self.map_data)
        self.cols = len(self.map_data[0]) if self.rows > 0 else 0

    def visualize_map(self):
        plt.imshow(np.array(self.map_data), cmap='Greys', interpolation='nearest')
        plt.gca().invert_yaxis()
        plt.title("Map Visualization")
        plt.show()


class RobotHandler:
    def __init__(self, map_handler):
        self.map_handler = map_handler
        self.robot_position = None
        self.robot_orientation = None

    def set_initial_position(self):
        while True:
            x, y = random.randint(0, self.map_handler.rows - 1), random.randint(0, self.map_handler.cols - 1)
            if self.map_handler.map_data[x][y] == 0:
                self.robot_position = (x, y)
                self.robot_orientation = random.choice(['N', 'S', 'E', 'W'])
                break

    def visualize_robot(self):
        map_array = np.array(self.map_handler.map_data)
        plt.imshow(map_array, cmap='Greys', interpolation='nearest')
        plt.gca().invert_yaxis()
        if self.robot_position:
            rx, ry = self.robot_position
            plt.scatter(ry, rx, color='blue', label='Robot', s=100)
        plt.title("Map with Robot")
        plt.legend()
        plt.show()


class HistogramFilter:
    def __init__(self, grid_size, sigma_move=1.0, sigma_sensor=0.5):
        self.grid_size = grid_size
        self.sigma_move = sigma_move
        self.sigma_sensor = sigma_sensor
        self.prob = np.ones((grid_size, grid_size)) / (grid_size * grid_size)

    def move_probability(self, move_x, move_y):
        new_prob = np.roll(self.prob, move_y, axis=0)
        new_prob = np.roll(new_prob, move_x, axis=1)
        new_prob = gaussian_filter(new_prob, sigma=self.sigma_move)
        self.prob = new_prob / new_prob.sum()

    def sensor_update(self, sensed_x, sensed_y):
        x, y = np.meshgrid(range(self.grid_size), range(self.grid_size))
        sensor_prob = np.exp(-((x - sensed_x)**2 + (y - sensed_y)**2) / (2 * self.sigma_sensor**2))
        self.prob *= sensor_prob
        self.prob += 1.e-300  # Avoid zeros
        self.prob /= self.prob.sum()

    def visualize_probability(self):
        sns.heatmap(self.prob, cmap='hot')
        plt.title("Histogram Filter Probability Distribution")
        plt.show()


class ParticleFilter:
    def __init__(self, num_particles, grid_size, sigma_move=1.0, sigma_sensor=0.5):
        self.num_particles = num_particles
        self.grid_size = grid_size
        self.sigma_move = sigma_move
        self.sigma_sensor = sigma_sensor
        self.particles = np.random.uniform(0, grid_size, (num_particles, 2))

    def move_particles(self, move_x, move_y):
        noise = np.random.normal(0, self.sigma_move, self.particles.shape)
        self.particles += [move_x, move_y] + noise
        self.particles = np.clip(self.particles, 0, self.grid_size - 1)

    def sensor_update(self, sensed_x, sensed_y):
        distances = np.linalg.norm(self.particles - np.array([sensed_x, sensed_y]), axis=1)
        weights = np.exp(-distances**2 / (2 * self.sigma_sensor**2))
        weights /= weights.sum()
        indices = np.random.choice(range(self.num_particles), self.num_particles, p=weights)
        self.particles = self.particles[indices]

    def visualize_particles(self):
        sns.scatterplot(x=self.particles[:, 0], y=self.particles[:, 1], alpha=0.6)
        plt.title("Particle Filter Distribution")
        plt.show()


# Main simulation function
def run_simulation():
    map_handler = MapHandler()
    map_handler.initialize_from_csv('map.csv')

    robot_handler = RobotHandler(map_handler)
    robot_handler.set_initial_position()

    # Choose filter type: histogram or particle
    use_histogram = True
    if use_histogram:
        filter_model = HistogramFilter(grid_size=map_handler.rows)
    else:
        filter_model = ParticleFilter(num_particles=1000, grid_size=map_handler.rows)

    for step in range(10):
        # Simulate motion
        move_x, move_y = random.randint(-5, 5), random.randint(-5, 5)
        if use_histogram:
            filter_model.move_probability(move_x, move_y)
        else:
            filter_model.move_particles(move_x, move_y)

        # Simulate sensing
        sensed_x, sensed_y = robot_handler.robot_position
        if use_histogram:
            filter_model.sensor_update(sensed_x, sensed_y)
        else:
            filter_model.sensor_update(sensed_x, sensed_y)

        # Visualize
        if use_histogram:
            filter_model.visualize_probability()
        else:
            filter_model.visualize_particles()

# Run the integrated system
run_simulation()
