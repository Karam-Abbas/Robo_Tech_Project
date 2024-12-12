import numpy as np
import pandas as pd  # Import pandas to handle CSV data
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter
import random
from tqdm import tqdm
import csv

# Other supporting classes: MapHandler, RobotHandler, and BeliefHandler would be used
# Integrate the histogram and particle filters for visualization of the robot path

def gaussian_2d_probability(x, y, mu_x, mu_y, sigma_x, sigma_y):
    exponent = -((x - mu_x)**2 / (2 * sigma_x**2) + (y - mu_y)**2 / (2 * sigma_y**2))
    norm_factor = 1 / (2 * np.pi * sigma_x * sigma_y)
    return norm_factor * np.exp(exponent)

class HistogramFilter:
    def __init__(self, grid_size, sigma_move=1.0, sigma_sensor=0.5):
        self.grid_size = grid_size
        self.sigma_move = sigma_move
        self.sigma_sensor = sigma_sensor
        self.grid = np.zeros((grid_size, grid_size))
        self.X, self.Y = np.meshgrid(
            np.linspace(0, grid_size-1, grid_size), 
            np.linspace(0, grid_size-1, grid_size))
        self.prob = None
        self.noisy_data = None  # Placeholder for noisy data

    def load_noisy_data(self, filepath):
        """Load noisy sensor data from a CSV file."""
        self.noisy_data = pd.read_csv(filepath, header=None).values

    def initialize_probability(self, mu_x, mu_y, sigma_x, sigma_y, distribution_type='gaussian'):
        self.initial_sigma_x = sigma_x
        self.initial_sigma_y = sigma_y
        if distribution_type == 'gaussian':
            self.prob = gaussian_2d_probability(self.X, self.Y, mu_x, mu_y, sigma_x, sigma_y)
        elif distribution_type == 'uniform':
            self.prob = np.ones((self.grid_size, self.grid_size))
        self.prob /= self.prob.sum()

    def move_probability(self, move_x, move_y):
        new_prob = np.roll(self.prob, move_y, axis=0)
        new_prob = np.roll(new_prob, move_x, axis=1)
        new_prob = gaussian_filter(new_prob, sigma=self.sigma_move)
        self.prob = new_prob / new_prob.sum()

    def sensor_update(self, robot_x, robot_y):
        """Update beliefs using sensor readings from noisy data."""
        # Get the 1D index for the current position
        index = 600 * int(robot_x) + int(robot_y)
        noisy_readings = self.noisy_data[index]
        
        # Extract distances from noisy readings
        dist_above, dist_below, dist_left, dist_right = noisy_readings[1:]

        # Use distances to adjust likelihoods for adjacent cells
        likelihood = np.zeros_like(self.prob)

        if robot_x > 0:
            likelihood[robot_x - 1, robot_y] += np.exp(-dist_above / self.sigma_sensor)
        if robot_x < self.grid_size - 1:
            likelihood[robot_x + 1, robot_y] += np.exp(-dist_below / self.sigma_sensor)
        if robot_y > 0:
            likelihood[robot_x, robot_y - 1] += np.exp(-dist_left / self.sigma_sensor)
        if robot_y < self.grid_size - 1:
            likelihood[robot_x, robot_y + 1] += np.exp(-dist_right / self.sigma_sensor)

        # Combine likelihood with current belief
        self.prob *= likelihood
        self.prob /= self.prob.sum()

    def visualize_movement(self, robot_path):
        """Visualize robot movement with the histogram filter."""
        fig, ax = plt.subplots(figsize=(10, 8))

        def update(frame):
            x, y = robot_path[frame]
            self.move_probability(0, 0)  # No movement; just visualize sensor update
            self.sensor_update(x, y)
            ax.clear()
            sns.heatmap(self.prob, cmap='hot', cbar=False, ax=ax)
            ax.scatter(y, x, color='blue', s=100, label='Robot')
            ax.set_title(f"Histogram Filter - Step {frame + 1}")
            ax.legend()

        ani = FuncAnimation(fig, update, frames=len(robot_path), repeat=False)
        plt.show()

class ParticleFilter:
    def __init__(self, num_particles, grid_size, sigma_move=1.0, sigma_sensor=0.5):
        self.num_particles = num_particles
        self.grid_size = grid_size
        self.sigma_move = sigma_move
        self.sigma_sensor = sigma_sensor
        self.particles = np.empty((num_particles, 2))
        self.weights = np.ones(num_particles) / num_particles
        self.noisy_data = None  # Placeholder for noisy data

    def load_noisy_data(self, filepath):
        """Load noisy sensor data from a CSV file."""
        self.noisy_data = pd.read_csv(filepath, header=None).values

    def move_particles(self, move_x, move_y):
        self.particles[:, 0] += move_x + np.random.normal(0, self.sigma_move, self.num_particles)
        self.particles[:, 1] += move_y + np.random.normal(0, self.sigma_move, self.num_particles)
        self.particles = np.clip(self.particles, 0, self.grid_size - 1)

    def sensor_update(self):
        """Update particle weights using noisy sensor data."""
        for i, (x, y) in enumerate(self.particles):
            # Get the 1D index for the current position
            index = 600 * int(x) + int(y)
            noisy_readings = self.noisy_data[index]

            # Extract distances from noisy readings
            dist_above, dist_below, dist_left, dist_right = noisy_readings[1:]

            # Adjust weights based on proximity to obstacles
            weight = 1.0
            if x > 0:
                weight *= np.exp(-dist_above / self.sigma_sensor)
            if x < self.grid_size - 1:
                weight *= np.exp(-dist_below / self.sigma_sensor)
            if y > 0:
                weight *= np.exp(-dist_left / self.sigma_sensor)
            if y < self.grid_size - 1:
                weight *= np.exp(-dist_right / self.sigma_sensor)

            self.weights[i] *= weight
        
        self.weights += 1.e-300
        self.weights /= self.weights.sum()

    def resample_particles(self):
        indices = np.random.choice(range(self.num_particles), self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.num_particles)

    def visualize_movement(self, robot_path):
        """Visualize robot movement with the particle filter."""
        fig, ax = plt.subplots(figsize=(10, 8))

        def update(frame):
            x, y = robot_path[frame]
            self.move_particles(0, 0)  # No movement; just visualize sensor update
            self.sensor_update()
            self.resample_particles()
            ax.clear()
            sns.scatterplot(x=self.particles[:, 1], y=self.particles[:, 0], alpha=0.6, ax=ax, label='Particles')
            ax.scatter(y, x, color='blue', s=100, label='Robot')
            ax.set_title(f"Particle Filter - Step {frame + 1}")
            ax.legend()

        ani = FuncAnimation(fig, update, frames=len(robot_path), repeat=False)
        plt.show()

# Example usage (assuming map_data_noisy.csv exists):
hf = HistogramFilter(grid_size=600, sigma_move=1.0, sigma_sensor=0.5)
hf.load_noisy_data("map_data_noisy.csv")
hf.initialize_probability(mu_x=300, mu_y=300, sigma_x=5, sigma_y=5)

pf = ParticleFilter(num_particles=1000, grid_size=600, sigma_move=1.0, sigma_sensor=0.5)
pf.load_noisy_data("map_data_noisy.csv")

# Robot path (example path: [(x1, y1), (x2, y2), ...])
robot_path = [(300, 300), (305, 310), (310, 320), (320, 330)]

# Visualize movement
hf.visualize_m
