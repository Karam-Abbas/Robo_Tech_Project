import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter

class HistogramFilter:
    def __init__(self, grid_size, sigma_move=1.0, sigma_sensor=0.5):
        self.grid_size = grid_size
        self.sigma_move = sigma_move
        self.sigma_sensor = sigma_sensor
        self.grid = np.zeros((grid_size, grid_size))
        self.X, self.Y = np.meshgrid(np.linspace(0, grid_size-1, grid_size), np.linspace(0, grid_size-1, grid_size))
        self.prob = None

    def initialize_probability(self, mu_x, mu_y, sigma_x, sigma_y, distribution_type='gaussian'):
        self.initial_sigma_x = sigma_x  # Ensure these attributes are set
        self.initial_sigma_y = sigma_y  # Ensure these attributes are set
        if distribution_type == 'gaussian':
            self.prob = gaussian_2d_probability(self.X, self.Y, mu_x, mu_y, sigma_x, sigma_y)
        elif distribution_type == 'uniform':
            self.prob = np.ones((self.grid_size, self.grid_size))
        self.prob /= self.prob.sum()  # Normalize

    def move_probability(self, move_x, move_y):
        new_prob = np.roll(self.prob, move_y, axis=0)  # Move in Y direction
        new_prob = np.roll(new_prob, move_x, axis=1)  # Move in X direction
        new_prob = gaussian_filter(new_prob, sigma=self.sigma_move)  # Apply Gaussian blur to simulate spread
        self.prob = new_prob / new_prob.sum()  # Normalize

    def sensor_update(self, sensed_x, sensed_y): 
        distances = np.sqrt((self.particles[:, 0] - sensed_x) ** 2 + (self.particles[:, 1] - sensed_y) ** 2)
        self.weights *= np.exp(-distances ** 2 / (2 * self.sigma_sensor ** 2))
        self.weights += 1.e-300 # avoid round-off to zero
        self.weights /= self.weights.sum() # Normalize

    def set_no_go_areas(self, no_go_grid):
        self.grid = no_go_grid
        self.prob[no_go_grid == 1] = 0  # Set probability of no-go areas to 0
        self.prob /= self.prob.sum()  # Normalize

    def plot_probability_distribution(self, ax, title):
        ax.clear()
        sns.heatmap(self.prob, cmap='hot', cbar=False, ax=ax)
        ax.set_ylim(0, self.grid_size)  # Set y-axis limits

class ParticleFilter:
    def __init__(self, num_particles, grid_size, sigma_move=1.0, sigma_sensor=0.5):
        self.num_particles = num_particles
        self.grid_size = grid_size
        self.sigma_move = sigma_move
        self.sigma_sensor = sigma_sensor
        self.particles = np.empty((num_particles, 2))
        self.weights = np.ones(num_particles) / num_particles

    def initialize_particles(self, distribution_type='uniform'):
        if distribution_type == 'uniform':
            self.particles[:, 0] = np.random.uniform(0, self.grid_size, self.num_particles)
            self.particles[:, 1] = np.random.uniform(0, self.grid_size, self.num_particles)
        elif distribution_type == 'gaussian':
            self.particles[:, 0] = np.random.normal(self.grid_size / 2, self.sigma_move, self.num_particles)
            self.particles[:, 1] = np.random.normal(self.grid_size / 2, self.sigma_move, self.num_particles)
            self.particles = np.clip(self.particles, 0, self.grid_size - 1)

    def move_particles(self, move_x, move_y):
        self.particles[:, 0] += move_x + np.random.normal(0, self.sigma_move, self.num_particles)
        self.particles[:, 1] += move_y + np.random.normal(0, self.sigma_move, self.num_particles)
        self.particles = np.clip(self.particles, 0, self.grid_size - 1)

    def sensor_update(self, sensed_x, sensed_y):
        distances = np.sqrt((self.particles[:, 0] - sensed_x) ** 2 + (self.particles[:, 1] - sensed_y) ** 2)
        self.weights *= np.exp(-distances ** 2 / (2 * self.sigma_sensor ** 2))
        self.weights += 1.e-300  # avoid round-off to zero
        self.weights /= self.weights.sum()  # Normalize

    def resample_particles(self):
        indices = np.random.choice(range(self.num_particles), self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.num_particles)

    def plot_particles(self, ax, title):
        ax.clear()
        sns.scatterplot(x=self.particles[:, 0], y=self.particles[:, 1], alpha=0.6, ax=ax)
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.invert_yaxis()

def gaussian_2d_probability(x, y, mu_x, mu_y, sigma_x, sigma_y):
    exponent = -((x - mu_x)**2 / (2 * sigma_x**2) + (y - mu_y)**2 / (2 * sigma_y**2))
    norm_factor = 1 / (2 * np.pi * sigma_x * sigma_y)
    return norm_factor * np.exp(exponent)

class LocalizationVisualizer:
    def __init__(self, grid_size=100, num_particles=1000, filter_type='histogram', distribution_type='gaussian'):
        self.grid_size = grid_size
        self.num_particles = num_particles
        self.filter_type = filter_type
        self.distribution_type = distribution_type
        self.center_x, self.center_y = (grid_size - 1) / 2, (grid_size - 1) / 2
        self.movements = [(15, 0), (0, 15), (-15, 0), (0, -15)]
        self.steps_per_movement = 1
        self.initialize_filter()

    def initialize_filter(self):
        if self.filter_type == 'histogram':
            self.filter = HistogramFilter(self.grid_size, sigma_move=1.5, sigma_sensor=5)
            self.filter.initialize_probability(self.center_x, self.center_y, sigma_x=5, sigma_y=5, distribution_type=self.distribution_type)
            no_go_grid = np.zeros((self.grid_size, self.grid_size))
            no_go_grid[20:40, 20:40] = 1
            self.filter.set_no_go_areas(no_go_grid)
        elif self.filter_type == 'particle':
            self.filter = ParticleFilter(self.num_particles, self.grid_size, sigma_move=1.0, sigma_sensor=5.0)
            self.filter.initialize_particles(distribution_type=self.distribution_type)

    def update(self, frame):
        if frame == 0:
            if self.filter_type == 'histogram':
                self.filter.plot_probability_distribution(self.axes, 'Histogram Filter: Initial Probability Distribution')
            elif self.filter_type == 'particle':
                self.filter.plot_particles(self.axes, 'Particle Filter: Initial Particle Distribution')
        else:
            self.axes.cla()
            step = (frame - 1) // self.steps_per_movement
            sub_step = (frame - 1) % self.steps_per_movement

            if step < len(self.movements):
                move_x, move_y = self.movements[step]
                move_x = int(move_x / self.steps_per_movement)
                move_y = int(move_y / self.steps_per_movement)

                if self.filter_type == 'histogram':
                    self.filter.move_probability(move_x, move_y)
                    self.filter.sensor_update(self.center_x + move_x * sub_step, self.center_y + move_y * sub_step)
                    self.filter.plot_probability_distribution(self.axes, f'Histogram Filter: Step {frame}')
                elif self.filter_type == 'particle':
                    self.filter.move_particles(move_x, move_y)
                    self.filter.sensor_update(self.center_x + move_x * sub_step, self.center_y + move_y * sub_step)
                    self.filter.resample_particles()
                    self.filter.plot_particles(self.axes, f'Particle Filter: Step {frame}')
            
            self.axes.set_xlim(0, self.grid_size - 1)
            self.axes.set_ylim(0, self.grid_size - 1)
            plt.draw()

    def visualize(self):
        fig, self.axes = plt.subplots(1, 1, figsize=(7, 7))
        ani = FuncAnimation(fig, self.update, frames=(len(self.movements) * self.steps_per_movement) + 1, repeat=False, interval=1000)
        plt.tight_layout()
        plt.show()

