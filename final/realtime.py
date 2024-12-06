import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter

# Histogram Filter
class HistogramFilter:
    def __init__(self, grid_size, sigma_move=1.0, sigma_sensor=0.5):
        self.grid_size = grid_size
        self.sigma_move = sigma_move
        self.sigma_sensor = sigma_sensor
        self.grid = np.zeros((grid_size, grid_size))
        self.X, self.Y = np.meshgrid(np.linspace(0, grid_size-1, grid_size), np.linspace(0, grid_size-1, grid_size))
        self.prob = None

    def initialize_probability(self, mu_x, mu_y, sigma_x, sigma_y):
        self.initial_sigma_x = sigma_x
        self.initial_sigma_y = sigma_y
        self.prob = gaussian_2d_probability(self.X, self.Y, mu_x, mu_y, sigma_x, sigma_y)
        self.prob /= self.prob.sum()  # Normalize

    def move_probability(self, move_x, move_y):
        new_prob = np.roll(self.prob, move_y, axis=0)  # Move in Y direction
        new_prob = np.roll(new_prob, move_x, axis=1)  # Move in X direction
        new_prob = gaussian_filter(new_prob, sigma=self.sigma_move)  # Apply Gaussian blur to simulate spread
        self.prob = new_prob / new_prob.sum()  # Normalize

    def sensor_update(self, sensed_x, sensed_y):
        update_prob = gaussian_2d_probability(self.X, self.Y, mu_x=sensed_x, mu_y=sensed_y, sigma_x=self.initial_sigma_x, sigma_y=self.initial_sigma_y)
        self.prob *= update_prob
        self.prob /= self.prob.sum()  # Normalize

    def set_no_go_areas(self, no_go_grid):
        self.grid = no_go_grid
        self.prob[no_go_grid == 1] = 0  # Set probability of no-go areas to 0
        self.prob /= self.prob.sum()  # Normalize

    def plot_probability_distribution(self, ax, title):
        ax.clear()
        sns.heatmap(self.prob, cmap='hot', cbar=False, ax=ax)
        ax.set_ylim(0, self.grid_size)  # Set y-axis limits

# Particle Filter
class ParticleFilter:
    def __init__(self, num_particles, grid_size, sigma_move=1.0, sigma_sensor=0.5):
        self.num_particles = num_particles
        self.grid_size = grid_size
        self.sigma_move = sigma_move
        self.sigma_sensor = sigma_sensor
        self.particles = np.empty((num_particles, 2))
        self.weights = np.ones(num_particles) / num_particles

    def initialize_particles(self):
        self.particles[:, 0] = np.random.uniform(0, self.grid_size, self.num_particles)
        self.particles[:, 1] = np.random.uniform(0, self.grid_size, self.num_particles)

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


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Example usage
grid_size = 100

# # Histogram Filter
# hf = HistogramFilter(grid_size, sigma_move=1.5, sigma_sensor=5)
# center_x, center_y = (grid_size - 1) / 2, (grid_size - 1) / 2
# hf.initialize_probability(center_x, center_y, sigma_x=5, sigma_y=5)
# no_go_grid = np.zeros((grid_size, grid_size))
# no_go_grid[20:40, 20:40] = 1
# hf.set_no_go_areas(no_go_grid)

# # Particle Filter (commented out)
# # num_particles = 1000
# # pf = ParticleFilter(num_particles, grid_size, sigma_move=1.0, sigma_sensor=5.0)
# # pf.initialize_particles()

# # Create subplots
# fig, axes = plt.subplots(1, 1, figsize=(7, 7))  # Single subplot, axes is a single object

# # Plot initial distributions
# hf.plot_probability_distribution(axes, 'Histogram Filter: Initial Probability Distribution')
# # pf.plot_particles(axes, 'Particle Filter: Initial Particle Distribution')  # If needed

# # Movements and animation configuration
# movements = [(15, 15), (15, -15), (-15, 15), (-15, -15)]
# steps_per_movement = 1

# def update(frame):
#     # Clear the axes to update the plot
#     axes.cla()  # Clear the current axis

#     # Calculate movement step
#     step = frame // steps_per_movement
#     sub_step = frame % steps_per_movement

#     if step < len(movements):
#         move_x, move_y = movements[step]
#         move_x = int(move_x / steps_per_movement)  # Convert to integer
#         move_y = int(move_y / steps_per_movement)  # Convert to integer

#         # Update Histogram Filter
#         hf.move_probability(move_x, move_y)
#         hf.sensor_update(center_x + move_x * sub_step, center_y + move_y * sub_step)

#         # Plot Histogram Filter
#         hf.plot_probability_distribution(axes, f'Histogram Filter: Step {frame + 1}')
    
#     # Set axis limits after clearing (so the axes don't reset their range)
#     axes.set_xlim(0, grid_size - 1)
#     axes.set_ylim(0, grid_size - 1)
    
#     # Refresh the plot
#     plt.draw()

# # Animation
# ani = FuncAnimation(fig, update, frames=len(movements) * steps_per_movement, repeat=False, interval=10000)
# plt.tight_layout()
# plt.show()

# Example usage
grid_size = 100

center_x, center_y = (grid_size - 1) / 2, (grid_size - 1) / 2
# Particle Filter Example usage
num_particles = 1000
pf = ParticleFilter(num_particles, grid_size, sigma_move=1.0, sigma_sensor=5.0)
pf.initialize_particles()

# Create subplots
fig, axes = plt.subplots(1, 1, figsize=(7, 7))  # Single subplot, axes is a single object

# Plot initial distribution
pf.plot_particles(axes, 'Particle Filter: Initial Particle Distribution')

# Movements and animation configuration
movements = [(15, 15), (15, -15), (-15, 15), (-15, -15)]
steps_per_movement = 1

def update(frame):
    if frame == 0:
        # Initial frame, plot initial distribution
        pf.plot_particles(axes, 'Particle Filter: Initial Particle Distribution')
    else:
        # Clear the axes to update the plot
        axes.cla()  # Clear the current axis

        # Calculate movement step
        step = (frame - 1) // steps_per_movement
        sub_step = (frame - 1) % steps_per_movement

        if step < len(movements):
            move_x, move_y = movements[step]
            move_x = int(move_x / steps_per_movement)  # Convert to integer
            move_y = int(move_y / steps_per_movement)  # Convert to integer

            # Update Particle Filter
            pf.move_particles(move_x, move_y)
            pf.sensor_update(center_x + move_x * sub_step, center_y + move_y * sub_step)
            pf.resample_particles()

            # Plot Particle Filter
            pf.plot_particles(axes, f'Particle Filter: Step {frame}')

    # Set axis limits after clearing (so the axes don't reset their range)
    axes.set_xlim(0, grid_size - 1)
    axes.set_ylim(0, grid_size - 1)
    
    # Refresh the plot
    plt.draw()

# Animation
ani = FuncAnimation(fig, update, frames=(len(movements) * steps_per_movement) + 1, repeat=False, interval=1000)
plt.tight_layout()
plt.show()

