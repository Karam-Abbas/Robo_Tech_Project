import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


class MotionModel:
    def __init__(self, grid_size=10, num_particles=500, sigma_move=1.0, sigma_sensor=1.0):
        self.grid_size = grid_size
        self.num_particles = num_particles
        self.sigma_move = sigma_move
        self.sigma_sensor = sigma_sensor

        # Initialize map, histogram belief, and particles
        self.map_grid = np.zeros((grid_size, grid_size))  # 0 = free space, 1 = obstacle
        self.histogram_belief = np.ones((grid_size, grid_size)) / (grid_size ** 2)  # Uniform belief
        self.particles = np.random.uniform(0, grid_size, (num_particles, 2))  # Randomly placed particles
        self.weights = np.ones(num_particles) / num_particles

    def set_obstacles(self, obstacle_positions):
        """Set obstacles on the map."""
        for pos in obstacle_positions:
            self.map_grid[pos] = 1
        self.histogram_belief[self.map_grid == 1] = 0
        self.histogram_belief /= self.histogram_belief.sum()

    def motion_update(self, move):
        """Update beliefs and particles based on motion."""
        dx, dy = move

        # Update histogram belief
        new_belief = np.roll(self.histogram_belief, dy, axis=0)
        new_belief = np.roll(new_belief, dx, axis=1)
        self.histogram_belief = gaussian_filter(new_belief, sigma=self.sigma_move)
        self.histogram_belief[self.map_grid == 1] = 0
        self.histogram_belief /= self.histogram_belief.sum()

        # Update particles
        self.particles[:, 0] += dx + np.random.normal(0, self.sigma_move, self.num_particles)
        self.particles[:, 1] += dy + np.random.normal(0, self.sigma_move, self.num_particles)
        self.particles = np.clip(self.particles, 0, self.grid_size - 1)

    def sensor_update(self, sensor_readings):
        """Refine beliefs and particles based on sensor readings."""
        for direction, distance in sensor_readings.items():
            # Update histogram belief
            mask = np.zeros_like(self.histogram_belief)
            if distance == 0:  # Skip if distance is 0
                continue
            if direction == "E" and distance < self.grid_size:
                mask[:, :-distance] = self.map_grid[:, distance:]
            elif direction == "W" and distance < self.grid_size:
                mask[:, distance:] = self.map_grid[:, :-distance]
            elif direction == "N" and distance < self.grid_size:
                mask[distance:, :] = self.map_grid[:-distance, :]
            elif direction == "S" and distance < self.grid_size:
                mask[:-distance, :] = self.map_grid[distance:, :]

            likelihood = 1 - mask  # Invert: 1 for free, 0 for obstacles
            self.histogram_belief *= likelihood
            self.histogram_belief /= self.histogram_belief.sum()

            # Update particle weights
            if direction == "E":
                distances = self.grid_size - self.particles[:, 1] - 1
            elif direction == "W":
                distances = self.particles[:, 1]
            elif direction == "N":
                distances = self.particles[:, 0]
            elif direction == "S":
                distances = self.grid_size - self.particles[:, 0] - 1

            self.weights *= np.exp(-((distances - distance) ** 2) / (2 * self.sigma_sensor ** 2))
            self.weights += 1.e-300  # Avoid zero weights
            self.weights /= self.weights.sum()

        # Resample particles
        indices = np.random.choice(range(self.num_particles), self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.num_particles)

    def get_estimated_position(self):
        """Return estimated positions from histogram and particles."""
        hist_est = np.unravel_index(np.argmax(self.histogram_belief), self.histogram_belief.shape)
        part_est = np.mean(self.particles, axis=0)
        return {"histogram": hist_est, "particle": part_est}


# Visualization Function
def visualize_beliefs(motion_model, step, move, sensor_readings):
    """
    Visualize the map, histogram belief, particles, and estimated positions.

    Args:
        motion_model: The MotionModel instance.
        step: The current step number.
        move: The move (dx, dy) for this step.
        sensor_readings: The sensor readings for this step.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Visualize the histogram belief as a heatmap
    ax[0].imshow(motion_model.histogram_belief, cmap="hot", interpolation="nearest")
    ax[0].set_title(f"Histogram Belief (Step {step})")
    ax[0].set_xlabel("X-axis")
    ax[0].set_ylabel("Y-axis")
    ax[0].invert_yaxis()  # Flip Y-axis to match grid convention
    for i in range(motion_model.grid_size):
        for j in range(motion_model.grid_size):
            ax[0].text(j, i, f"{motion_model.histogram_belief[i, j]:.2f}", ha="center", va="center", color="white")

    # Visualize the particles
    ax[1].imshow(motion_model.map_grid, cmap="Greys", interpolation="nearest")
    particles = motion_model.particles
    ax[1].scatter(particles[:, 1], particles[:, 0], c="blue", s=10, label="Particles")
    ax[1].set_title(f"Particles and Estimated Position (Step {step})")
    ax[1].set_xlabel("X-axis")
    ax[1].set_ylabel("Y-axis")
    ax[1].invert_yaxis()

    # Estimated positions
    estimates = motion_model.get_estimated_position()
    hist_est = estimates["histogram"]
    part_est = estimates["particle"]
    ax[1].scatter(hist_est[1], hist_est[0], c="red", s=100, label="Histogram Estimate")
    ax[1].scatter(part_est[1], part_est[0], c="green", s=100, label="Particle Filter Estimate")

    ax[1].legend()
    plt.suptitle(f"Step {step}: Move {move}, Sensor {sensor_readings}")
    plt.tight_layout()
    plt.show()


# Testing the MotionModel
motion_model = MotionModel(grid_size=10, num_particles=500, sigma_move=1.0, sigma_sensor=2.0)
motion_model.set_obstacles([(3, 3), (3, 4), (4, 3), (4, 4)])  # Set obstacles

# Define the test scenario
moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Move right, down, left, up
sensor_readings = [
    {"E": 3, "W": 1, "N": 2, "S": 4},  # Example sensor readings
    {"E": 4, "W": 0, "N": 3, "S": 2},
    {"E": 1, "W": 2, "N": 4, "S": 3},
    {"E": 2, "W": 3, "N": 1, "S": 4},
]

# Run the simulation and visualize
for i, move in enumerate(moves):
    print(f"\nStep {i + 1}: Move {move}")
    motion_model.motion_update(move)
    motion_model.sensor_update(sensor_readings[i])
    visualize_beliefs(motion_model, i + 1, move, sensor_readings[i])
