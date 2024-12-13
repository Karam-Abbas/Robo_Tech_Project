import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter
import csv
import random
from tqdm import tqdm
import re

class MapHandler:
    def __init__(self, rows=600, cols=600):
        self.rows = rows
        self.cols = cols
        self.map_data = [[0 for _ in range(cols)] for _ in range(rows)]
    
    def initialize_from_csv(self, filename):
        """
        Initialize the MapHandler's rows, cols, and map_data from a CSV file.
        """
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            map_data = [list(map(int, row)) for row in reader]

        # Update rows, cols, and map_data
        self.rows = len(map_data)
        self.cols = len(map_data[0]) if self.rows > 0 else 0
        self.map_data = map_data

    def create_obstacle_cluster(self, center_x, center_y, cluster_size):
        """Create a cluster of obstacles."""
        for i in range(-cluster_size, cluster_size):
            for j in range(-cluster_size, cluster_size):
                x, y = center_x + i, center_y + j
                if 0 <= x < self.rows and 0 <= y < self.cols:
                    self.map_data[x][y] = 1

    def is_overlap(self, center_x, center_y, cluster_size):
        """Check if a cluster overlaps with existing obstacles."""
        for i in range(-cluster_size, cluster_size):
            for j in range(-cluster_size, cluster_size):
                x, y = center_x + i, center_y + j
                if 0 <= x < self.rows and 0 <= y < self.cols and self.map_data[x][y] == 1:
                    return True
        return False

    def generate_map(self, num_clusters=5, min_cluster_size=30, max_cluster_size=45):
        """Generate a map with clusters of obstacles."""
        attempts = 0
        progress = tqdm(total=num_clusters, desc="Generating Map")
        for _ in range(num_clusters):
            placed = False
            while not placed and attempts < 100:
                center_x = random.randint(max_cluster_size, self.rows - max_cluster_size - 1)
                center_y = random.randint(max_cluster_size, self.cols - max_cluster_size - 1)
                cluster_size = random.randint(min_cluster_size, max_cluster_size)
                if not self.is_overlap(center_x, center_y, cluster_size):
                    self.create_obstacle_cluster(center_x, center_y, cluster_size)
                    placed = True
                attempts += 1
            progress.update(1)
        progress.close()
        self.save_map_to_csv('map.csv')

    def save_map_to_csv(self, filename):
        """Save the map to a CSV file."""
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self.map_data)

    def load_map_from_csv(self, filename):
        """Load the map from a CSV file."""
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            self.map_data = [[int(cell) for cell in row] for row in reader]

    def visualize_map(self):
        """Visualize the map."""
        map_array = np.array(self.map_data)
        plt.figure(figsize=(10, 10))
        plt.imshow(map_array, cmap='Greys', interpolation='nearest')
        plt.gca().invert_yaxis()
        plt.title("Map Visualization")
        plt.show()

    def find_distance_to_obstacle(self, x, y, dx, dy):
        """Calculate distance to the nearest obstacle in a given direction."""
        distance = 0
        while 0 <= x < self.rows and 0 <= y < self.cols:
            x += dx
            y += dy
            distance += 1
            if not (0 <= x < self.rows and 0 <= y < self.cols) or self.map_data[x][y] == 1:
                break
        return distance if 0 <= x < self.rows and 0 <= y < self.cols else self.rows

    def calculate_distances(self):
        """Calculate distances to obstacles and store in a CSV."""
        data = []
        progress = tqdm(total=self.rows * self.cols, desc="Calculating Distances")
        for x in range(self.rows):
            for y in range(self.cols):
                obstacle_status = self.map_data[x][y]
                dist_above = self.find_distance_to_obstacle(x, y, -1, 0)
                dist_below = self.find_distance_to_obstacle(x, y, 1, 0)
                dist_left = self.find_distance_to_obstacle(x, y, 0, -1)
                dist_right = self.find_distance_to_obstacle(x, y, 0, 1)
                data.append([obstacle_status, dist_above, dist_below, dist_left, dist_right])
                progress.update(1)
        progress.close()
        df = pd.DataFrame(data)
        df.to_csv('map_data.csv', index=False, header=False)

    def add_noise_to_distances(self, noise_range=(-3, 5), valid_range=None):
        """
        Add integer noise to distances and save as a new CSV.
        """
        if valid_range is None:
            valid_range = (0, self.rows - 1)

        data = pd.read_csv('map_data.csv', header=None).values
        noisy_data = []
        progress = tqdm(total=len(data), desc="Adding Noise to Distances")

        for row in data:
            obstacle_status, dist_above, dist_below, dist_left, dist_right = row
            noise = np.random.randint(noise_range[0], noise_range[1] + 1, size=4)
            noisy_distances = np.clip(
                [dist_above + noise[0], dist_below + noise[1], dist_left + noise[2], dist_right + noise[3]],
                valid_range[0],
                valid_range[1]
            )
            noisy_data.append([obstacle_status] + noisy_distances.tolist())
            progress.update(1)

        progress.close()
        df_noisy = pd.DataFrame(noisy_data)
        df_noisy.to_csv('map_data_noisy.csv', index=False, header=False)

class RobotHandler:
    def __init__(self, map_handler):
        """
        Initialize the RobotHandler with a reference to the MapHandler instance.
        """
        self.map_handler = map_handler  # Reference to the map handler
        self.robot_position = None
        self.robot_orientation = None
        self.goal_cells = []

    def set_initial_position(self):
        """Set an initial position and orientation for the robot."""
        while True:
            x, y = random.randint(0, self.map_handler.rows - 1), random.randint(0, self.map_handler.cols - 1)
            if self.map_handler.map_data[x][y] == 0:  # Free cell
                self.robot_position = (x, y)
                self.robot_orientation = random.choice(['N', 'S', 'E', 'W'])  # Random orientation
                break

    def set_goal_cells(self, num_goals=3):
        """Set goal cells that the robot must visit."""
        self.goal_cells = []
        while len(self.goal_cells) < num_goals:
            x, y = random.randint(0, self.map_handler.rows - 1), random.randint(0, self.map_handler.cols - 1)
            if self.map_handler.map_data[x][y] == 0 and (x, y) != self.robot_position:
                self.goal_cells.append((x, y))

    def visualize_robot_and_goals(self):
        """Visualize the map with the robot's position and goal cells."""
        map_array = np.array(self.map_handler.map_data)
        plt.figure(figsize=(10, 10))
        plt.imshow(map_array, cmap='Greys', interpolation='nearest')
        plt.gca().invert_yaxis()

        # Mark the robot's position
        if self.robot_position:
            rx, ry = self.robot_position
            plt.scatter(ry, rx, color='blue', label='Robot Position', s=100)

        # Mark the goal cells
        for gx, gy in self.goal_cells:
            plt.scatter(gy, gx, color='red', label='Goal Cell', s=100)

        plt.title("Map with Robot and Goals")
        plt.legend()
        plt.show()
    
    def save_robot_data(self, filename="robot_data.txt"):
        """
        Save the robot's initial position, orientation, and goal cells in compact format.
        """
        with open(filename, "w") as file:
            # Write robot position, orientation, and goal cells in the specified format
            file.write(f"[{self.robot_position}], [{self.robot_orientation}], [{', '.join(str(goal) for goal in self.goal_cells)}]\n")
        print(f"Robot data saved to {filename}")
        
    def load_robot_data(self, filename="robot_data.txt"):
        """
        Load the robot's initial position, orientation, and goal cells from a file.
        Initializes the RobotHandler object with the loaded information.
        """
        with open(filename, "r") as file:
            data = file.readline().strip()

        # Use regex to extract the components
        pattern = r"\[\((\d+), (\d+)\)\], \[(\w+)\], \[(.*?)\]"
        match = re.match(pattern, data)

        if match:
            # Extract values from the match groups
            position = (int(match.group(1)), int(match.group(2)))  # robot position (x, y)
            orientation = match.group(3)  # robot orientation (e.g., 'N')

            # Extract goal cells as a list of tuples
            goals_str = match.group(4)
            goal_cells = []

            # Use regex to find individual goal cells in the format (x, y)
            goal_pattern = r"\((\d+), (\d+)\)"
            for goal_match in re.finditer(goal_pattern, goals_str):
                goal_cells.append((int(goal_match.group(1)), int(goal_match.group(2))))

            # Set values in the class
            self.robot_position = position
            self.robot_orientation = orientation
            self.goal_cells = goal_cells

            print("Robot data loaded successfully.")
            print(f"Position: {self.robot_position}, Orientation: {self.robot_orientation}")
            print(f"Goal Cells: {self.goal_cells}")
        else:
            print("Error: Data format is incorrect.")

class BeliefHandler:
    def __init__(self, map_handler, belief_type='gaussian', mu=None, sigma=None, uniform_range=None):
        """
        Initialize the BeliefHandler with the map and chosen belief distribution.
        
        :param map_handler: Instance of the MapHandler class containing the map data.
        :param belief_type: Type of distribution to use ('gaussian', 'uniform').
        :param mu: Mean for the Gaussian distribution as (mu_x, mu_y).
        :param sigma: Standard deviation for the Gaussian distribution.
        :param uniform_range: Range for the uniform distribution as (min, max).
        """
        self.map_handler = map_handler
        self.belief_type = belief_type
        self.mu = mu if mu else (self.map_handler.rows // 2, self.map_handler.cols // 2)  # Default center
        self.sigma = sigma if sigma else (self.map_handler.rows // 6, self.map_handler.cols // 6)  # Default sigma
        self.uniform_range = uniform_range if uniform_range else (0, 1)  # Default uniform range
        self.belief_map = np.zeros((self.map_handler.rows, self.map_handler.cols))  # Initialize belief map

        # Initialize belief based on the specified distribution
        self.initialize_belief()

    def initialize_belief(self):
        """Initialize the belief map using the selected distribution."""
        if self.belief_type == 'gaussian':
            self._initialize_gaussian_belief()
        elif self.belief_type == 'uniform':
            self._initialize_uniform_belief()
        else:
            raise ValueError("Unsupported belief type. Use 'gaussian' or 'uniform'.")

    def _initialize_gaussian_belief(self):
        """Initialize belief map using 2D Gaussian distribution."""
        mu_x, mu_y = self.mu
        sigma_x, sigma_y = self.sigma
        
        # Generate a grid of coordinates
        x = np.arange(self.map_handler.rows)
        y = np.arange(self.map_handler.cols)
        X, Y = np.meshgrid(y, x)  # Swap X and Y for correct grid alignment

        # Calculate 2D Gaussian distribution
        gauss_x = np.exp(-0.5 * ((X - mu_y) / sigma_y) ** 2)
        gauss_y = np.exp(-0.5 * ((Y - mu_x) / sigma_x) ** 2)
        self.belief_map = gauss_x * gauss_y  # Element-wise multiplication of the Gaussian distributions

        # Normalize the belief map so that the sum is 1 (i.e., a valid probability distribution)
        self.belief_map /= np.sum(self.belief_map)

    def _initialize_uniform_belief(self):
        """Initialize belief map using a uniform distribution."""
        # Assign equal probability to all cells
        self.belief_map.fill(1)
        
        # Normalize the belief map so that the sum is 1
        self.belief_map /= np.sum(self.belief_map)

    def visualize_belief(self):
        """Visualize the belief map."""
        plt.figure(figsize=(10, 10))
        plt.imshow(self.belief_map, cmap='hot', interpolation='nearest')
        plt.gca().invert_yaxis()
        plt.title("Robot Belief Map")
        plt.colorbar(label="Belief Value")
        plt.show()

    def get_belief(self, x, y):
        """Get the belief value for a specific cell."""
        return self.belief_map[x, y]

    def update_belief(self, new_belief_map):
        """Update the belief map with a new belief state."""
        self.belief_map = new_belief_map
        # Normalize the belief map to maintain valid probabilities
        self.belief_map /= np.sum(self.belief_map)

    def save_belief_to_csv(self, filename="belief_map.csv"):
        """Save the belief map to a CSV file."""
        np.savetxt(filename, self.belief_map, delimiter=",")

    def load_belief_from_csv(self, filename="belief_map.csv"):
        """Load the belief map from a CSV file."""
        self.belief_map = np.loadtxt(filename, delimiter=",")

class MotionModelHistogram:
    def __init__(self, map_handler, robot_handler, belief_handler, sigma_move=1.0, sigma_sensor=0.5):
        self.map_handler = map_handler
        self.robot_handler = robot_handler
        self.belief_handler = belief_handler
        self.sigma_move = sigma_move
        self.sigma_sensor= sigma_sensor
        self.grid_size = self.map_handler.rows
        self.prob = self.belief_handler.belief_map
        self.actual_data = pd.read_csv('map_data.csv', header=None).values
        self.noisy_data = pd.read_csv('map_data_noisy.csv', header=None).values
        
    def hist_move_probability(self, move_x, move_y):
        random_factor = random.uniform(-5, 5)
        dynamic_sigma = max(0.1, self.sigma_move + random_factor)
        new_prob = np.roll(self.prob, move_y, axis=0)  # Move in Y direction
        new_prob = np.roll(new_prob, move_x, axis=1)  # Move in X direction
        new_prob = gaussian_filter(new_prob, sigma=dynamic_sigma)  # Apply Gaussian blur to simulate spread
        self.prob = new_prob / new_prob.sum()  # Normalize
        
    def hist_sensor_update_single_point(self, i, j):
        """
        Updates the belief grid for a single grid cell based on sensor data read from files.

        Parameters:
            i, j (int): Indices of the grid cell to update.
        """
       
        row_index = i * self.grid_size + j
        obstacle_status = self.actual_data[row_index, 0]
        expected_data = self.actual_data[row_index, 1:]
        observed_data = self.noisy_data[row_index, 1:] # [dist_up, dist_down, dist_left, dist_right]

        # Handle obstacle cells
        if obstacle_status == 1:  # Assuming `1` represents an obstacle
            self.prob[i, j] = 1e-300  # Very small probability for obstacle cells
            return
        random_factor = random.uniform(-0.5, 0.5)
        dynamic_sigma = max(0.1, self.sigma_sensor + random_factor)
        # Compute likelihoods using Gaussian model
        likelihood_up = np.exp(-((observed_data[0] - expected_data[0]) ** 2) / (2 * dynamic_sigma**2))
        likelihood_down = np.exp(-((observed_data[1] - expected_data[1]) ** 2) / (2 * dynamic_sigma**2))
        likelihood_left = np.exp(-((observed_data[2] - expected_data[2]) ** 2) / (2 * dynamic_sigma**2))
        likelihood_right = np.exp(-((observed_data[3] - expected_data[3]) ** 2) / (2 * dynamic_sigma**2))

        # Combine likelihoods for this cell
        sensor_prob = likelihood_up * likelihood_down * likelihood_left * likelihood_right

        # Update belief for this cell
        self.prob[i, j] *= sensor_prob
        self.prob[i, j] += 1e-300  # Avoid zeros

        # Normalize the belief grid
        self.prob /= self.prob.sum()

    def hist_plot_probability_distribution(self, ax, title):
        ax.clear()
        sns.heatmap(self.prob, cmap='hot', cbar=False, ax=ax)
        ax.set_ylim(self.grid_size, 0)  # Invert the y-axis by setting the limits in reverse order
        ax.set_title(title)

class LocalizationVisualizerHistogram:
    def __init__(self, motion_model):
        self.motion_model = motion_model
        self.grid_size = self.motion_model.grid_size
        self.movements = [(-2,0),(-2,0),(-2,0),(-2,0),(-2,0),(-2,0),(-2,0),(-2,0),(-2,0),(-2,0),(0,-2),(0,-2),(0,-2),(0,-2),(0,-2),(0,-2),(0,-2),(0,-2),(0,-2),(0,-2)]
        self.steps_per_movement = 1
        self.robot_x, self.robot_y = self.motion_model.robot_handler.robot_position

    def update(self, frame):       
        self.axes.cla()
        step = (frame - 1) // self.steps_per_movement
        sub_step = (frame - 1) % self.steps_per_movement
        if step < len(self.movements):
            move_x, move_y = self.movements[step]
            move_x = int(move_x / self.steps_per_movement)
            move_y = int(move_y / self.steps_per_movement)
            # Update the robot's position
            self.robot_x += move_x
            self.robot_y += move_y
            self.motion_model.hist_move_probability(move_x, move_y)
            self.motion_model.hist_sensor_update_single_point(self.robot_x, self.robot_y)
            self.motion_model.hist_plot_probability_distribution(self.axes, f'Histogram Filter: Step {frame}')
        self.axes.set_xlim(0, self.grid_size - 1)
        self.axes.set_ylim(self.grid_size - 1,0)
        self.axes.invert_yaxis()
        plt.draw()

    def plot_initial_distribution(self):
        self.motion_model.hist_plot_probability_distribution(self.axes, 'Histogram Filter: Initial Probability Distribution')
        self.axes.invert_yaxis()  # Ensure y-axis is inverted for initial plot

    def visualize(self):
        fig, self.axes = plt.subplots(1, 1, figsize=(10, 10))
        ani = FuncAnimation(fig, self.update, frames=(len(self.movements) * self.steps_per_movement) + 1, repeat=False, interval=1)
        plt.tight_layout()
        plt.show()

class MotionModelParticleFilter:
    def __init__(self, map_handler, robot_handler, belief_handler ,num_particles=1000, sigma_move=10, sigma_sensor=50):
        self.map_handler = map_handler
        self.robot_handler = robot_handler
        self.belief_handler = belief_handler
        self.num_particles = num_particles
        self.grid_size = self.map_handler.rows
        self.sigma_move = sigma_move
        self.sigma_sensor = sigma_sensor
        self.particles = np.empty((num_particles, 2))
        self.weights = np.ones(num_particles) / num_particles
        self.distribution_type = self.belief_handler.belief_type

    def initialize_particles(self): 
        initial_position = self.robot_handler.robot_position
        if self.distribution_type == 'uniform':
            self.particles[:, 0] = np.random.uniform(0, self.grid_size, self.num_particles)
            self.particles[:, 1] = np.random.uniform(0, self.grid_size, self.num_particles)
        elif self.distribution_type == 'gaussian':
            if initial_position is None:
                raise ValueError("For Gaussian distribution, an initial position (x, y) must be provided.")
            x_mean, y_mean = initial_position
            self.particles[:, 0] = np.random.normal(x_mean, self.sigma_move*2, self.num_particles)
            self.particles[:, 1] = np.random.normal(y_mean, self.sigma_move*2, self.num_particles)
            # Clip the particles to ensure they remain within the grid boundaries
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
        ax.set_title(title)
        ax.invert_yaxis()

class LocalizationVisualizerParticleFilter:
    def __init__(self, map_handler, robot_handler, belief_handler, num_particles=1000):
        self.map_handler = map_handler
        self.robot_handler = robot_handler
        self.belief_handler = belief_handler
        self.grid_size = self.map_handler.rows
        self.num_particles = num_particles
        self.distribution_type = self.belief_handler.belief_type
        self.robot_x, self.robot_y = self.robot_handler.robot_position
        self.movements = [(-2,0),(-2,0),(-2,0),(-2,0),(-2,0),(-2,0),(-2,0),(-2,0),(-2,0),(-2,0),(-2,0),(-2,0),(-2,0),(-2,0),(-2,0),(-2,0),(-2,0),(-2,0),(-2,0),(-2,0),(0,-2),(0,-2),(0,-2),(0,-2),(0,-2),(0,-2),(0,-2),(0,-2),(0,-2),(0,-2),(0,-2),(0,-2),(0,-2),(0,-2),(0,-2),(0,-2),(0,-2),(0,-2),(0,-2),(0,-2)]
        self.steps_per_movement = 1
        self.initialize_filter()

    def initialize_filter(self):
        self.filter = MotionModelParticleFilter(self.map_handler, self.robot_handler, self.belief_handler,self.num_particles)
        self.filter.initialize_particles()

    def update(self, frame):
        if frame == 0:
            self.filter.plot_particles(self.axes, 'Particle Filter: Initial Particle Distribution')
            self.axes.invert_yaxis()  # Add this line to invert the y-axis for the initial frame
        else:
            self.axes.cla()
            step = (frame - 1) // self.steps_per_movement
            sub_step = (frame - 1) % self.steps_per_movement

            if step < len(self.movements):
                move_x, move_y = self.movements[step]
                move_x = int(move_x / self.steps_per_movement)
                move_y = int(move_y / self.steps_per_movement)

                # Update the robot's position
                self.robot_x += move_x
                self.robot_y += move_y

                # Update the particle filter
                self.filter.move_particles(move_x, move_y)
                self.filter.sensor_update(self.robot_x, self.robot_y)
                self.filter.resample_particles()
                self.filter.plot_particles(self.axes, f'Particle Filter: Step {frame}')

            self.axes.set_xlim(0, self.grid_size - 1)
            self.axes.set_ylim(0, self.grid_size - 1)
            plt.draw()

    def visualize(self):
        fig, self.axes = plt.subplots(1, 1, figsize=(10, 10))
        ani = FuncAnimation(fig, self.update, frames=(len(self.movements) * self.steps_per_movement) + 1, repeat=False, interval=10)
        plt.tight_layout()
        plt.show()

