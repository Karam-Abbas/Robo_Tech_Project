import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter

def gaussian_2d_probability(x, y, mu_x, mu_y, sigma_x, sigma_y):
    exponent = -((x - mu_x)**2 / (2 * sigma_x**2) + (y - mu_y)**2 / (2 * sigma_y**2))
    norm_factor = 1 / (2 * np.pi * sigma_x * sigma_y)
    return norm_factor * np.exp(exponent)

class MotionModel:
    def __init__(self, 
                 grid_size=100, 
                 num_particles=1000, 
                 filter_type='histogram', 
                 distribution_type='gaussian',
                 initial_position=None,
                 obstacles=None):
        """
        Initialize the Motion Model with advanced configuration
        
        Args:
            grid_size (int): Size of the grid for localization
            num_particles (int): Number of particles for particle filter
            filter_type (str): Type of filter - 'histogram' or 'particle'
            distribution_type (str): Initial distribution type - 'gaussian' or 'uniform'
            initial_position (tuple): Starting (x, y) coordinates of the robot
            obstacles (list): List of obstacle coordinates or obstacle grid
        """
        self.grid_size = grid_size
        self.num_particles = num_particles
        self.filter_type = filter_type
        self.distribution_type = distribution_type
        
        # Set initial position
        if initial_position is None:
            # Random initial position if not specified
            self.initial_x = np.random.randint(0, grid_size)
            self.initial_y = np.random.randint(0, grid_size)
        else:
            self.initial_x, self.initial_y = initial_position
        
        # Prepare obstacles
        self.obstacles = self._prepare_obstacles(obstacles, grid_size)
        
        # Prepare grid for meshgrid calculations
        self.X, self.Y = np.meshgrid(np.linspace(0, grid_size-1, grid_size), 
                                     np.linspace(0, grid_size-1, grid_size))
        
        # Initialize the appropriate filter
        self._initialize_filter()
        
        # Predefined movements and steps
        self.movements = [(15, 0), (0, 15), (-15, 0), (0, -15)]
        self.steps_per_movement = 1

    def _prepare_obstacles(self, obstacles, grid_size):
        """
        Prepare obstacle representation
        
        Args:
            obstacles (list or np.ndarray): Obstacles specification
            grid_size (int): Size of the grid
        
        Returns:
            np.ndarray: Grid representing obstacles
        """
        # Create obstacle grid
        obstacle_grid = np.zeros((grid_size, grid_size), dtype=bool)
        
        if obstacles is None:
            # Default obstacle placement if not specified
            obstacle_grid[20:40, 20:40] = True
            obstacle_grid[60:80, 60:80] = True
        elif isinstance(obstacles, list):
            # List of obstacle coordinates
            for obs in obstacles:
                x, y = obs
                obstacle_grid[x, y] = True
        elif isinstance(obstacles, np.ndarray):
            # If already a boolean grid of obstacles
            obstacle_grid = obstacles.astype(bool)
        
        return obstacle_grid

    def _initialize_filter(self):
        """Initialize the selected filter type"""
        if self.filter_type == 'histogram':
            self.filter = HistogramFilter(self.grid_size, sigma_move=1.5, sigma_sensor=5)
            self.filter.initialize_probability(
                self.initial_x, self.initial_y, 
                sigma_x=5, sigma_y=5, 
                distribution_type=self.distribution_type
            )
            # Set no-go areas based on obstacles
            self.filter.set_no_go_areas(self.obstacles)
        
        elif self.filter_type == 'particle':
            self.filter = ParticleFilter(
                self.num_particles, 
                self.grid_size, 
                sigma_move=1.0, 
                sigma_sensor=5.0
            )
            self.filter.initialize_particles(
                distribution_type=self.distribution_type,
                initial_position=(self.initial_x, self.initial_y),
                obstacles=self.obstacles
            )
        
        else:
            raise ValueError(f"Invalid filter type: {self.filter_type}. Choose 'histogram' or 'particle'.")

    def switch_filter(self, filter_type='histogram', distribution_type=None, initial_position=None, obstacles=None):
        """
        Switch between histogram and particle filters
        
        Args:
            filter_type (str): Type of filter to switch to
            distribution_type (str, optional): Distribution type for initialization
            initial_position (tuple, optional): New initial position
            obstacles (list or np.ndarray, optional): New obstacle configuration
        """
        self.filter_type = filter_type
        
        # Update initial position if provided
        if initial_position is not None:
            self.initial_x, self.initial_y = initial_position
        
        # Update obstacles if provided
        if obstacles is not None:
            self.obstacles = self._prepare_obstacles(obstacles, self.grid_size)
        
        # Update distribution type if provided
        if distribution_type:
            self.distribution_type = distribution_type
        
        # Reinitialize the filter with new settings
        self._initialize_filter()

    def visualize(self, movements=None):
        """
        Visualize the motion and filtering process
        
        Args:
            movements (list, optional): Custom list of (x,y) movements. 
                                        If None, uses predefined movements.
        """
        if movements:
            self.movements = movements
        
        fig, self.axes = plt.subplots(1, 1, figsize=(7, 7))
        
        # Create animation
        ani = FuncAnimation(
            fig, 
            self._update, 
            frames=(len(self.movements) * self.steps_per_movement) + 1, 
            repeat=False, 
            interval=1000
        )
        
        plt.tight_layout()
        plt.show()

    def _update(self, frame):
        """
        Update method for animation frames
        
        Args:
            frame (int): Current animation frame
        """
        if frame == 0:
            # Initial state visualization
            if self.filter_type == 'histogram':
                self.filter.plot_probability_distribution(
                    self.axes, 
                    f'Histogram Filter: Initial Position ({self.initial_x}, {self.initial_y})'
                )
            elif self.filter_type == 'particle':
                self.filter.plot_particles(
                    self.axes, 
                    f'Particle Filter: Initial Position ({self.initial_x}, {self.initial_y})'
                )
        else:
            self.axes.cla()
            step = (frame - 1) // self.steps_per_movement
            sub_step = (frame - 1) % self.steps_per_movement

            if step < len(self.movements):
                move_x, move_y = self.movements[step]
                move_x = int(move_x / self.steps_per_movement)
                move_y = int(move_y / self.steps_per_movement)

                if self.filter_type == 'histogram':
                    # Check if movement is possible
                    if not self._is_movement_valid(move_x, move_y):
                        move_x, move_y = 0, 0
                    
                    self.filter.move_probability(move_x, move_y)
                    self.filter.sensor_update(
                        self.initial_x + move_x * sub_step, 
                        self.initial_y + move_y * sub_step
                    )
                    self.filter.plot_probability_distribution(
                        self.axes, 
                        f'Histogram Filter: Step {frame}'
                    )
                elif self.filter_type == 'particle':
                    # Check if movement is possible
                    if not self._is_movement_valid(move_x, move_y):
                        move_x, move_y = 0, 0
                    
                    self.filter.move_particles(move_x, move_y)
                    self.filter.sensor_update(
                        self.initial_x + move_x * sub_step, 
                        self.initial_y + move_y * sub_step
                    )
                    self.filter.resample_particles()
                    self.filter.plot_particles(
                        self.axes, 
                        f'Particle Filter: Step {frame}'
                    )
            
            self.axes.set_xlim(0, self.grid_size - 1)
            self.axes.set_ylim(0, self.grid_size - 1)
            
            # Overlay obstacles
            if self.obstacles is not None:
                obstacle_coords = np.where(self.obstacles)
                self.axes.scatter(
                    obstacle_coords[1], 
                    obstacle_coords[0], 
                    color='red', 
                    alpha=0.5, 
                    marker='s'
                )
            
            plt.draw()

    def _is_movement_valid(self, move_x, move_y):
        """
        Check if the proposed movement is valid (not into an obstacle)
        
        Args:
            move_x (int): Proposed x movement
            move_y (int): Proposed y movement
        
        Returns:
            bool: True if movement is valid, False otherwise
        """
        new_x = self.initial_x + move_x
        new_y = self.initial_y + move_y
        
        # Check if new position is within grid and not an obstacle
        if (0 <= new_x < self.grid_size and 
            0 <= new_y < self.grid_size and 
            not self.obstacles[new_y, new_x]):
            return True
        return False

# Rest of the classes (HistogramFilter, ParticleFilter) remain the same as in the previous implementation

# Modify the ParticleFilter's initialize_particles method to support initial position and obstacles
class ParticleFilter:
    def __init__(self, num_particles, grid_size, sigma_move=1.0, sigma_sensor=0.5):
        # ... (previous initialization remains the same)
        pass

    def initialize_particles(self, distribution_type='uniform', initial_position=None, obstacles=None):
        """
        Initialize particles with support for specific initial position and obstacles
        
        Args:
            distribution_type (str): Type of distribution - 'uniform' or 'gaussian'
            initial_position (tuple): Specific (x, y) initial position
            obstacles (np.ndarray): Grid of obstacles
        """
        if initial_position is None:
            initial_x = self.grid_size // 2
            initial_y = self.grid_size // 2
        else:
            initial_x, initial_y = initial_position

        if distribution_type == 'uniform':
            # Generate particles avoiding obstacles
            valid_particles = 0
            while valid_particles < self.num_particles:
                candidate_x = np.random.uniform(0, self.grid_size, self.num_particles - valid_particles)
                candidate_y = np.random.uniform(0, self.grid_size, self.num_particles - valid_particles)
                
                # Filter out particles in obstacle areas
                if obstacles is not None:
                    valid_mask = ~obstacles[candidate_y.astype(int), candidate_x.astype(int)]
                    candidate_x = candidate_x[valid_mask]
                    candidate_y = candidate_y[valid_mask]
                
                # Update particles and valid count
                start_idx = valid_particles
                end_idx = start_idx + len(candidate_x)
                self.particles[start_idx:end_idx, 0] = candidate_x
                self.particles[start_idx:end_idx, 1] = candidate_y
                valid_particles = end_idx

        elif distribution_type == 'gaussian':
            # Generate particles around initial position, avoiding obstacles
            valid_particles = 0
            while valid_particles < self.num_particles:
                candidate_x = np.random.normal(initial_x, self.sigma_move, self.num_particles - valid_particles)
                candidate_y = np.random.normal(initial_y, self.sigma_move, self.num_particles - valid_particles)
                
                # Clip to grid
                candidate_x = np.clip(candidate_x, 0, self.grid_size - 1)
                candidate_y = np.clip(candidate_y, 0, self.grid_size - 1)
                
                # Filter out particles in obstacle areas
                if obstacles is not None:
                    valid_mask = ~obstacles[candidate_y.astype(int), candidate_x.astype(int)]
                    candidate_x = candidate_x[valid_mask]
                    candidate_y = candidate_y[valid_mask]
                
                # Update particles and valid count
                start_idx = valid_particles
                end_idx = start_idx + len(candidate_x)
                self.particles[start_idx:end_idx, 0] = candidate_x
                self.particles[start_idx:end_idx, 1] = candidate_y
                valid_particles = end_idx

        # Ensure all particles are within grid and not on obstacles
        self.particles = np.clip(self.particles, 0, self.grid_size - 1)

# Example usage
if __name__ == "__main__":
    # Custom obstacles
    custom_obstacles = np.zeros((100, 100), dtype=bool)
    custom_obstacles[20:40, 20:40] = True  # Rectangle obstacle
    custom_obstacles[60:80, 60:80] = True  # Another rectangle obstacle

    # Create a MotionModel with custom initial position and obstacles
    model = MotionModel(
        grid_size=100, 
        filter_type='histogram', 
        initial_position=(50, 50),  # Custom initial position
        obstacles=custom_obstacles  # Custom obstacles
    )
    
    # Visualize the model
    model.visualize()

    # Switch to particle filter with different settings
    model.switch_filter(
        filter_type='particle', 
        distribution_type='gaussian',
        initial_position=(75, 75)
    )
    model.visualize()