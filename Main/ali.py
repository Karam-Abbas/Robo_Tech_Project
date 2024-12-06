import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import math

class Environment:
    def __init__(self, map_file):
        self.map_data = np.genfromtxt(map_file, delimiter=',', skip_header=1)
        self.rows = int(np.max(self.map_data[:, 0]) + 1)
        self.cols = int(np.max(self.map_data[:, 1]) + 1)
        self.map = np.zeros((self.rows, self.cols), dtype=int)
        self.horizontal_distances = {}
        self.vertical_distances = {}
        self.load_map()
        
    def load_map(self):
        """Load map data with improved error handling"""
        try:
            for row in self.map_data:
                x, y = int(row[0]), int(row[1])
                if 0 <= x < self.cols and 0 <= y < self.rows:
                    self.map[y, x] = int(row[2])
                    self.horizontal_distances[(y, x)] = (row[3], row[4])
                    self.vertical_distances[(y, x)] = (row[5], row[6])
                else:
                    raise ValueError(f"Invalid coordinates: ({x}, {y})")
        except Exception as e:
            raise Exception(f"Error loading map: {str(e)}")

    def is_valid_position(self, x, y):
        """Check if position is valid and obstacle-free"""
        x, y = int(round(x)), int(round(y))
        return (0 <= x < self.cols and 
                0 <= y < self.rows and 
                not self.is_obstacle(x, y))

    def is_obstacle(self, x, y):
        x, y = int(round(x)), int(round(y))
        return not (0 <= y < self.rows and 0 <= x < self.cols) or self.map[y, x] == 1

    def get_distances(self, x, y):
        """Get distances with boundary checking"""
        x, y = int(round(x)), int(round(y))
        if self.is_valid_position(x, y):
            return (
                self.horizontal_distances.get((y, x), (0, 0)),
                self.vertical_distances.get((y, x), (0, 0))
            )
        return ((0, 0), (0, 0))

class SensorModel:
    def _init_(self, noise_std=0.5):
        self.noise_std = noise_std
    
    def get_reading(self, env, x, y):
        """Simulate sensor readings with improved noise model"""
        if env.is_obstacle(x, y):
            return None
            
        (right, left), (up, down) = env.get_distances(x, y)
        
        # Add correlated noise to simulate systematic errors
        systematic_error = np.random.normal(0, 0.2)  # Systematic error component
        noise = np.random.normal(0, self.noise_std, 4) + systematic_error
        
        # Ensure non-negative distances
        readings = {
            "right": max(0, right + noise[0]),
            "left": max(0, left + noise[1]),
            "up": max(0, up + noise[2]),
            "down": max(0, down + noise[3])
        }
        
        return readings

class MotionModel:
    def __init__(self, turn_noise_std=2.0, move_noise_std=0.2):
        self.turn_noise_std = turn_noise_std
        self.move_noise_std = move_noise_std
    
    def execute_motion(self, x, y, orientation, turn, move):
        """Execute motion with improved noise model and collision checking"""
        # Add systematic and random components to noise
        turn_systematic = np.random.normal(0, 1.0)
        move_systematic = np.random.normal(0, 0.1)
        
        turn_noise = np.random.normal(0, self.turn_noise_std) + turn_systematic
        move_noise = np.random.normal(0, self.move_noise_std) + move_systematic
        
        new_orientation = (orientation + turn + turn_noise) % 360
        
        # Convert to radians for precise calculations
        theta = np.deg2rad(new_orientation)
        
        # Calculate new position with noise
        new_x = x + (move + move_noise) * np.cos(theta)
        new_y = y + (move + move_noise) * np.sin(theta)
        
        return new_x, new_y, new_orientation

class HistogramFilter:
    def _init_(self, env):
        self.env = env
        # Initialize with Gaussian distribution around start position
        self.belief = self._initialize_gaussian_belief(0, 0)
        self.sensor_model = SensorModel()
    
    def _initialize_gaussian_belief(self, start_x, start_y, sigma=1.0):
        """Initialize belief state with 2D Gaussian"""
        belief = np.zeros((self.env.rows, self.env.cols))
        x, y = np.meshgrid(np.arange(self.env.cols), np.arange(self.env.rows))
        pos = np.dstack((x, y))
        rv = multivariate_normal([start_x, start_y], [[sigma, 0], [0, sigma]])
        belief = rv.pdf(pos)
        
        # Set zero probability for obstacles
        for y in range(self.env.rows):
            for x in range(self.env.cols):
                if self.env.is_obstacle(x, y):
                    belief[y, x] = 0
                    
        return belief / np.sum(belief)  # Normalize

    def update_motion(self, dx, dy):
        """Update belief based on motion with improved accuracy"""
        new_belief = np.zeros_like(self.belief)
        
        for y in range(self.env.rows):
            for x in range(self.env.cols):
                if not self.env.is_obstacle(x, y):
                    # Calculate source position
                    src_x, src_y = x - dx, y - dy
                    
                    # Interpolate belief for non-integer positions
                    x0, y0 = int(np.floor(src_x)), int(np.floor(src_y))
                    x1, y1 = x0 + 1, y0 + 1
                    
                    if (0 <= x0 < self.env.cols - 1 and 
                        0 <= y0 < self.env.rows - 1):
                        # Bilinear interpolation weights
                        wx = src_x - x0
                        wy = src_y - y0
                        
                        # Interpolate belief
                        b00 = self.belief[y0, x0]
                        b01 = self.belief[y0, x1]
                        b10 = self.belief[y1, x0]
                        b11 = self.belief[y1, x1]
                        
                        interpolated_belief = (
                            b00 * (1 - wx) * (1 - wy) +
                            b01 * wx * (1 - wy) +
                            b10 * (1 - wx) * wy +
                            b11 * wx * wy
                        )
                        
                        new_belief[y, x] = interpolated_belief
        
        # Normalize
        total = np.sum(new_belief)
        if total > 0:
            self.belief = new_belief / total
        else:
            self.belief = self._initialize_gaussian_belief(dx, dy)

    def update_sensor(self, sensor_readings, x, y):
        """Update belief based on sensor readings with improved accuracy"""
        if sensor_readings is None:
            return

        likelihood = np.ones_like(self.belief)
        
        for i in range(self.env.rows):
            for j in range(self.env.cols):
                if not self.env.is_obstacle(j, i):
                    expected_readings = self.sensor_model.get_reading(self.env, j, i)
                    
                    if expected_readings:
                        # Calculate likelihood using Mahalanobis distance
                        error = 0
                        for direction in sensor_readings:
                            diff = sensor_readings[direction] - expected_readings[direction]
                            error += (diff ** 2) / (2 * self.sensor_model.noise_std ** 2)
                        
                        likelihood[i, j] = np.exp(-error)
        
        # Update belief
        self.belief *= likelihood
        
        # Normalize
        total = np.sum(self.belief)
        if total > 0:
            self.belief /= total
        else:
            self.belief = self._initialize_gaussian_belief(x, y)

class ParticleFilter:
    def _init_(self, env, num_particles=1000):
        self.env = env
        self.num_particles = num_particles
        self.motion_model = MotionModel()
        self.sensor_model = SensorModel()
        self.initialize_particles()
    
    def initialize_particles(self):
        """Initialize particles with improved distribution"""
        self.particles = []
        self.weights = np.ones(self.num_particles) / self.num_particles
        
        count = 0
        while count < self.num_particles:
            x = np.random.uniform(0, self.env.cols)
            y = np.random.uniform(0, self.env.rows)
            if not self.env.is_obstacle(x, y):
                self.particles.append({
                    'x': x,
                    'y': y,
                    'orientation': np.random.uniform(0, 360)
                })
                count += 1

    def motion_update(self, dx, dy):
        """Update particles based on motion with improved noise model"""
        for particle in self.particles:
            # Calculate required turn and move
            target_angle = np.rad2deg(np.arctan2(dy, dx))
            current_angle = particle['orientation']
            
            # Calculate shortest turn angle
            turn = ((target_angle - current_angle + 180) % 360) - 180
            move = np.sqrt(dx*2 + dy*2)
            
            # Execute motion
            new_x, new_y, new_orientation = self.motion_model.execute_motion(
                particle['x'], particle['y'], particle['orientation'], turn, move
            )
            
            # Update particle if new position is valid
            if self.env.is_valid_position(new_x, new_y):
                particle['x'] = new_x
                particle['y'] = new_y
                particle['orientation'] = new_orientation

    def sensor_update(self, sensor_readings):
        """Update particle weights based on sensor readings with improved likelihood model"""
        if sensor_readings is None:
            return

        for i, particle in enumerate(self.particles):
            if self.env.is_valid_position(particle['x'], particle['y']):
                expected_readings = self.sensor_model.get_reading(
                    self.env, particle['x'], particle['y']
                )
                
                if expected_readings:
                    # Calculate likelihood using Mahalanobis distance
                    error = 0
                    for direction in sensor_readings:
                        diff = sensor_readings[direction] - expected_readings[direction]
                        error += (diff ** 2) / (2 * self.sensor_model.noise_std ** 2)
                    
                    self.weights[i] *= np.exp(-error)
        
        # Normalize weights
        total_weight = np.sum(self.weights)
        if total_weight > 0:
            self.weights /= total_weight
        else:
            self.weights = np.ones(self.num_particles) / self.num_particles

    def resample(self):
        """Resample particles using systematic resampling"""
        cumsum = np.cumsum(self.weights)
        cumsum[-1] = 1.0  # Handle numerical errors
        
        # Systematic resampling
        positions = (np.random.random() + np.arange(self.num_particles)) / self.num_particles
        new_particles = []
        
        i = 0
        for position in positions:
            while cumsum[i] < position:
                i += 1
            new_particles.append(dict(self.particles[i]))
        
        self.particles = new_particles
        self.weights = np.ones(self.num_particles) / self.num_particles
        
        



def main():
    # Load environment from a sample map file
    map_file = 'map.csv'  # Replace with the path to your map file
    env = Environment(map_file)
    
    # Set up the motion and sensor models
    motion_model = MotionModel()
    sensor_model = SensorModel()
    
    # Initialize the histogram and particle filters
    histogram_filter = HistogramFilter()
    particle_filter = ParticleFilter()
    
    # Define the initial position and orientation
    x, y, orientation = 0, 0, 0  # Starting at (0, 0) facing right (0°)
    
    # Simulation parameters
    steps = 10
    motions = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # Right, Up, Left, Down
    
    for step in range(steps):
        print(f"\nStep {step + 1}")
        
        # Choose a motion command
        dx, dy = motions[step % len(motions)]  # Cycle through motions
        
        # Simulate motion
        new_x, new_y, new_orientation = motion_model.execute_motion(x, y, orientation, 0, math.sqrt(dx**2 + dy**2))
        
        # Validate and update position
        if env.is_valid_position(new_x, new_y):
            x, y, orientation = new_x, new_y, new_orientation
        print(f"New position: ({x:.2f}, {y:.2f}), Orientation: {orientation:.2f}°")
        
        # Simulate sensor readings
        sensor_readings = sensor_model.get_reading(env, x, y)
        print(f"Sensor readings: {sensor_readings}")
        
        # Update histogram filter
        histogram_filter.update_motion(dx, dy)
        histogram_filter.update_sensor(sensor_readings, x, y)
        print("Updated histogram belief:")
        print(histogram_filter.belief)
        
        # Update particle filter
        particle_filter.motion_update(dx, dy)
        particle_filter.sensor_update(sensor_readings)
        particle_filter.resample()
        print("Particle filter resampled particles:")
        for particle in particle_filter.particles[:5]:  # Show a sample of particles
            print(particle)
    
    print("\nSimulation complete.")

if __name__ == "__main__":
    main()
