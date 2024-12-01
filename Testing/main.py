from classes import MapHandler,RobotHandler,BeliefHandler,MotionModel,FilterHandler

import matplotlib.pyplot as plt
# # Example usage
# map_handler = MapHandler()
# map_handler.generate_map()
# map_handler.calculate_distances()
# map_handler.add_noise_to_distances()
# map_handler.visualize_map()

map_handler = MapHandler()
map_handler.initialize_from_csv("map.csv")
robot_handler = RobotHandler(map_handler)
robot_handler.load_robot_data()
# robot_handler.set_initial_position()
# robot_handler.set_goal_cells(num_goals=5)
# robot_handler.save_robot_data("robot_data.txt")
# robot_handler.visualize_robot_and_goals()


# Assuming map_handler is already initialized and robot_handler has the robot position:
robot_position = robot_handler.robot_position  # (mu_x, mu_y)

belief_handler = BeliefHandler(
    map_handler=map_handler,
    belief_type='gaussian', 
    mu=robot_position,
    sigma=(30, 30)  # Optional: you can set sigma based on map size or a fixed value
)
belief_handler.visualize_belief()

motion_model = MotionModel(
    belief_handler=belief_handler
)

# Perform some actions
motion_model.turn(45)  # Turn by (_) degree with noise
motion_model.move(100)  # Move by (_) cm with noise

# Get the updated belief map
belief_map = motion_model.get_current_belief()

# Visualize the belief map
belief_handler.visualize_belief()

