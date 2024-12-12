from classes import MapHandler,RobotHandler,BeliefHandler

import matplotlib.pyplot as plt
# # Example usage
map_handler = MapHandler()
# map_handler.generate_map()
# map_handler.calculate_distances()
# map_handler.add_noise_to_distances()
# map_handler.save_map_to_csv("map.csv")
map_handler.initialize_from_csv("map.csv")
map_handler.visualize_map()

robot_handler = RobotHandler(map_handler)
robot_handler.load_robot_data()
# robot_handler.set_initial_position()
# robot_handler.set_goal_cells(num_goals=5)
# robot_handler.save_robot_data("robot_data.txt")
robot_handler.visualize_robot_and_goals()


# Assuming map_handler is already initialized and robot_handler has the robot position:
robot_position = robot_handler.robot_position  # (mu_x, mu_y)

belief_handler = BeliefHandler(
    map_handler=map_handler,
    belief_type='gaussian', 
    mu=robot_position,  # Position of the robot in the grid, if no position then center of the grid will be set by default.
    sigma=(30, 30)  # spread of the probability distribution in the grid, if no value then 1/6(grid size will be set.)
)
belief_handler.visualize_belief()



