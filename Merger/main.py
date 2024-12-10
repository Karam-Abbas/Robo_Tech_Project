from functions import MapHandler,BeliefHandler,MotionModel,RobotHandler

# # Example usage
# map_handler = MapHandler()
# robot_handler = RobotHandler(map_handler)
# belief_handler = BeliefHandler(map_handler)
# robot_motion_model = RobotMotionModel(map_handler, belief_handler)

# # Set the initial robot position and goal cells
# robot_handler.set_initial_position()
# robot_handler.set_goal_cells()

# # Visualize the robot and goals
# robot_handler.visualize_robot_and_goals()

# # Implement a single command
# action = "Turn"  # or "Move"
# robot_motion_model.update_belief(action)

# # Visualize the updated belief map
# belief_handler.visualize_belief()




map_handler = MapHandler()
map_handler.initialize_from_csv("map.csv")

robot_handler = RobotHandler(map_handler)
robot_handler.load_robot_data()

robot_position = robot_handler.robot_position  # (mu_x, mu_y)
robot_orientation = robot_handler.robot_orientation

belief_handler = BeliefHandler(
    map_handler=map_handler,
    belief_type='gaussian', 
    mu=robot_position,  # Position of the robot in the grid, if no position then center of the grid will be set by default.
    sigma=(10, 10)  # spread of the probability distribution in the grid, if no value then 1/6(grid size will be set.)
)
belief_handler.visualize_belief()

motion_model = MotionModel(
    map_handler=map_handler,
    belief_handler=belief_handler,
    robot_handler=robot_handler
)

# motion_model.simulate_turn(270)  # Turn 45 degrees
# motion_model.simulate_move(146)  # Move 10 steps forward
# motion_model.visualize_update()

# # # Get the updated belief map
# # belief_map = motion_model.get_current_belief()

# # Visualize the belief map

motion_model.turn(270)
# motion_model.move(100)

motion_model.move(400)



belief_handler.visualize_belief()


motion_model.turn(180)
# motion_model.move(100)

motion_model.move(400)

belief_handler.visualize_belief()


# print("Initial State")
# motion_model.visualize_update()

# # Perform noisy moves and turns
# motion_model.noisy_turn(270, noise_range=(-10, 10))  # Noisy turn of 45° ± 10°
# for i in range(5):
#     motion_model.noisy_move(10, distance_noise=(-2, 2), angle_noise=(-15, 15))  # Noisy move of 10 ± 2 steps

# print("Final State")
# motion_model.visualize_update()
