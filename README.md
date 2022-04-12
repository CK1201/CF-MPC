# Graduation Project

## Usage

### start simulation env
roslaunch mpc_ros quadrotor_empty_world.launch

### start MPC node
roslaunch mpc_ros test.launch
### start trigger
rostopic pub /hummingbird/trigger std_msgs/Empty "{}"