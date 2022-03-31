# Graduation Project

## Usage

### start simulation env
roslaunch mpc_ros quadrotor_empty_world.launch
### start safe flight corridor node
<!-- roslaunch flight_corridor flight_corridor.launch -->
### start MPC node
roslaunch mpc_ros test.launch
### start trigger
rostopic pub /hummingbird/trigger std_msgs/Empty "{}"