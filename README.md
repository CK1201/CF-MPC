# Graduation Project

## Simple Usage

#### start simulation env
roslaunch mpc_ros quadrotor_empty_world.launch

### 1. Fit Param

#### start MPC node

roslaunch mpc_ros fit_coeff.launch

#### start trigger

rostopic pub /hummingbird/trigger std_msgs/Empty "{}"

### 2. Tracking

#### start MPC node
roslaunch mpc_ros test.launch
#### start trigger
rostopic pub /hummingbird/trigger std_msgs/Empty "{}"