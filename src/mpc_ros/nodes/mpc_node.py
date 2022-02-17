#!/usr/bin/env python3.6
import imp
from logging import shutdown

from matplotlib import use
import rospy, genpy
import numpy as np
from std_msgs.msg import Int16, Header
from nav_msgs.msg import Odometry
from mav_msgs.msg import Actuators
from quadrotor_msgs.msg import ControlCommand
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Empty
from src.utils.utils import *
from src.quad_mpc.quad_optimizer import QuadrotorOptimizer

'''
RotorS
/gazebo/link_states
/gazebo/model_states
/gazebo/parameter_descriptions
/gazebo/parameter_updates
/gazebo/set_link_state
/gazebo/set_model_state
/hummingbird/command/motor_speed
/hummingbird/command/pose
/hummingbird/command/trajectory
/hummingbird/gazebo/command/motor_speed
/hummingbird/ground_truth/imu
/hummingbird/ground_truth/odometry
/hummingbird/ground_truth/pose
/hummingbird/ground_truth/pose_with_covariance
/hummingbird/ground_truth/position
/hummingbird/ground_truth/transform
/hummingbird/imu
/hummingbird/joint_states
/hummingbird/odometry_sensor1/odometry
/hummingbird/odometry_sensor1/pose
/hummingbird/odometry_sensor1/pose_with_covariance
/hummingbird/odometry_sensor1/position
/hummingbird/odometry_sensor1/transform
/hummingbird/wind_speed



/hummingbird/autopilot/control_command_input
/hummingbird/autopilot/feedback
/hummingbird/autopilot/force_hover
/hummingbird/autopilot/land
/hummingbird/autopilot/off
/hummingbird/autopilot/pose_command
/hummingbird/autopilot/reference_state
/hummingbird/autopilot/start
/hummingbird/autopilot/trajectory
/hummingbird/autopilot/velocity_command
/hummingbird/bridge/arm
/hummingbird/command/motor_speed
/hummingbird/control_command
/hummingbird/gazebo/command/motor_speed
/hummingbird/ground_truth/imu
/hummingbird/ground_truth/odometry
/hummingbird/ground_truth/pose
/hummingbird/ground_truth/pose_with_covariance
/hummingbird/ground_truth/position
/hummingbird/ground_truth/transform
/hummingbird/imu
/hummingbird/joint_states
/hummingbird/joy
/hummingbird/joy/set_feedback
/hummingbird/low_level_feedback
/hummingbird/odometry_sensor1/odometry
/hummingbird/odometry_sensor1/pose
/hummingbird/odometry_sensor1/pose_with_covariance
/hummingbird/odometry_sensor1/position
/hummingbird/odometry_sensor1/transform
/hummingbird/received_sbus_message
/hummingbird/wind_speed
'''
class QuadMPC:
    def __init__(self, t_horizon = 1, N = 20) -> None:
        rospy.init_node("mpc_node")
        quad_name = rospy.get_param('quad_name', default='hummingbird')
        self.t_horizon = t_horizon # prediction horizon
        self.N = N # number of discretization steps
        period = t_horizon / N
        self.have_odom = False
        self.traj_type = 0
        # self.traj_ref = np.zeros((N, 3))
        # load model
        self.quadrotorOptimizer = QuadrotorOptimizer(t_horizon, N)
        # Subscribers
        self.odom_sub = rospy.Subscriber('/' + quad_name + '/ground_truth/odometry', Odometry, self.odom_callback)
        self.traj_type_sub = rospy.Subscriber('/' + quad_name + '/traj_type', Int16, self.traj_type_callback)
        # Publishers
        self.control_pub = rospy.Publisher('/' + quad_name + '/command/motor_speed', Actuators, queue_size=1, tcp_nodelay=True)
        self.control_pose_pub = rospy.Publisher('/' + quad_name + '/command/pose', PoseStamped, queue_size=1, tcp_nodelay=True)
        self.control_cmd_pub = rospy.Publisher('/' + quad_name + '/control_command', ControlCommand, queue_size=1, tcp_nodelay=True)
        # Trying to unpause Gazebo for 10 seconds.
        rospy.wait_for_service('/gazebo/unpause_physics')
        unpause_gazebo = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        unpaused = unpause_gazebo.call()
        self.u_set = np.zeros((self.N, self.quadrotorOptimizer.ocp.model.u.size()[0]))
        self.x_set = np.zeros((self.N, self.quadrotorOptimizer.ocp.model.x.size()[0]))
        self.have_uset = False
        i = 0
        while (i <= 10 and not(unpaused)):
            i += 1
            rospy.sleep(1.0)
            unpaused = unpause_gazebo.call()
        if not(unpaused):
            rospy.logfatal("Could not wake up Gazebo.")
            rospy.signal_shutdown("Could not wake up Gazebo.")
        else:
            rospy.loginfo("Unpaused the Gazebo simulation.") 
        
        # Timer
        self.timer = rospy.Timer(rospy.Duration(t_horizon / N), self.QuadMPCFSM)

        rate = rospy.Rate(N / t_horizon)
        self.control_num = 0
        while not rospy.is_shutdown():
            if self.have_uset:
            #     print(self.u_set[self.control_num])
            #     cmd = Actuators()
            #     cmd.angular_velocities = self.u_set[self.control_num]
            #     self.control_pub.publish(cmd)
                x1 = self.x_set[self.control_num]
                q = unit_quat(x1[6: 10])
                cmd = ControlCommand()
                cmd.header.stamp = rospy.Time.now()
                cmd.expected_execution_time = rospy.Time.now()
                cmd.control_mode = 2
                cmd.armed = True
                # cmd.orientation.w, cmd.orientation.x, cmd.orientation.y, cmd.orientation.z = q[0], q[1], q[2], q[3]
                cmd.bodyrates.x, cmd.bodyrates.y, cmd.bodyrates.z = x1[-3], x1[-2], x1[-1]
                cmd.collective_thrust = np.sum(self.u_set[0] ** 2 * self.quadrotorOptimizer.quadrotorModel.kT) / self.quadrotorOptimizer.quadrotorModel.mass
                # cmd.rotor_thrusts = self.u_set[0] ** 2 * self.quadrotorOptimizer.quadrotorModel.kT
                self.control_cmd_pub.publish(cmd)
                # print(self.p)
                print(self.u_set[self.control_num], cmd.collective_thrust)
                print()
                self.control_num += 1
                if self.control_num == self.N:
                    self.have_uset = False
            rate.sleep()

    def odom_callback(self, msg):
        self.have_odom = True
        self.p = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        self.q = [msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z]
        self.v = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]
        self.w = [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]
        # self.x0 = np.concatenate((self.p, self.v, flatten(quat_to_rotation_matrix(np.array(self.q))), self.w))
        self.x0 = np.concatenate((self.p, self.v, self.q, self.w))

        # self.quadrotorOptimizer.acados_solver.set(0, "lbx", self.x0)
        # self.quadrotorOptimizer.acados_solver.set(0, "ubx", self.x0)

        # self.traj_ref = self.getReference(self.traj_type, 1)
        
        # for i in range(self.N):
        #     yref = np.concatenate((self.traj_ref[i], np.zeros(3), np.array([1, 0, 0, 0]), np.zeros(3)))
        #     if i != self.N - 1:
        #         yref = np.concatenate((self.traj_ref[i], np.zeros(3), np.array([1, 0, 0, 0]), np.zeros(3), np.zeros(4)))
            
        #     self.quadrotorOptimizer.acados_solver.set(i + 1, 'yref', yref)
        #     self.quadrotorOptimizer.acados_solver.set(i, 'u', np.array([458, 458, 458, 458]))

        # time = rospy.Time.now().to_sec()
        # status = self.quadrotorOptimizer.acados_solver.solve()
        # self.lastMPCTime = rospy.Time.now().to_sec()
        # print('runtime: ', self.lastMPCTime - time)
        
        # if status != 0:
        #     print("acados returned status {}".format(status))

        # for i in range(self.N):
        #     self.u_set[i] = self.quadrotorOptimizer.acados_solver.get(i, "u")
        # self.have_uset = True
        # self.control_num = 0

        # print(self.u_set[self.control_num])
        # cmd = Actuators()
        # cmd.angular_velocities = self.u_set[self.control_num]
        # self.control_pub.publish(cmd)
    
    def traj_type_callback(self, msg):
        self.traj_type = msg.data
        self.begin_time = rospy.Time.now().to_sec()

    def QuadMPCFSM(self, event):
        
        # if self.have_uset:
        #     print(self.u_set[self.control_num])
        #     cmd = Actuators()
        #     cmd.angular_velocities = self.u_set[self.control_num]
        #     self.control_pub.publish(cmd)
        #     self.control_num += 1
        #     if self.control_num == self.N:
        #         self.have_uset = False


        if not(self.have_odom):# or not(self.traj_type):
            return

        self.quadrotorOptimizer.acados_solver.set(0, "lbx", self.x0)
        self.quadrotorOptimizer.acados_solver.set(0, "ubx", self.x0)

        # self.traj_ref = self.getReference(self.traj_type, rospy.Time.now().to_sec() - self.begin_time)
        p, v, q, br = self.getReference(self.traj_type, 1)
        
        for i in range(self.N):
            yref = np.concatenate((p[i], v[i], q[i], br[i]))
            if i != self.N - 1:
                yref = np.concatenate((yref, np.zeros(4)))
                # yref = np.concatenate((p[i], v[i], q[i], br[i], np.zeros(4)))
            
            self.quadrotorOptimizer.acados_solver.set(i + 1, 'yref', yref)
            self.quadrotorOptimizer.acados_solver.set(i, 'u', np.array([458, 458, 458, 458]))

        time = rospy.Time.now().to_sec()
        status = self.quadrotorOptimizer.acados_solver.solve()
        self.lastMPCTime = rospy.Time.now().to_sec()
        # print('runtime: ', self.lastMPCTime - time)
        
        # if status != 0:
        #     print("acados returned status {}".format(status))

        for i in range(self.N):
            self.u_set[i] = self.quadrotorOptimizer.acados_solver.get(i, "u")
            self.x_set[i] = self.quadrotorOptimizer.acados_solver.get(i + 1, "x")
        self.have_uset = True
        self.control_num = 0
        x1 = self.x_set[self.control_num]
        
        cmd = Actuators()
        cmd.angular_velocities = self.u_set[0]
        # self.control_pub.publish(cmd)
        cmd = PoseStamped()
        cmd.header.stamp = rospy.Time.now()
        cmd.pose.position.x, cmd.pose.position.y, cmd.pose.position.z = self.x_set[0, 0], self.x_set[0, 1], self.x_set[0, 2]
        # self.control_pose_pub.publish(cmd)
        
        # rmat = np.array([[x1[6], x1[7], x1[8]],
        # [x1[9], x1[10], x1[11]],
        # [x1[12], x1[13], x1[14]]])
        # q = rotation_matrix_to_quat(rmat)

        # q = unit_quat(x1[6: 10])
        # cmd = ControlCommand()
        # cmd.header.stamp = rospy.Time.now()
        # cmd.expected_execution_time = rospy.Time.now()
        # cmd.control_mode = 1
        # cmd.armed = True
        # cmd.orientation.w, cmd.orientation.x, cmd.orientation.y, cmd.orientation.z = q[0], q[1], q[2], q[3]
        # cmd.bodyrates.x, cmd.bodyrates.y, cmd.bodyrates.z = x1[-3], x1[-2], x1[-1]
        # cmd.collective_thrust = np.sum(self.u_set[0] ** 2 * self.quadrotorOptimizer.quadrotorModel.kT) / self.quadrotorOptimizer.quadrotorModel.mass
        # cmd.rotor_thrusts = self.u_set[0] ** 2 * self.quadrotorOptimizer.quadrotorModel.kT
        # self.control_cmd_pub.publish(cmd)
        # print(self.p)
        # print(cmd)
        # print()
        return

    def getReference(self, traj_type, time):
        p = np.zeros((self.N, 3))
        v = np.zeros((self.N, 3))
        q = np.zeros((self.N, 4))
        q[:, 0] = 1
        br = np.zeros((self.N, 3))

        dt = self.t_horizon / self.N
        if traj_type == 0:
            v[:, 2] = 0.1
            point = np.array([0, 0, 5])
            p[0, 0] = self.p[0] + v[0, 0] * dt
            p[0, 1] = self.p[1] + v[0, 1] * dt
            p[0, 2] = self.p[2] + v[0, 2] * dt
            for i in range(self.N - 1):
                p[i + 1, 0] = p[i, 0] + v[i, 0] * dt
                p[i + 1, 1] = p[i, 1] + v[i, 1] * dt
                p[i + 1, 2] = p[i, 2] + v[i, 2] * dt
        elif traj_type == 1:
            p[:, 2] = 5
        return p, v, q, br

    def shutdown_node(self):
        print("closed")

def main():
    QuadMPC()
    
if __name__ == "__main__":
    main()