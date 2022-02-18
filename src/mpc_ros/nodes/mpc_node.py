#!/usr/bin/env python3.6
import rospy, genpy, math
import numpy as np
from std_msgs.msg import Int16, Bool
from nav_msgs.msg import Odometry
from mav_msgs.msg import Actuators
from quadrotor_msgs.msg import ControlCommand
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Empty
from src.utils.utils import *
from src.quad_mpc.quad_optimizer import QuadrotorOptimizer

'''
RotorS
/hummingbird/command/motor_speed
/hummingbird/command/pose
/hummingbird/command/trajectory
/hummingbird/gazebo/command/motor_speed
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
        self.arm_pub = rospy.Publisher('/' + quad_name + '/bridge/arm', Bool, queue_size=1, tcp_nodelay=True)
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
        # print("********************************11")
        # p, v, qq, br = self.getReference(2, 0)
        # print(qq)

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
        # arm = Bool()
        # arm.data = True
        # self.arm_pub.publish(arm)
        while not rospy.is_shutdown():
            if self.have_uset:
            #     print(self.u_set[self.control_num])
                cmd = Actuators()
                cmd.angular_velocities = self.u_set[self.control_num]
                # self.control_pub.publish(cmd)
                x1 = self.x_set[self.control_num]
                q = unit_quat(x1[6: 10])
                cmd = ControlCommand()
                cmd.header.stamp = rospy.Time.now()
                cmd.expected_execution_time = rospy.Time.now()
                cmd.control_mode = 1 # NONE=0 ATTITUDE=1 BODY_RATES=2 ANGULAR_ACCELERATIONS=3 ROTOR_THRUSTS=4
                cmd.armed = True
                cmd.orientation.w, cmd.orientation.x, cmd.orientation.y, cmd.orientation.z = q[0], q[1], q[2], q[3]
                cmd.bodyrates.x, cmd.bodyrates.y, cmd.bodyrates.z = x1[-3], x1[-2], x1[-1]
                cmd.collective_thrust = np.sum(self.u_set[0] ** 2 * self.quadrotorOptimizer.quadrotorModel.kT) / self.quadrotorOptimizer.quadrotorModel.mass
                cmd.rotor_thrusts = self.u_set[0] ** 2 * self.quadrotorOptimizer.quadrotorModel.kT
                self.control_cmd_pub.publish(cmd)
                print("pose: [{:.2f}, {:.2f}, {:.2f}]".format(self.p[0], self.p[1], self.p[2]))
                print("rotor:[{:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(self.u_set[self.control_num, 0], self.u_set[self.control_num, 1], self.u_set[self.control_num, 2], self.u_set[self.control_num, 3]))
                print()
                self.control_num += 1
                if self.control_num == self.N:
                    self.have_uset = False
            rate.sleep()

    def odom_callback(self, msg):
        if not(self.have_odom):
            self.have_odom = True
            # arm = Bool()
            # arm.data = True
            # self.arm_pub.publish(arm)
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
        return
    
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


        if not(self.have_odom) or not(self.traj_type):# or not(self.traj_type)
            return

        self.quadrotorOptimizer.acados_solver.set(0, "lbx", self.x0)
        self.quadrotorOptimizer.acados_solver.set(0, "ubx", self.x0)

        # self.traj_ref = self.getReference(self.traj_type, rospy.Time.now().to_sec() - self.begin_time)
        p, v, q, br = self.getReference(self.traj_type, rospy.Time.now().to_sec() - self.begin_time)
        
        for i in range(self.N):
            yref = np.concatenate((p[i], v[i], q[i], br[i]))
            if i != self.N - 1:
                yref = np.concatenate((yref, np.zeros(4)))
                # yref = np.concatenate((p[i], v[i], q[i], br[i], np.zeros(4)))
            
            self.quadrotorOptimizer.acados_solver.set(i + 1, 'yref', yref)
            # self.quadrotorOptimizer.acados_solver.set(i, 'u', np.array([458, 458, 458, 458]))

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
        a = np.zeros((self.N, 3))
        j = np.zeros((self.N, 3))
        yaw = np.zeros((self.N, 1))
        yawdot = np.zeros((self.N, 1))
        dt = self.t_horizon / self.N
        if traj_type == 1:
            # q = np.zeros((self.N, 4))
            # q[:, 0] = 1
            # br = np.zeros((self.N, 3))
            p[:, 2] = 1
            # v[:, 2] = 0.1
            # point = np.array([0, 0, 5])
            # p[0, 0] = self.p[0] + v[0, 0] * dt
            # p[0, 1] = self.p[1] + v[0, 1] * dt
            # p[0, 2] = self.p[2] + v[0, 2] * dt
            # for i in range(self.N - 1):
            #     # p[i + 1, 0] = p[i, 0] + v[i, 0] * dt
            #     # p[i + 1, 1] = p[i, 1] + v[i, 1] * dt
            #     if(p[i, 2] + v[i, 2] * dt >= 2):
            #         p[i + 1, 2] = 2
            #         v[i, 2] = 0
            #     else:
            #         p[i + 1, 2] = p[i, 2] + v[i, 2] * dt
        elif traj_type == 2:
            r = 2
            w = 0.05 # rad/s
            t = np.arange(time + dt, time + dt * self.N + dt, dt)
            phi = 0
            p[:, 0] = r * np.cos(w * t + phi) - r
            p[:, 1] = r * np.sin(w * t + phi)
            p[:, 2] = 1
            v[:, 0] = - w ** 1 * r * np.sin(w * t + phi)
            v[:, 1] = w ** 1 * r * np.cos(w * t + phi)
            v[:, 2] = 0
            a[:, 0] = - w ** 2 * r * np.cos(w * t + phi)
            a[:, 1] = - w ** 2 * r * np.sin(w * t + phi)
            a[:, 2] = 0
            j[:, 0] = w ** 3 * r * np.sin(w * t + phi)
            j[:, 1] = - w ** 3 * r * np.cos(w * t + phi)
            j[:, 2] = 0
            q = self.getReferenceQuaternions(v, a, yaw)
            br = self.getReferenceBodyrates(j, yawdot)
        return p, v, q, br

    def getReferenceQuaternions(self, v, a, yaw):
        q = np.zeros((self.N, 4))
        m = self.quadrotorOptimizer.quadrotorModel.mass
        g = np.array([0, 0, 9.81])
        d = self.quadrotorOptimizer.quadrotorModel.D
        dx, dy = d[0, 0], d[1, 1]
        for i in range(self.N):
            # xc = np.array([np.cos(yaw[i]), np.sin(yaw[i]), 0])
            yc = np.array([-math.sin(yaw[i]), math.cos(yaw[i]), 0])
            alpha = m * (a[i] + g) + dx * v[i]
            beta = m * (a[i] + g) + dy * v[i]
            # print(alpha)
            xb = v1_cross_v2(yc, alpha)
            xb = xb / np.linalg.norm(xb)
            yb = v1_cross_v2(beta, xb)
            yb = yb / np.linalg.norm(yb)
            zb = v1_cross_v2(xb, yb)
            rotation_mat = np.concatenate((xb.reshape(3, 1), yb.reshape(3, 1), zb.reshape(3, 1)), axis=1)
            # print(rotation_mat)
            q[i] = rotation_matrix_to_quat(rotation_mat)
        return q

    def getReferenceBodyrates(self, j, yawdot):
        br = np.zeros((self.N, 3))
        return br

    def shutdown_node(self):
        print("closed")

def main():
    QuadMPC()
    
if __name__ == "__main__":
    main()