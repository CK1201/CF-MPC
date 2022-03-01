#!/usr/bin/env python3.6
import rospy, math
import numpy as np
from std_msgs.msg import Int16, Bool
from nav_msgs.msg import Odometry
from mav_msgs.msg import Actuators
from quadrotor_msgs.msg import ControlCommand
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Empty
from src.utils.utils import *
from src.quad_mpc.quad_optimizer1 import QuadrotorOptimizer

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
    def __init__(self, t_horizon = 1, N = 10) -> None:
        rospy.init_node("mpc_node1")
        quad_name = rospy.get_param('quad_name', default='hummingbird')
        self.t_horizon = t_horizon # prediction horizon
        self.N = N # number of discretization steps
        self.have_odom = False
        self.have_motor = False
        self.traj_type = 1
        # load model
        self.quadrotorOptimizer = QuadrotorOptimizer(t_horizon, N)
        # Subscribers
        self.odom_sub = rospy.Subscriber('/' + quad_name + '/ground_truth/odometry', Odometry, self.odom_callback)
        self.traj_type_sub = rospy.Subscriber('/' + quad_name + '/traj_type', Int16, self.traj_type_callback)
        self.motor_speed_sub = rospy.Subscriber('/' + quad_name + '/motor_speed', Actuators, self.motor_speed_callback)
        # Publishers
        self.arm_pub = rospy.Publisher('/' + quad_name + '/bridge/arm', Bool, queue_size=1, tcp_nodelay=True)
        self.control_motor_pub = rospy.Publisher('/' + quad_name + '/command/motor_speed', Actuators, queue_size=1, tcp_nodelay=True)
        # self.control_pose_pub = rospy.Publisher('/' + quad_name + '/command/pose', PoseStamped, queue_size=1, tcp_nodelay=True)
        self.control_cmd_pub = rospy.Publisher('/' + quad_name + '/control_command', ControlCommand, queue_size=1, tcp_nodelay=True)
        # Trying to unpause Gazebo for 10 seconds.
        rospy.wait_for_service('/gazebo/unpause_physics')
        unpause_gazebo = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        unpaused = unpause_gazebo.call()
        self.u_set = np.zeros((self.N, self.quadrotorOptimizer.ocp.model.u.size()[0]))
        self.x_set = np.zeros((self.N, self.quadrotorOptimizer.ocp.model.x.size()[0]))
        self.have_uset = False
        self.begin_time = rospy.Time.now().to_sec()
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
        self.timer = rospy.Timer(rospy.Duration(t_horizon / 2), self.QuadMPCFSM)

        arm = Bool()
        arm.data = True
        self.arm_pub.publish(arm)

        rate = rospy.Rate(N / t_horizon)
        while not rospy.is_shutdown():
            if self.have_uset:
                x1 = self.x_set[self.control_num, :-4]
                u1 = self.x_set[self.control_num, -4:]

                # motor speed
                cmd = Actuators()
                cmd.angular_velocities = u1
                # self.control_motor_pub.publish(cmd)

                # controller
                q = x1[6: 10]
                br = x1[-3:]
                cmd = ControlCommand()
                cmd.header.stamp = rospy.Time.now()
                cmd.expected_execution_time = rospy.Time.now()
                cmd.control_mode = 4 # NONE=0 ATTITUDE=1 BODY_RATES=2 ANGULAR_ACCELERATIONS=3 ROTOR_THRUSTS=4
                cmd.armed = True
                cmd.orientation.w, cmd.orientation.x, cmd.orientation.y, cmd.orientation.z = q[0], q[1], q[2], q[3]
                cmd.bodyrates.x, cmd.bodyrates.y, cmd.bodyrates.z = br[0], br[1], br[2]
                cmd.collective_thrust = np.sum(u1 ** 2 * self.quadrotorOptimizer.quadrotorModel.kT) / self.quadrotorOptimizer.quadrotorModel.mass
                cmd.rotor_thrusts = u1 ** 2 * self.quadrotorOptimizer.quadrotorModel.kT
                self.control_cmd_pub.publish(cmd)
                print("pose: [{:.2f}, {:.2f}, {:.2f}]".format(self.p[0], self.p[1], self.p[2]))
                print("rotor:[{:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(u1[0], u1[1], u1[2], u1[3]))
                print()
                self.control_num += 1
                if self.control_num == self.N:
                    self.have_uset = False
            rate.sleep()

    def odom_callback(self, msg):
        if not(self.have_odom):
            self.have_odom = True
            arm = Bool()
            arm.data = True
            self.arm_pub.publish(arm)
        self.p = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        self.q = [msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z]
        self.v = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]
        self.w = [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]
        self.x0 = np.concatenate((self.p, self.v, self.q, self.w))
        return
    
    def traj_type_callback(self, msg):
        self.traj_type = msg.data
        self.begin_time = rospy.Time.now().to_sec()
        return

    def motor_speed_callback(self, msg):
        self.have_motor = True
        self.u = np.array(msg.angular_velocities)
        return

    def QuadMPCFSM(self, event):
        if not(self.have_odom) or not(self.traj_type) or not(self.have_motor):# 
            return

        x0 = np.concatenate((self.x0, self.u))
        self.quadrotorOptimizer.acados_solver.set(0, "lbx", x0)
        self.quadrotorOptimizer.acados_solver.set(0, "ubx", x0)

        p, v, q, br, u = self.getReference(self.traj_type, rospy.Time.now().to_sec() - self.begin_time)
        for i in range(self.N):
            yref = np.concatenate((p[i], v[i], q[i], br[i], u[i]))
            if i != self.N - 1:
                yref = np.concatenate((yref, np.zeros(4)))
            
            self.quadrotorOptimizer.acados_solver.set(i + 1, 'yref', yref)
            self.quadrotorOptimizer.acados_solver.set(i, 'u', np.zeros(4))

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

        print(self.u_set)
        print(self.x_set)

        # x1 = self.x_set[0]
        # x2 = self.x_set[1]
        # u1 = self.u_set[0]
        # dt = self.t_horizon / self.N
        # x2_ = self.quadrotorOptimizer.quadrotorModel.Simulation(x1[:3], x1[3:6], x1[6:10], x1[10:], u1, dt)
        # print(x2)
        # print(x2_)
        return

    def getReference(self, traj_type, time):
        p = np.zeros((self.N, 3))
        v = np.zeros((self.N, 3))
        a = np.zeros((self.N, 3))
        j = np.zeros((self.N, 3))
        s = np.zeros((self.N, 3))
        yaw = np.zeros((self.N, 1))
        yawdot = np.zeros((self.N, 1))
        yawdotdot = np.zeros((self.N, 1))
        dt = self.t_horizon / self.N
        t = np.arange(time + dt, time + dt * self.N + dt, dt)
        if traj_type == 1:
            q = np.zeros((self.N, 4))
            q[:, 0] = 1
            br = np.zeros((self.N, 3))
            p[:, 2] = 1
            u = np.ones((self.N, 4)) * 457
        elif traj_type == 2:
            r = 2
            w = 0.01 # rad/s
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
            s[:, 0] = w ** 4 * r * np.cos(w * t + phi)
            s[:, 1] = w ** 4 * r * np.sin(w * t + phi)
            s[:, 2] = 0

            q, br, u = self.getReferenceAll(v=v, a=a, j=j, s=s, yaw=yaw, yawdot=yawdot, yawdotdot=yawdotdot)
        return p, v, q, br, u
    
    def getReferenceAll(self, v, a, j, s, yaw, yawdot, yawdotdot):
        q = np.zeros((self.N, 4))
        br = np.zeros((self.N, 3))
        brdot = np.zeros((self.N, 3))
        u = np.zeros((self.N, 4))
        thrust = np.zeros((self.N,))
        m = self.quadrotorOptimizer.quadrotorModel.mass
        g = np.array([0, 0, 9.81])
        Ginv = np.linalg.inv(self.quadrotorOptimizer.quadrotorModel.G)
        Inertia = self.quadrotorOptimizer.quadrotorModel.Inertia
        d = self.quadrotorOptimizer.quadrotorModel.D
        dx, dy, dz = d[0, 0], d[1, 1], d[2, 2]
        kh = self.quadrotorOptimizer.quadrotorModel.kh
        for i in range(self.N):
            # Quaternions
            xc = np.array([math.cos(yaw[i]), math.sin(yaw[i]), 0])
            yc = np.array([-math.sin(yaw[i]), math.cos(yaw[i]), 0])
            alpha = m * (a[i] + g) + dx * v[i]
            beta = m * (a[i] + g) + dy * v[i]
            
            xb = v1_cross_v2(yc, alpha)
            xb = xb / np.linalg.norm(xb)
            yb = v1_cross_v2(beta, xb)
            yb = yb / np.linalg.norm(yb)
            zb = v1_cross_v2(xb, yb)
            rotation_mat = np.concatenate((xb.reshape(3, 1), yb.reshape(3, 1), zb.reshape(3, 1)), axis=1)
            q[i] = rotation_matrix_to_quat(rotation_mat)
            drag = kh * (np.dot(xb, v[i].T) ** 2 + np.dot(yb, v[i].T) ** 2)
            thrust[i] = np.dot(zb, m * a[i].T + m * g.T + dz * v[i].T) - drag
            

            # Bodyrates
            A_br = np.zeros((3, 3))
            b_br = np.zeros((3,))
            A_br[0, 1] = thrust[i] + (dx - dz) * np.dot(zb, v[i].T) + drag
            A_br[0, 2] = (dy - dx) * np.dot(yb, v[i].T)
            A_br[1, 0] = thrust[i] + (dy - dz) * np.dot(zb, v[i].T) + drag
            A_br[1, 2] = (dx - dy) * np.dot(xb, v[i].T)
            A_br[2, 1] = -np.dot(yc, zb.T)
            A_br[2, 2] = np.linalg.norm(v1_cross_v2(yc, zb))
            b_br[0] = m * np.dot(xb, j[i].T) + dx * np.dot(xb, a[i].T)
            b_br[1] = -m * np.dot(yb, j[i].T) - dy * np.dot(yb, a[i].T)
            b_br[2] = yawdot[i] * np.dot(xc, xb.T)
            br[i] = np.linalg.solve(A_br, b_br)
            brx, bry, brz = br[i, 0], br[i, 1], br[i, 2]

            # Bodyratesdot
            b_brdot = np.zeros((3,))
            b_brdot[0] = m * np.dot(xb, s[i].T) + m * np.dot(brz * yb - bry * zb, j[i].T) + dx * (brz * np.dot(yb, a[i].T) - bry * np.dot(zb, a[i].T) + np.dot(xb, j[i].T)) - brz * (dx - dy) #+ bry * (thrustdot[i])
            b_brdot[1] = -m * np.dot(yb, s[i].T) - m * np.dot(-brz * xb + brx * zb, j[i].T) - dy * (-brz * np.dot(xb, a[i].T) + brx * np.dot(zb, a[i].T) + np.dot(yb, j[i].T)) - brz * (dx - dy) #- brx * (thrustdot[i])
            b_brdot[2] = yawdotdot[i] * np.dot(xc, xb.T) + (yawdot[i] ** 2 + bry ** 2) * np.dot(yc, xb.T) - 2 * yawdot[i] * bry * np.dot(xc, zb.T) - brx * bry * np.dot(yc, yb.T) + yawdot[i] * brz * np.dot(xc, yb.T)
            brdot[i] = np.linalg.solve(A_br, b_brdot)

            # u
            tao = np.dot(Inertia, brdot[i].T) + np.dot(v1_cross_v2(br[i], Inertia), br[i].T)
            u[i] = np.sqrt(np.dot(Ginv, np.array([thrust[i], tao[0], tao[1], tao[2]]).T))

        return q, br, u

    def shutdown_node(self):
        print("closed")

def main():
    QuadMPC()
    
if __name__ == "__main__":
    main()