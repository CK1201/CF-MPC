#!/usr/bin/env python3.6
import imp
from telnetlib import PRAGMA_HEARTBEAT

from pandas import array
import rospy, std_msgs.msg, message_filters
import numpy as np
from std_msgs.msg import Int16, Bool
from nav_msgs.msg import Odometry
from mav_msgs.msg import Actuators
from sensor_msgs.msg import Imu
from quadrotor_msgs.msg import ControlCommand, AutopilotFeedback
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_srvs.srv import Empty
from src.utils.utils import *
from src.quad_mpc.quad_optimizer import QuadrotorOptimizer
# from message_filters import Subscriber, TimeSynchronizer

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
        rospy.init_node("mpc_node")
        quad_name = rospy.get_param('~quad_name', default='hummingbird')
        self.expriment = rospy.get_param('~expriment', default='hover')
        self.start_point = np.zeros(3)
        self.start_point[0] = rospy.get_param('~hover_x', default='2')
        self.start_point[1] = rospy.get_param('~hover_y', default='0')
        self.start_point[2] = rospy.get_param('~hover_z', default='5')
        plot = rospy.get_param('~plot', default='true')
        self.havenot_fit_param = rospy.get_param('~fit_param', default='false')
        self.t_horizon = t_horizon # prediction horizon
        self.N = N # number of discretization steps
        self.have_odom = False
        self.have_ap_fb = False
        self.trigger = False
        self.reach_start_point = False
        self.start_record_fit_data = True
        self.start_record = False
        # load model
        self.quadrotorOptimizer = QuadrotorOptimizer(self.t_horizon, self.N)
        # Subscribers
        self.odom_sub = rospy.Subscriber('/' + quad_name + '/ground_truth/odometry', Odometry, self.odom_callback)
        self.trigger_sub = rospy.Subscriber('/' + quad_name + '/trigger', std_msgs.msg.Empty, self.trigger_callback)
        self.ap_fb_sub = rospy.Subscriber('/' + quad_name + '/autopilot/feedback', AutopilotFeedback, self.ap_fb_callback)
        # message filter
        # if self.havenot_fit_param:
        self.imu_sub_filter = message_filters.Subscriber('/' + quad_name + '/ground_truth/imu', Imu)
        self.odom_sub_filter = message_filters.Subscriber('/' + quad_name + '/ground_truth/odometry', Odometry)
        self.motor_sub_filter = message_filters.Subscriber('/' + quad_name + '/motor_speed', Actuators)
        self.TimeSynchronizer = message_filters.TimeSynchronizer([self.imu_sub_filter, self.odom_sub_filter, self.motor_sub_filter], 1)
        self.TimeSynchronizer.registerCallback(self.TimeSynchronizer_callback)
        self.x_data = np.array([])
        # Publishers
        self.arm_pub = rospy.Publisher('/' + quad_name + '/bridge/arm', Bool, queue_size=1, tcp_nodelay=True)
        self.start_autopilot_pub = rospy.Publisher('/' + quad_name + '/autopilot/start', std_msgs.msg.Empty, queue_size=1, tcp_nodelay=True)
        # self.control_motor_pub = rospy.Publisher('/' + quad_name + '/command/motor_speed', Actuators, queue_size=1, tcp_nodelay=True)
        self.desire_pub = rospy.Publisher('/' + quad_name + '/desire', Odometry, queue_size=1, tcp_nodelay=True)
        self.desire_motor_pub = rospy.Publisher('/' + quad_name + '/desire_motor', Actuators, queue_size=1, tcp_nodelay=True)
        # self.control_vel_pub = rospy.Publisher('/' + quad_name + '/autopilot/velocity_command', TwistStamped, queue_size=1, tcp_nodelay=True)
        # self.control_cmd_pub = rospy.Publisher('/' + quad_name + '/control_command', ControlCommand, queue_size=1, tcp_nodelay=True)
        self.ap_control_cmd_pub = rospy.Publisher('/' + quad_name + '/autopilot/control_command_input', ControlCommand, queue_size=1, tcp_nodelay=True)
        # Trying to unpause Gazebo for 10 seconds.
        rospy.wait_for_service('/gazebo/unpause_physics')
        unpause_gazebo = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        unpaused = unpause_gazebo.call()
        # self.u_set = np.zeros((self.N, self.quadrotorOptimizer.ocp.model.u.size()[0]))
        # self.x_set = np.zeros((self.N + 1, self.quadrotorOptimizer.ocp.model.x.size()[0]))
        self.have_uset = False
        self.begin_time = rospy.Time.now().to_sec()
        self.start_record_time = rospy.Time.now().to_sec()
        self.record_time = 5
        self.reach_last = False
        self.finish_tracking = False
        self.pose_error = 0
        self.pose_error_max = 0
        self.pose_error_num = 0

        if plot:
            self.getReference(experiment=self.expriment, start_point=self.start_point, time_now=0.1, t_horizon=35, N_node=350, model=self.quadrotorOptimizer.quadrotorModel, plot=True)
            return

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

        i = 0
        while (i < 1):
            i += 1
            rospy.sleep(1.0)

        arm = Bool()
        arm.data = True
        self.arm_pub.publish(arm)

        # while not self.have_ap_fb:
        #     rospy.sleep(0.5)
        if(self.ap_state == AutopilotFeedback().OFF):
            self.start_autopilot_pub.publish(std_msgs.msg.Empty())
            i = 0
            while (i < 6):
                print(6 - i - 1)
                i += 1
                rospy.sleep(1.0)

        # get to initial point
        # initial_point = np.array([0, 0, 3])
        # cmd = PoseStamped()
        # cmd.header.stamp = rospy.Time.now()
        # cmd.pose.position.x, cmd.pose.position.y, cmd.pose.position.z = initial_point[0], initial_point[1], initial_point[2]
        # self.control_pose_pub.publish(cmd)

        # while(np.linalg.norm(np.array(self.p) - initial_point) > 0.05):
        #     rospy.sleep(0.5)
        
        # Timer
        self.timer = rospy.Timer(rospy.Duration(1 / 50), self.QuadMPCFSM)

        rate = rospy.Rate(N / t_horizon)
        while not rospy.is_shutdown():
            if self.finish_tracking:
                x_sim = self.Simulation(self.x_data, self.motor_data, self.t_data - self.begin_time)
                draw_data_sim(self.x_data, x_sim, self.motor_data, self.t_data - self.begin_time)
                self.finish_tracking = False
            if self.havenot_fit_param:
                print("Rest of the time to record data: ", self.record_time - (rospy.Time.now().to_sec() - self.start_record_time))
                if (rospy.Time.now().to_sec() - self.start_record_time) >= self.record_time:
                    self.fitParam(self.x_data, self.a_data, self.motor_data, self.t_data, True)
                    return
            rate.sleep()

    def odom_callback(self, msg):
        if not(self.have_odom):
            self.have_odom = True
        self.p = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        self.q = [msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z]
        self.v = v_dot_q(np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]), np.array(self.q)).tolist()
        self.w = [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]
        self.x0 = np.concatenate((self.p, self.v, self.q, self.w))
        return
    
    def trigger_callback(self, msg):
        self.trigger = True
        self.reach_start_point = False
        self.reach_last = False
        self.begin_time = rospy.Time.now().to_sec()

    def ap_fb_callback(self, msg):
        self.have_ap_fb = True
        self.ap_state =  msg.autopilot_state

    def TimeSynchronizer_callback(self, imu_msg, odom_msg, motor_msg):
        if not(self.start_record):
            return
        p = [odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, odom_msg.pose.pose.position.z]
        q = [odom_msg.pose.pose.orientation.w, odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y, odom_msg.pose.pose.orientation.z]
        v = v_dot_q(np.array([odom_msg.twist.twist.linear.x, odom_msg.twist.twist.linear.y, odom_msg.twist.twist.linear.z]), np.array(self.q)).tolist()
        w = [odom_msg.twist.twist.angular.x, odom_msg.twist.twist.angular.y, odom_msg.twist.twist.angular.z]
        a = [imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y, imu_msg.linear_acceleration.z]

        if len(self.x_data) == 0:
            self.x_data = np.zeros((1, 13))
            self.x_data[0] = np.concatenate((p, v, q, w))
            self.a_data = np.zeros((1, 3))
            self.a_data[0] = np.array(a)
            self.motor_data = np.zeros((1, 4))
            self.motor_data[0] = motor_msg.angular_velocities
            self.t_data = np.array([motor_msg.header.stamp.to_sec()])
        else:
            self.x_data = np.concatenate((self.x_data, np.concatenate((p, v, q, w))[np.newaxis,:]), axis=0)
            self.a_data = np.concatenate((self.a_data, np.array(a)[np.newaxis,:]), axis=0)
            self.motor_data = np.concatenate((self.motor_data, np.array(motor_msg.angular_velocities)[np.newaxis,:]), axis=0)
            self.t_data = np.concatenate((self.t_data, np.array([motor_msg.header.stamp.to_sec()])))
            
    def QuadMPCFSM(self, event):
        if not(self.have_odom) or not(self.trigger) or self.havenot_fit_param:
            return

        if not self.reach_start_point:
            # get to initial point
            p, v, q, br, u = self.getReference(experiment='hover', start_point=self.start_point, time_now=rospy.Time.now().to_sec() - self.begin_time, t_horizon=self.t_horizon, N_node=self.N, model=self.quadrotorOptimizer.quadrotorModel, plot=False)
            error_pose = np.linalg.norm(np.array(self.p) - p[0])
            error_vel  = np.linalg.norm(np.array(self.v) - v[0])
            error_q    = np.linalg.norm(np.array(self.q) - q[0])
            error_br   = np.linalg.norm(np.array(self.w) - br[0])
            if (np.linalg.norm(np.array(self.p) - self.start_point) < 0.05):
                if not self.reach_last:
                    self.reach_last = True
                else:
                    self.reach_start_point = True
                    self.begin_time = rospy.Time.now().to_sec()
                    self.start_record = True
                    # self.start_x = self.x0
                    # self.motor_data_rec = np.zeros((1,4))
                    # self.motor_data_rec[0] = 
            else:
                self.reach_last = False
        elif (rospy.Time.now().to_sec() - self.begin_time <= 15):
            p, v, q, br, u = self.getReference(experiment=self.expriment, start_point=self.start_point, time_now=rospy.Time.now().to_sec() - self.begin_time, t_horizon=self.t_horizon, N_node=self.N, model=self.quadrotorOptimizer.quadrotorModel, plot=False)
            error_pose = np.linalg.norm(np.array(self.p) - p[0])
            error_vel  = np.linalg.norm(np.array(self.v) - v[0])
            error_q    = np.linalg.norm(np.array(self.q) - q[0])
            error_br   = np.linalg.norm(np.array(self.w) - br[0])
            self.pose_error += error_pose
            self.pose_error_num += 1
            if error_pose > self.pose_error_max:
                self.pose_error_max = error_pose
        else:
            self.trigger = False
            self.start_record = False
            self.pose_error_mean = self.pose_error / self.pose_error_num
            self.pose_error = 0
            self.pose_error_num = 0
            print("error max : ", self.pose_error_max)
            print("error mean: ", self.pose_error_mean)
            self.finish_tracking = True
            return

        v_b = v_dot_q(v[0], quaternion_inverse(np.array(self.q)))
        vb_abs = np.linalg.norm(v_dot_q(self.v, quaternion_inverse(np.array(self.q))))
        desire = Odometry()
        desire.pose.pose.position.x, desire.pose.pose.position.y, desire.pose.pose.position.z = p[0, 0], p[0, 1], p[0, 2]
        desire.pose.pose.orientation.w, desire.pose.pose.orientation.x, desire.pose.pose.orientation.y, desire.pose.pose.orientation.z = q[0, 0], q[0, 1], q[0, 2], q[0, 3]
        desire.twist.twist.linear.x, desire.twist.twist.linear.y, desire.twist.twist.linear.z = v_b[0], v_b[1], v_b[2]
        desire.twist.twist.angular.x, desire.twist.twist.angular.x, desire.twist.twist.angular.x = br[0, 0], br[0, 1], br[0, 2]
        self.desire_pub.publish(desire)

        desire_motor = Actuators()
        desire_motor.angular_velocities = u[0]
        desire_motor.angles = np.array([error_pose, error_vel, error_q, error_br, vb_abs])
        self.desire_motor_pub.publish(desire_motor)

        self.quadrotorOptimizer.acados_solver.set(0, "lbx", self.x0)
        self.quadrotorOptimizer.acados_solver.set(0, "ubx", self.x0)
        for i in range(self.N):
            xref = np.concatenate((p[i], v[i], q[i], br[i], ))
            self.quadrotorOptimizer.acados_solver.set(i, 'x', xref)
            self.quadrotorOptimizer.acados_solver.set(i, 'yref', np.concatenate((xref, u[i])))
            self.quadrotorOptimizer.acados_solver.set(i, 'u', u[i])
            # self.quadrotorOptimizer.acados_solver.set(i, 'xdot_guess', xdot_guessref)
        xref = np.concatenate((p[self.N], v[self.N], q[self.N], br[self.N]))
        self.quadrotorOptimizer.acados_solver.set(self.N, 'x', xref)
        self.quadrotorOptimizer.acados_solver.set(self.N, 'yref', xref)
        # self.quadrotorOptimizer.acados_solver.set(i, 'xdot_guess', xdot_guessref)

        time = rospy.Time.now().to_sec()
        self.quadrotorOptimizer.acados_solver.solve()
        self.lastMPCTime = rospy.Time.now().to_sec()

        

        
        # if status != 0:
        #     print("acados returned status {}".format(status))
        # self.x_set[0] = self.quadrotorOptimizer.acados_solver.get(0, "x")
        # for i in range(self.N):
        #     self.u_set[i] = self.quadrotorOptimizer.acados_solver.get(i, "u")
        #     self.x_set[i + 1] = self.quadrotorOptimizer.acados_solver.get(i + 1, "x")
        # self.have_uset = True
        # self.control_num = 0

        # x1 = self.x_set[self.control_num + 1]
        # u1 = self.u_set[self.control_num]
        x1 = self.quadrotorOptimizer.acados_solver.get(1, "x")
        u1 = self.quadrotorOptimizer.acados_solver.get(0, "u")
        p = x1[: 3]
        v = x1[3: 6]
        q = x1[6: 10]
        br = x1[-3:]
        angle = quaternion_to_euler(q)

        cmd = ControlCommand()
        cmd.header.stamp = rospy.Time.now()
        cmd.expected_execution_time = rospy.Time.now()
        cmd.control_mode = 2 # NONE=0 ATTITUDE=1 BODY_RATES=2 ANGULAR_ACCELERATIONS=3 ROTOR_THRUSTS=4
        cmd.armed = True
        cmd.orientation.w, cmd.orientation.x, cmd.orientation.y, cmd.orientation.z = q[0], q[1], q[2], q[3]
        cmd.bodyrates.x, cmd.bodyrates.y, cmd.bodyrates.z = br[0], br[1], br[2]
        cmd.collective_thrust = np.sum(u1 ** 2 * self.quadrotorOptimizer.quadrotorModel.kT) / self.quadrotorOptimizer.quadrotorModel.mass
        cmd.rotor_thrusts = u1 ** 2 * self.quadrotorOptimizer.quadrotorModel.kT
        self.ap_control_cmd_pub.publish(cmd)

        
        print("********",self.expriment , "********")
        print('runtime: ', self.lastMPCTime - time)
        print("pose:        [{:.2f}, {:.2f}, {:.2f}]".format(self.p[0], self.p[1], self.p[2]))
        print("vel:         [{:.2f}, {:.2f}, {:.2f}]".format(self.v[0], self.v[1], self.v[2]))
        print("desir pose:  [{:.2f}, {:.2f}, {:.2f}]".format(p[0], p[1], p[2]))
        print("desir vel :  [{:.2f}, {:.2f}, {:.2f}]".format(v[0], v[1], v[2]))
        print("desir angle: [{:.2f}, {:.2f}, {:.2f}]".format(angle[0], angle[1], angle[2]))
        print("desir br:    [{:.2f}, {:.2f}, {:.2f}]".format(br[0], br[1], br[2]))
        print("ref rotor:   [{:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(u[0, 0], u[0, 1], u[0, 2], u[0, 3]))
        print("rotor:       [{:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(u1[0], u1[1], u1[2], u1[3]))
        print("rotor error: [{:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(u[0, 0] - u1[0], u[0, 1] - u1[1], u[0, 2] - u1[2], u[0, 3] - u1[3]))
        

        return

    def getReference(self, experiment, start_point, time_now, t_horizon, N_node, model, plot):
        if start_point is None:
            start_point = np.array([5, 0, 5])
        if abs(self.start_point[0]) < 0.1:
            r = 5
        else:
            r = self.start_point[0]
        p = np.zeros((N_node + 1, 3))
        v = np.zeros((N_node + 1, 3))
        a = np.zeros((N_node + 1, 3))
        j = np.zeros((N_node + 1, 3))
        s = np.zeros((N_node + 1, 3))
        yaw = np.zeros((N_node + 1, 1))
        yawdot = np.zeros((N_node + 1, 1))
        yawdotdot = np.zeros((N_node + 1, 1))
        t = time_now + np.linspace(0, t_horizon, N_node + 1)
        dt = t[1] - t[0]
        # print(t)
        if experiment == 'hover':
            p[:, 0] = start_point[0]
            p[:, 1] = start_point[1]
            p[:, 2] = start_point[2]
            yaw[:] = 2 * np.pi
            # u = math.sqrt(self.quadrotorOptimizer.quadrotorModel.g[-1] * self.quadrotorOptimizer.quadrotorModel.mass / self.quadrotorOptimizer.quadrotorModel.kT / 4)
            # u = np.ones((self.N, 4)) * u

        elif experiment == 'circle':
            w = 0.5 # rad/s
            phi = 0
            p[:, 0] = r * np.cos(w * t + phi)
            p[:, 1] = r * np.sin(w * t + phi)
            p[:, 2] = start_point[2]
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
        elif experiment == 'circle_speedup':
            w_rate = 0.03
            # w = t * w_rate # rad/s
            phi = 0
            p[:, 0] = r * np.cos(w_rate * t ** 2 + phi)
            p[:, 1] = r * np.sin(w_rate * t ** 2 + phi)
            p[:, 2] = start_point[2]
            v[:, 0] = - 2 * w_rate * t * r * np.sin(w_rate * t ** 2 + phi)
            v[:, 1] = 2 * w_rate * t * r * np.cos(w_rate * t ** 2 + phi)
            v[:, 2] = 0
            a[:, 0] = - 2 * w_rate * r * np.sin(w_rate * t ** 2 + phi) - (2 * w_rate * t) ** 2 * r * np.cos(w_rate * t ** 2 + phi)
            a[:, 1] = 2 * w_rate * r * np.cos(w_rate * t ** 2 + phi) - (2 * w_rate * t) ** 2 * r * np.sin(w_rate * t ** 2 + phi)
            a[:, 2] = 0
            j[:, 0] = - 4 * w_rate ** 2 * t * r * np.cos(w_rate * t ** 2 + phi) - 8 * w_rate ** 2 * t * r * np.cos(w_rate * t ** 2 + phi) + (2 * w_rate * t) ** 3 * r * np.sin(w_rate * t ** 2 + phi)
            j[:, 1] = - 4 * w_rate ** 2 * t * r * np.sin(w_rate * t ** 2 + phi) - 8 * w_rate ** 2 * t * r * np.sin(w_rate * t ** 2 + phi) - (2 * w_rate * t) ** 3 * r * np.cos(w_rate * t ** 2 + phi)
            j[:, 2] = 0
            s[:, 0] = - 4 * w_rate ** 2 * r * np.cos(w_rate * t ** 2 + phi) + 8 * w_rate ** 3 * t ** 2 * r * np.sin(w_rate * t ** 2 + phi) - 8 * w_rate ** 2 * r * np.cos(w_rate * t ** 2 + phi) + 16 * w_rate ** 3 * t ** 2 * r * np.sin(w_rate * t ** 2 + phi) + 6 * (2 * w_rate * t) ** 2 * w_rate * r * np.sin(w_rate * t ** 2 + phi) + (2 * w_rate * t) ** 4 * r * np.cos(w_rate * t ** 2 + phi)
            s[:, 1] = - 4 * w_rate ** 2 * r * np.sin(w_rate * t ** 2 + phi) - 8 * w_rate ** 3 * t ** 2 * r * np.cos(w_rate * t ** 2 + phi) - 8 * w_rate ** 2 * r * np.sin(w_rate * t ** 2 + phi) - 16 * w_rate ** 3 * t ** 2 * r * np.cos(w_rate * t ** 2 + phi) - 6 * (2 * w_rate * t) ** 2 * w_rate * r * np.cos(w_rate * t ** 2 + phi) - (2 * w_rate * t) ** 4 * r * np.sin(w_rate * t ** 2 + phi)
            s[:, 2] = 0
        elif experiment == 'lemniscate':
            w = 0.5
            phi = 0
            theta = t * w + phi
            p[:, 0] = - r * np.cos(theta) * np.sin(theta) / (1 + np.sin(theta) ** 2)
            p[:, 1] = r * np.cos(theta) / (1 + np.sin(theta) ** 2)
            p[:, 2] = start_point[2]
        elif experiment == 'lemniscate_speedup':
            w_rate = 0.02
            phi = np.pi / 2
            theta = w_rate * t ** 2 + phi
            # p[:, 0] = r * np.cos(theta) * np.sqrt(np.cos(2 * theta))
            # p[:, 1] = r * np.sin(theta) * np.sqrt(np.cos(2 * theta))
            p[:, 0] = - r * np.cos(theta) * np.sin(theta) / (1 + np.sin(theta) ** 2)
            p[:, 1] = r * np.cos(theta) / (1 + np.sin(theta) ** 2)
            p[:, 2] = start_point[2]

        # for i in range(len(p)):
        #     yaw[i] = math.atan2(v[i, 1], v[i, 0])
            # if i == 1:
            #     if yaw[i] < -2.8 and yaw[i - 1] > 2.8:
            #         yawdot[i - 1] = (yaw[i] + 2 * np.pi - yaw[i - 1]) / dt
            #     else:
            #         yawdot[i - 1] = (yaw[i] - yaw[i - 1]) / dt
            # elif i >= 2 and i < len(p) - 1:
            #     if yaw[i] < -2.8 and yaw[i - 2] > 2.8:
            #         yawdot[i - 1] = (yaw[i] + 2 * np.pi - yaw[i - 2]) / 2 / dt 
            #     else:
            #         yawdot[i - 1] = (yaw[i] - yaw[i - 2]) / 2 / dt 
            # elif i == len(p) - 1:
            #     if yaw[i] < -2.8 and yaw[i - 1] > 2.8:
            #         yawdot[i - 1] = (yaw[i] + 2 * np.pi - yaw[i - 2]) / 2 / dt 
            #         yawdot[i] = (yaw[i] + 2 * np.pi - yaw[i - 1]) / dt 
            #     else:
            #         yawdot[i - 1] = (yaw[i] - yaw[i - 2]) / 2 / dt 
            #         yawdot[i] = (yaw[i] - yaw[i - 1]) / dt 
            # yawdot[i] = 1 / (1 + (v[i, 1] / v[i, 0]) ** 2) * (a[i, 1] * v[i, 0] - v[i, 1] * a[i, 0]) * v[i, 0] ** 2
            # yawdotdot[i] = math.atan2(j[i, 1], j[i, 0])
        # yawdot = np.gradient(yaw, axis=0) / dt
        # yawdotdot = np.gradient(yawdot, axis=0) / dt
        # print(v[:10])
        # print(yaw[:100])
        # print(yawdot[:20])
        # fig1=plt.figure()
        # plt.plot(t, yaw, label='yaw')
        # plt.plot(t, yawdot, label='yawdot')
        # # plt.plot(t, yawdotdot, label='yawdotdot')
        # plt.legend()
        # plt.show()

        q, euler_angle, br, u = getReference_Quaternion_Bodyrates_RotorSpeed(v=v, a=a, j=j, s=s, yaw=yaw, yawdot=yawdot, yawdotdot=yawdotdot, model=model, dt=dt)

        if plot:
            # print(np.sqrt(np.array([4,9,16,2])))
            # print(np.linalg.inv(model.G))
            N_node = len(p)
            p_sim = np.zeros((N_node, 3))
            p_sim[0] = p[0]
            v_sim = np.zeros((N_node, 3))
            q_sim = np.zeros((N_node, 4))
            q_sim[0, 0] = 1
            euler_angle_sim = np.zeros((N_node, 3))
            euler_angle_sim[0] = quaternion_to_euler(q_sim[0])
            br_sim = np.zeros((N_node, 3))
            for i in range(N_node - 1):
                p_sim[i + 1], v_sim[i + 1], q_sim[i + 1], br_sim[i + 1] = model.Simulation(p_sim[i], v_sim[i], q_sim[i], br_sim[i], u[i], dt)
                # p_sim[i + 1], v_sim[i + 1], q_sim[i + 1], br_sim[i + 1] = model.Simulation(p_sim[i], v_sim[i], q_sim[i], br[i], u[i], dt)
                euler_angle_sim[i + 1] = quaternion_to_euler(q_sim[i + 1])

            fig=plt.figure(num=1,figsize=(27,18))# ,figsize=(9,9)

            ax1=fig.add_subplot(261) # , projection='3d'
            ax1.set_title("pose")
            # ax1.plot(p[:, 0],p[:, 1], p[:, 2], label='pose')
            ax1.plot(t,p[:,0], label='x')
            ax1.plot(t,p[:,1], label='y')
            ax1.plot(t,p[:,2], label='z')
            ax1.legend()
            ax1.grid()

            ax2=fig.add_subplot(262)
            ax2.set_title("velocity")
            ax2.plot(t,v[:,0], label='x')
            ax2.plot(t,v[:,1], label='y')
            ax2.plot(t,v[:,2], label='z')
            ax2.legend()
            ax2.grid()

            ax3=fig.add_subplot(263)
            ax3.set_title("euler angle")
            ax3.plot(t,euler_angle[:, 0], label='x')
            ax3.plot(t,euler_angle[:, 1], label='y')
            ax3.plot(t,euler_angle[:, 2], '.', label='z')
            ax3.legend()
            ax3.grid()

            ax4=fig.add_subplot(264)
            ax4.set_title("quat")
            ax4.plot(t,q[:,0], label='w')
            ax4.plot(t,q[:,1], label='x')
            ax4.plot(t,q[:,2], label='y')
            ax4.plot(t,q[:,3], label='z')
            ax4.legend()
            ax4.grid()

            ax5=fig.add_subplot(265)
            ax5.set_title("bodyrate")
            ax5.plot(t,br[:,0], label='x')
            ax5.plot(t,br[:,1], label='y')
            ax5.plot(t,br[:,2], label='z')
            ax5.legend()
            ax5.grid()
            
            ax6=fig.add_subplot(266)
            ax6.set_title("motor speed")
            ax6.plot(t,u[:, 0], label='u1')
            ax6.plot(t,u[:, 1], label='u2')
            ax6.plot(t,u[:, 2], label='u3')
            ax6.plot(t,u[:, 3], label='u4')
            ax6.legend()
            ax6.grid()
            
            ax7=fig.add_subplot(267)
            ax7.set_title("sim pose")
            ax7.plot(t,p_sim[:,0], label='x')
            ax7.plot(t,p_sim[:,1], label='y')
            ax7.plot(t,p_sim[:,2], label='z')
            ax7.legend()
            ax7.grid()

            ax8=fig.add_subplot(2,6,8)
            ax8.set_title("sim velocity")
            ax8.plot(t,v_sim[:,0], label='x')
            ax8.plot(t,v_sim[:,1], label='y')
            ax8.plot(t,v_sim[:,2], label='z')
            ax8.legend()
            ax8.grid()

            ax9=fig.add_subplot(269)
            ax9.set_title("sim euler angle")
            ax9.plot(t,euler_angle_sim[:, 0], label='x')
            ax9.plot(t,euler_angle_sim[:, 1], label='y')
            ax9.plot(t,euler_angle_sim[:, 2], '.', label='z')
            ax9.legend()
            ax9.grid()

            ax10=fig.add_subplot(2,6,10)
            ax10.set_title("sim quat")
            ax10.plot(t,q_sim[:,0], label='w')
            ax10.plot(t,q_sim[:,1], label='x')
            ax10.plot(t,q_sim[:,2], label='y')
            ax10.plot(t,q_sim[:,3], label='z')
            ax10.legend()
            ax10.grid()

            ax11=fig.add_subplot(2,6,11)
            ax11.set_title("sim bodyrate")
            ax11.plot(t,br_sim[:,0], label='x')
            ax11.plot(t,br_sim[:,1], label='y')
            ax11.plot(t,br_sim[:,2], label='z')
            ax11.legend()
            ax11.grid()

            ax12=fig.add_subplot(2,6,12, projection='3d')
            ax12.set_title("pose")
            ax12.plot(p[:, 0],p[:, 1], p[:, 2], label='pose')
            ax12.legend()
            ax12.grid()
            plt.show()

        return p, v, q, br, u

    def Simulation(self, x_data, motor_data, t):
        model = self.quadrotorOptimizer.quadrotorModel
        x_sim = np.zeros((len(motor_data), 13))
        x_sim[0] = x_data[0]
        for i in range(len(motor_data) - 1):
            x_sim[i + 1] = np.concatenate((model.Simulation(x_sim[i, :3], x_sim[i, 3:6], x_sim[i, 6:10], x_data[i, 10:], np.abs(motor_data[i]), t[i + 1] - t[i])))
        return x_sim

    def fitParam(self, x_data, a_data, motor_data, t, draw):
        self.havenot_fit_param = False
        t = t - self.start_record_time
        x_sim = self.Simulation(x_data[0], motor_data, t)
        if draw:
            draw_data_sim(x_data, x_sim, motor_data, t)
        return

    def shutdown_node(self):
        print("closed")

def main():
    QuadMPC()
    
if __name__ == "__main__":
    main()