#!/usr/bin/env python3.6
import imp
import rospy, std_msgs.msg, message_filters, os, yaml
import pandas as pd
import numpy as np
import seaborn as sns
from std_msgs.msg import Int16, Bool
from nav_msgs.msg import Odometry
from mav_msgs.msg import Actuators
from sensor_msgs.msg import Imu
from quadrotor_msgs.msg import ControlCommand, AutopilotFeedback
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_srvs.srv import Empty
from src.utils.utils import *
from src.quad_mpc.quad_optimizer import QuadrotorOptimizer
from src.quad_mpc.quad_model import QuadrotorModel

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
        self.with_drag = rospy.get_param('~with_drag', default='false')
        self.expriment = rospy.get_param('~expriment', default='hover')
        self.experiment_time = rospy.get_param('~experiment_time', default='35')
        self.experiment_times = rospy.get_param('~experiment_times', default='1')
        self.start_point = np.zeros(3)
        self.hover_point = np.zeros(3)
        self.start_point[0] = rospy.get_param('~start_x', default='0')
        self.start_point[1] = rospy.get_param('~start_y', default='0')
        self.start_point[2] = rospy.get_param('~start_z', default='1')
        self.hover_point[0] = rospy.get_param('~hover_x', default='2')
        self.hover_point[1] = rospy.get_param('~hover_y', default='0')
        self.hover_point[2] = rospy.get_param('~hover_z', default='5')
        plot = rospy.get_param('~plot', default='true')
        self.havenot_fit_param = rospy.get_param('~fit_param', default='true')
        self.t_horizon = t_horizon # prediction horizon
        self.N = N # number of discretization steps'
        if self.havenot_fit_param:
            self.experiment_time = 40
            self.experiment_times = 1
            self.record_time = self.experiment_time + 10
        self.pose_error_num = 0
        self.print_cnt = 0
        self.a_data = None

        self.have_odom = False
        self.have_ap_fb = False
        self.have_reach_start_point_cmd = False
        self.trigger = False
        self.reach_start_point = False
        # self.start_record_fit_data = True
        self.start_record = False
        self.finish_tracking = False
        
        # load Drag coefficient
        dir_path = os.path.dirname(os.path.realpath(__file__))
        config_dir = dir_path + '/../config'
        result_dir = dir_path + '/../result'
        self.result_file = os.path.join(result_dir, self.expriment + '_without_drag.txt')
        self.yaml_file = os.path.join(config_dir, quad_name + '.yaml')
        Drag_A = np.zeros((3, 3))
        if self.havenot_fit_param or not(self.with_drag):
            Drag_D = np.zeros((3, 3))
            Drag_kh = 0
            Drag_B = np.zeros((3, 3))
        else:
            try:
                with open(self.yaml_file) as file:
                    config = yaml.load(file)
                    Drag_D = np.diag([config["D_dx"], config["D_dy"], config["D_dz"]])
                    Drag_kh = config["kh"]
                    Drag_A[0, 1] = config["D_ax"]
                    Drag_A[1, 0] = config["D_ay"]
                    Drag_B = np.diag([config["D_bx"], config["D_by"], config["D_bz"]])
                    self.result_file = os.path.join(result_dir, self.expriment + '_with_drag.txt')
            except FileNotFoundError:
                warn_msg = "Tried to load Drag coefficient but the file was not found. Using default Drag coefficient."
                rospy.logwarn(warn_msg)
                Drag_D = np.zeros((3, 3))
                Drag_kh = 0
                Drag_B = np.zeros((3, 3))
        print(Drag_D)
        # load model
        self.quadrotorOptimizer = QuadrotorOptimizer(self.t_horizon, self.N, QuadrotorModel(Drag_D=Drag_D, Drag_kh=Drag_kh, Drag_A=Drag_A, Drag_B=Drag_B))
        # Subscribers
        self.odom_sub = rospy.Subscriber('/' + quad_name + '/ground_truth/odometry', Odometry, self.odom_callback)
        self.trigger_sub = rospy.Subscriber('/' + quad_name + '/trigger', std_msgs.msg.Empty, self.trigger_callback)
        self.ap_fb_sub = rospy.Subscriber('/' + quad_name + '/autopilot/feedback', AutopilotFeedback, self.ap_fb_callback)
        # message filter
        self.imu_sub_filter = message_filters.Subscriber('/' + quad_name + '/ground_truth/imu', Imu)
        self.odom_sub_filter = message_filters.Subscriber('/' + quad_name + '/ground_truth/odometry', Odometry)
        self.motor_sub_filter = message_filters.Subscriber('/' + quad_name + '/motor_speed', Actuators)
        self.TimeSynchronizer = message_filters.TimeSynchronizer([self.imu_sub_filter, self.odom_sub_filter, self.motor_sub_filter], 10)
        # message_filters.ApproximateTimeSynchronizer
        self.TimeSynchronizer.registerCallback(self.TimeSynchronizer_callback)
        self.x_data = np.array([])
        # Publishers
        self.arm_pub = rospy.Publisher('/' + quad_name + '/bridge/arm', Bool, queue_size=1, tcp_nodelay=True)
        self.start_autopilot_pub = rospy.Publisher('/' + quad_name + '/autopilot/start', std_msgs.msg.Empty, queue_size=1, tcp_nodelay=True)
        # self.control_motor_pub = rospy.Publisher('/' + quad_name + '/command/motor_speed', Actuators, queue_size=1, tcp_nodelay=True)
        self.desire_pub = rospy.Publisher('/' + quad_name + '/desire', Odometry, queue_size=1, tcp_nodelay=True)
        self.desire_motor_pub = rospy.Publisher('/' + quad_name + '/desire_motor', Actuators, queue_size=1, tcp_nodelay=True)
        self.control_pose_pub = rospy.Publisher('/' + quad_name + '/autopilot/pose_command', PoseStamped, queue_size=1, tcp_nodelay=True)
        # self.control_vel_pub = rospy.Publisher('/' + quad_name + '/autopilot/velocity_command', TwistStamped, queue_size=1, tcp_nodelay=True)
        # self.control_cmd_pub = rospy.Publisher('/' + quad_name + '/control_command', ControlCommand, queue_size=1, tcp_nodelay=True)
        self.ap_control_cmd_pub = rospy.Publisher('/' + quad_name + '/autopilot/control_command_input', ControlCommand, queue_size=1, tcp_nodelay=True)
        # Trying to unpause Gazebo for 10 seconds.
        rospy.wait_for_service('/gazebo/unpause_physics')
        unpause_gazebo = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        unpaused = unpause_gazebo.call()

        if plot:
            self.getReference(experiment=self.expriment, start_point=self.start_point, hover_point=self.hover_point, time_now=0.1, t_horizon=35, N_node=350, model=self.quadrotorOptimizer.quadrotorModel, plot=True)
            return

        i = 0
        # rospy.loginfo("11Unpaused the Gazebo simulation.")
        while (i <= 10 and not(unpaused)):
            i += 1
            rospy.sleep(1.0)
            unpaused = unpause_gazebo.call()
        if not(unpaused):
            rospy.logfatal("Could not wake up Gazebo.")
            rospy.signal_shutdown("Could not wake up Gazebo.")
        else:
            rospy.loginfo("Unpaused the Gazebo simulation.")
        # rospy.loginfo("22Unpaused the Gazebo simulation.")
        rospy.sleep(1.0)

        arm = Bool()
        arm.data = True
        self.arm_pub.publish(arm)

        # while not self.have_ap_fb:
        #     rospy.sleep(0.5)
        if(self.ap_state == AutopilotFeedback().OFF):
            self.start_autopilot_pub.publish(std_msgs.msg.Empty())
            i = 0
            while (self.ap_state != AutopilotFeedback().HOVER):
                # print(6 - i - 1)
                # i += 1
                rospy.sleep(1.0)
        print("HOVER STATE")
        
        # Timer
        self.timer = rospy.Timer(rospy.Duration(1 / 50), self.QuadMPCFSM)

        rate = rospy.Rate(N / t_horizon)
        while not rospy.is_shutdown():
            if self.finish_tracking:
                # x_sim = self.Simulation(self.x_data, self.motor_data, self.t_data - self.begin_time)
                # draw_data_sim(self.x_data, x_sim, self.motor_data, self.t_data - self.begin_time)
                rospy.sleep(3.0)
                if self.pose_error_max[self.experiment_times_current] > 2:
                    self.pose_error[self.experiment_times_current] = 0
                    self.pose_error_max[self.experiment_times_current] = 0
                    self.pose_error_mean[self.experiment_times_current] = 0
                    self.max_v_w[self.experiment_times_current] = 0
                    self.max_motor_speed[self.experiment_times_current] = 0
                    self.experiment_times_current -= 1
                self.experiment_times_current += 1
                self.finish_tracking = False

                if (self.experiment_times_current < self.experiment_times):
                    self.trigger = True
                    self.begin_time = rospy.Time.now().to_sec()
                else:
                    print("max error : ", self.pose_error_max)
                    print("mean error: ", self.pose_error_mean)
                    print("max v_w   : ", self.max_v_w)
                    print("max motor : ", self.max_motor_speed)

                    if not(self.havenot_fit_param):
                        np.savetxt(self.result_file, np.concatenate((self.pose_error_max[np.newaxis,:], self.pose_error_mean[np.newaxis,:], self.max_v_w[np.newaxis,:], self.max_motor_speed[np.newaxis,:]), axis=0), fmt='%0.8f', delimiter=',')
                        
                        fig=plt.figure()
                        ax1=fig.add_subplot(2,2,1)
                        # ax1.set_title("pose_error_max")
                        ax1.boxplot([self.pose_error_max], labels=['pose_error_max'], showmeans=True)
                        ax1.set_ylabel("m")
                        ax1.grid()

                        ax2=fig.add_subplot(2,2,2)
                        # ax2.set_title("pose_error_mean")
                        ax2.boxplot([self.pose_error_mean], labels=['pose_error_mean'], showmeans=True)
                        ax2.set_ylabel("m")
                        ax2.grid()

                        ax3=fig.add_subplot(2,2,3)
                        # ax3.set_title("max_motor_speed_percent")
                        ax3.boxplot([self.max_motor_speed / self.quadrotorOptimizer.quadrotorModel.model.RotorSpeed_max], labels=['max_motor_speed_percent'], showmeans=True)
                        ax3.grid()

                        ax4=fig.add_subplot(2,2,4)
                        # ax4.set_title("max_v_w")
                        ax4.boxplot([self.max_v_w], labels=['max_v_w'], showmeans=True)
                        ax4.set_ylabel("m/s")
                        ax4.grid()

                        plt.show()

                # return
            if self.havenot_fit_param and self.start_record:
                if (rospy.Time.now().to_sec() - self.start_record_time) >= self.record_time:
                    drag_coefficient = np.squeeze(self.fitParam(self.x_data, self.a_data, self.motor_data, self.t_data)).tolist()
                    print(drag_coefficient)

                    with open(self.yaml_file) as file:
                        config = yaml.load(file)
                    config.update({'D_dx': drag_coefficient[0]})
                    config.update({'D_dy': drag_coefficient[1]})
                    config.update({'D_dz': drag_coefficient[2]})
                    config.update({'kh': drag_coefficient[3]})
                    config.update({'D_ax': drag_coefficient[4]})
                    config.update({'D_ay': drag_coefficient[5]})
                    config.update({'D_bx': drag_coefficient[6]})
                    config.update({'D_by': drag_coefficient[7]})
                    config.update({'D_bz': drag_coefficient[8]})
                    with open(self.yaml_file, 'w') as file:
                        file.write(yaml.dump(config, default_flow_style=False))
                    return
            rate.sleep()

    def odom_callback(self, msg):
        if not(self.have_odom):
            self.have_odom = True
        self.p = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        self.q = [msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z]
        self.v_w = v_dot_q(np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]), np.array(self.q)).tolist()
        self.w = [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]
        self.x0 = np.concatenate((self.p, self.v_w, self.q, self.w))
        return
    
    def trigger_callback(self, msg):
        self.trigger = True
        self.begin_time = rospy.Time.now().to_sec()
        self.experiment_times_current = 0
        self.pose_error = np.zeros(self.experiment_times)
        self.pose_error_max = np.zeros(self.experiment_times)
        self.pose_error_mean = np.zeros(self.experiment_times)
        self.max_v_w = np.zeros(self.experiment_times)
        self.max_motor_speed = np.zeros(self.experiment_times)
        self.reach_times = 0

    def ap_fb_callback(self, msg):
        self.have_ap_fb = True
        self.ap_state =  msg.autopilot_state

    def TimeSynchronizer_callback(self, imu_msg, odom_msg, motor_msg):
        if not(self.start_record):
            return
        p = [odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, odom_msg.pose.pose.position.z]
        q = [odom_msg.pose.pose.orientation.w, odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y, odom_msg.pose.pose.orientation.z]
        v_w = v_dot_q(np.array([odom_msg.twist.twist.linear.x, odom_msg.twist.twist.linear.y, odom_msg.twist.twist.linear.z]), np.array(self.q)).tolist()
        w = [odom_msg.twist.twist.angular.x, odom_msg.twist.twist.angular.y, odom_msg.twist.twist.angular.z]
        a_w = v_dot_q(np.array([imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y, imu_msg.linear_acceleration.z]), np.array(self.q)).tolist()

        if len(self.x_data) == 0:
            self.x_data = np.zeros((1, 13))
            self.x_data[0] = np.concatenate((p, v_w, q, w))
            self.a_data = np.zeros((1, 3))
            self.a_data[0] = np.array(a_w)
            self.motor_data = np.zeros((1, 4))
            self.motor_data[0] = motor_msg.angular_velocities
            self.t_data = np.array([odom_msg.header.stamp.to_sec()])
        else:
            self.x_data = np.concatenate((self.x_data, np.concatenate((p, v_w, q, w))[np.newaxis,:]), axis=0)
            self.a_data = np.concatenate((self.a_data, np.array(a_w)[np.newaxis,:]), axis=0)
            self.motor_data = np.concatenate((self.motor_data, np.array(motor_msg.angular_velocities)[np.newaxis,:]), axis=0)
            self.t_data = np.concatenate((self.t_data, np.array([motor_msg.header.stamp.to_sec()])))
            
    def QuadMPCFSM(self, event):
        if not(self.have_odom) or not(self.trigger):
            return

        if not self.reach_start_point:
            # get to initial point
            if not self.have_reach_start_point_cmd:
                print("go to start point")
                cmd = PoseStamped()
                cmd.header.stamp = rospy.Time.now()
                cmd.pose.position.x, cmd.pose.position.y, cmd.pose.position.z = self.start_point[0], self.start_point[1], self.start_point[2]
                self.control_pose_pub.publish(cmd)
                if self.ap_state != AutopilotFeedback().HOVER:
                    self.have_reach_start_point_cmd = True
                else:
                    return
            if (self.ap_state == AutopilotFeedback().HOVER):
                self.reach_times += 1
                if self.reach_times > 50:
                    self.reach_times = 0
                    self.reach_start_point = True
                    self.begin_time = rospy.Time.now().to_sec()
                    self.start_record = True
                    self.start_record_time = rospy.Time.now().to_sec()
                    self.have_reach_start_point_cmd = False
                    print("arrive start point")
            return
        elif (not self.finish_tracking):
            p, v, q, br, u = self.getReference(experiment=self.expriment, start_point=self.start_point, hover_point=self.hover_point, time_now=rospy.Time.now().to_sec() - self.begin_time, t_horizon=self.t_horizon, N_node=self.N, model=self.quadrotorOptimizer.quadrotorModel, plot=False)
            error_pose = np.linalg.norm(np.array(self.p) - p[0])
            error_vel  = np.linalg.norm(np.array(self.v_w) - v[0])
            error_q    = np.linalg.norm(np.array(self.q) - q[0])
            error_br   = np.linalg.norm(np.array(self.w) - br[0])
            self.pose_error[self.experiment_times_current] += error_pose
            self.pose_error_num += 1
            self.pose_error_max[self.experiment_times_current] = max(self.pose_error_max[self.experiment_times_current], error_pose)
            if self.expriment == 'hover':
                if (np.linalg.norm(np.array(self.p) - self.hover_point) < 0.05):
                    self.reach_times += 1
                    if self.reach_times > 500:
                        self.finish_tracking = True
                        self.reach_times = 0    
            elif rospy.Time.now().to_sec() - self.begin_time >= self.experiment_time:
                self.finish_tracking = True
        else:
            self.trigger = False
            self.reach_start_point = False
            self.pose_error_mean[self.experiment_times_current] = self.pose_error[self.experiment_times_current] / self.pose_error_num
            self.pose_error_num = 0
            print("max error : ", self.pose_error_max[self.experiment_times_current])
            print("mean error: ", self.pose_error_mean[self.experiment_times_current])
            print("max v_w   : ", self.max_v_w[self.experiment_times_current])
            print("max motor :  {:.2f}, {:.2f}%".format(self.max_motor_speed[self.experiment_times_current], self.max_motor_speed[self.experiment_times_current] / self.quadrotorOptimizer.quadrotorModel.model.RotorSpeed_max * 100))
            return

        v_b = v_dot_q(v[0], quaternion_inverse(np.array(self.q)))
        vb_abs = np.linalg.norm(v_dot_q(self.v_w, quaternion_inverse(np.array(self.q))))
        vw_abs = np.linalg.norm(self.v_w)
        desire = Odometry()
        desire.pose.pose.position.x, desire.pose.pose.position.y, desire.pose.pose.position.z = p[0, 0], p[0, 1], p[0, 2]
        desire.pose.pose.orientation.w, desire.pose.pose.orientation.x, desire.pose.pose.orientation.y, desire.pose.pose.orientation.z = q[0, 0], q[0, 1], q[0, 2], q[0, 3]
        desire.twist.twist.linear.x, desire.twist.twist.linear.y, desire.twist.twist.linear.z = v_b[0], v_b[1], v_b[2]
        desire.twist.twist.angular.x, desire.twist.twist.angular.x, desire.twist.twist.angular.x = br[0, 0], br[0, 1], br[0, 2]
        self.desire_pub.publish(desire)

        max_motor_speed_now = np.max(self.motor_data[-1])
        self.max_motor_speed[self.experiment_times_current] = max(self.max_motor_speed[self.experiment_times_current], max_motor_speed_now)
        self.max_v_w[self.experiment_times_current] = max(self.max_v_w[self.experiment_times_current], vw_abs)

        desire_motor = Actuators()
        desire_motor.angular_velocities = u[0]
        if self.a_data is not(None):
            desire_motor.angles = np.array([error_pose, error_vel, max_motor_speed_now, self.max_motor_speed[self.experiment_times_current], vw_abs, self.a_data[-1, 0], self.a_data[-1, 1], self.a_data[-1, 2] - 9.81, self.v_w[0], self.v_w[1], self.v_w[2]])
        self.desire_motor_pub.publish(desire_motor)

        ubx = np.array([11, np.pi * 2, np.pi * 2, np.pi * 1])
        lbx = np.array([0, -np.pi * 2, -np.pi * 2, -np.pi * 1])
        self.quadrotorOptimizer.acados_solver.set(0, "lbx", self.x0)
        self.quadrotorOptimizer.acados_solver.set(0, "ubx", self.x0)
        for i in range(self.N):
            # if i != 0:
            #     self.quadrotorOptimizer.acados_solver.constraints_set(i, "lbx", lbx)
            #     self.quadrotorOptimizer.acados_solver.constraints_set(i, "ubx", ubx)
            xref = np.concatenate((p[i], v[i], q[i], br[i]))
            if self.quadrotorOptimizer.ocp.cost.cost_type == "NONLINEAR_LS":
                xref = np.concatenate((xref, np.zeros(self.quadrotorOptimizer.ocp.dims.np)))

            # self.quadrotorOptimizer.acados_solver.set(i, 'x', xref)
            self.quadrotorOptimizer.acados_solver.set(i, 'yref', np.concatenate((xref, u[i])))
            self.quadrotorOptimizer.acados_solver.set(i, 'u', u[i])
            # self.quadrotorOptimizer.acados_solver.set(i, 'xdot_guess', xdot_guessref)
        xref = np.concatenate((p[self.N], v[self.N], q[self.N], br[self.N]))
        if self.quadrotorOptimizer.ocp.cost.cost_type == "NONLINEAR_LS":
                xref = np.concatenate((xref, np.zeros(self.quadrotorOptimizer.ocp.dims.np)))
        # self.quadrotorOptimizer.acados_solver.set(self.N, 'x', xref)
        self.quadrotorOptimizer.acados_solver.set(self.N, 'yref', xref)
        # self.quadrotorOptimizer.acados_solver.set(i, 'xdot_guess', xdot_guessref)

        time = rospy.Time.now().to_sec()
        self.quadrotorOptimizer.acados_solver.solve()
        self.lastMPCTime = rospy.Time.now().to_sec()

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

        
        self.print_cnt += 1
        if self.print_cnt > 5:
            self.print_cnt = 0
            print("********",self.expriment, self.experiment_times_current + 1, ':', self.experiment_times, "********")
            print('rest of time: ', self.experiment_time - (rospy.Time.now().to_sec() - self.begin_time))
            print('runtime: ', self.lastMPCTime - time)
            print("pose:        [{:.2f}, {:.2f}, {:.2f}]".format(self.p[0], self.p[1], self.p[2]))
            print("vel:         [{:.2f}, {:.2f}, {:.2f}], norm = {:.2f}".format(self.v_w[0], self.v_w[1], self.v_w[2], vw_abs))
            print("pose error:  [{:.2f}, {:.2f}, {:.2f}], norm = {:.2f}".format(p[0] - self.p[0], p[1] - self.p[1], p[2] - self.p[2], error_pose))
            print("max motor :  {:.2f}, {:.2f}%".format(max_motor_speed_now, max_motor_speed_now / self.quadrotorOptimizer.quadrotorModel.model.RotorSpeed_max * 100))
            print()
        # print("desir vel :  [{:.2f}, {:.2f}, {:.2f}]".format(v[0], v[1], v[2]))
        # print("desir angle: [{:.2f}, {:.2f}, {:.2f}]".format(angle[0], angle[1], angle[2]))
        # print("desir br:    [{:.2f}, {:.2f}, {:.2f}]".format(br[0], br[1], br[2]))
        # print("ref rotor:   [{:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(u[0, 0], u[0, 1], u[0, 2], u[0, 3]))
        # print("rotor:       [{:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(u1[0], u1[1], u1[2], u1[3]))
        # print("rotor error: [{:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(u[0, 0] - u1[0], u[0, 1] - u1[1], u[0, 2] - u1[2], u[0, 3] - u1[3]))
        

        return

    def getReference(self, experiment, start_point, hover_point, time_now, t_horizon, N_node, model, plot):
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
        if experiment == 'hover':
            p[:, 0] = hover_point[0]
            p[:, 1] = hover_point[1]
            p[:, 2] = hover_point[2]
            yaw[:, 0] = np.pi / 2

        elif experiment == 'circle':
            w = 2 # rad/s
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
        elif experiment == 'circle_vertical':
            w = 1 # rad/s
            phi = 0
            p[:, 0] = r * np.cos(w * t + phi)
            p[:, 1] = 0
            p[:, 2] = start_point[2] + r * np.sin(w * t + phi)
            v[:, 0] = - w ** 1 * r * np.sin(w * t + phi)
            v[:, 1] = 0
            v[:, 2] = w ** 1 * r * np.cos(w * t + phi)
            a[:, 0] = - w ** 2 * r * np.cos(w * t + phi)
            a[:, 1] = 0
            a[:, 2] = - w ** 2 * r * np.sin(w * t + phi)
            j[:, 0] = w ** 3 * r * np.sin(w * t + phi)
            j[:, 1] = 0
            j[:, 2] = - w ** 3 * r * np.cos(w * t + phi)
            s[:, 0] = w ** 4 * r * np.cos(w * t + phi)
            s[:, 1] = 0
            s[:, 2] = w ** 4 * r * np.sin(w * t + phi)
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

    def fitParam(self, x_data, a_data, motor_data, t_data):
        self.havenot_fit_param = False
        kT = self.quadrotorOptimizer.quadrotorModel.kT
        mass = self.quadrotorOptimizer.quadrotorModel.mass
        # g = self.quadrotorOptimizer.quadrotorModel.g[2]
        b_drag = np.zeros((3 * len(x_data), 1))
        A_drag = np.zeros((3 * len(x_data), 4))
        d_drag = np.zeros((3 * len(x_data), 1))
        C_drag = np.zeros((3 * len(x_data), 5))
        brDotset = np.gradient(x_data[:, 10:13], t_data, axis=0)
        Inertia = self.quadrotorOptimizer.quadrotorModel.Inertia
        for i in range(len(x_data)):
            # p = x_data[i, :3]
            v = x_data[i, 3:6]
            a = a_data[i]
            # a[2] += g
            q = x_data[i, 6:10]
            br = x_data[i, 10:13]
            brDot = brDotset[i, :]
            temp_input = self.quadrotorOptimizer.quadrotorModel.G.dot(motor_data[i] ** 2)
            # f_thrust = kT * motor_data[i].dot(motor_data[i])
            f_thrust = temp_input[0]
            torque = temp_input[1:]
            rotMat = quat_to_rotation_matrix(q)
            vh = rotMat.T.dot(v.T)
            vh[2] = 0

            # b_drag[i * 3 + 0, 0] = f_thrust * rotMat[0, 2] - mass * a[0]
            # b_drag[i * 3 + 1, 0] = f_thrust * rotMat[1, 2] - mass * a[1]
            # b_drag[i * 3 + 2, 0] = f_thrust * rotMat[2, 2] - mass * (a[2] + g)
            A_drag[i * 3: i * 3 + 3, 0] = rotMat[:,0].dot(v) * rotMat[:,0]
            A_drag[i * 3: i * 3 + 3, 1] = rotMat[:,1].dot(v) * rotMat[:,1]
            A_drag[i * 3: i * 3 + 3, 2] = rotMat[:,2].dot(v) * rotMat[:,2]
            A_drag[i * 3: i * 3 + 3, 3] = vh.dot(vh) * rotMat[:,2]
            b_drag[i * 3: i * 3 + 3, 0] = f_thrust * rotMat[:,2] - mass * a

            C_drag[i * 3, 0] = -rotMat[:,1].dot(v)
            C_drag[i * 3 + 1, 1] = -rotMat[:,0].dot(v)
            C_drag[i * 3: i * 3 + 3, 2:] = -np.diag(br)
            d_drag[i * 3: i * 3 + 3, 0] = Inertia.dot(brDot.T) + crossmat(br).dot(Inertia).dot(br.T) - torque

        return np.concatenate((np.linalg.inv(np.dot(A_drag.T, A_drag)).dot(A_drag.T).dot(b_drag), np.linalg.inv(np.dot(C_drag.T, C_drag)).dot(C_drag.T).dot(d_drag)))

    def shutdown_node(self):
        print("closed")

def main():
    QuadMPC()
    
if __name__ == "__main__":
    main()