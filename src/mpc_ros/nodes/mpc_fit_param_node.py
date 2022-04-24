#!/usr/bin/env python3.6
import rospy, std_msgs.msg, message_filters, os, yaml, math
import numpy as np
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
from mav_msgs.msg import Actuators
from sensor_msgs.msg import Imu
from quadrotor_msgs.msg import ControlCommand, AutopilotFeedback
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker
from std_srvs.srv import Empty
from src.utils.utils import *
from src.quad_mpc.quad_optimizer import QuadrotorOptimizer
from src.quad_mpc.quad_model import QuadrotorModel

class QuadMPC:
    def __init__(self, t_horizon = 1, N = 10) -> None:
        rospy.init_node("mpc_node")
        quad_name = rospy.get_param('~quad_name', default='hummingbird')
        self.expriment = rospy.get_param('~expriment', default='hover')
        self.experiment_time = rospy.get_param('~experiment_time', default='20')
        self.start_point = np.zeros(3)
        self.hover_point = np.zeros(3)
        self.start_point[0] = rospy.get_param('~start_x', default='0')
        self.start_point[1] = rospy.get_param('~start_y', default='0')
        self.start_point[2] = rospy.get_param('~start_z', default='1')
        self.hover_point[0] = rospy.get_param('~hover_x', default='0')
        self.hover_point[1] = rospy.get_param('~hover_y', default='0')
        self.hover_point[2] = rospy.get_param('~hover_z', default='2')
        self.max_vel = rospy.get_param('~max_vel', default='0.1')
        self.w_rate = rospy.get_param('~w_rate', default='0.03')
        self.heading = rospy.get_param('~heading', default='false')
        self.fit_method = rospy.get_param('~fit_method', default='LFS')
        self.fit_file_num = str(rospy.get_param('~fit_file_num', default='1'))
        self.cost_type = rospy.get_param('~cost_type', default='EXTERNAL')

        if self.start_point[0] < 1:
            r = 10
        else:
            r = self.start_point[0]
        if self.expriment == 'circle_speedup_stay':
            self.stable_time = self.max_vel / (2 * self.w_rate * r) + 5
            self.experiment_time = self.experiment_time + self.stable_time
        elif self.expriment == 'circle_speedup':
            self.stable_time = 0
            self.experiment_time = self.max_vel / (2 * self.w_rate * r)
        else:
            self.stable_time = rospy.get_param('~stable_time', default='5')
        

        self.t_horizon = t_horizon # prediction horizon
        self.N = N # number of discretization steps'
        self.pose_error_num = 0
        self.print_cnt = 0
        self.experiment_times_current = 0
        self.a_data = None

        

        self.have_odom = False
        self.have_motor_speed = False
        self.have_ap_fb = False
        self.have_reach_start_point_cmd = False
        self.have_polyhedron = False
        self.trigger = True
        self.reach_start_point = False
        self.start_record = False
        self.finish_tracking = False
        self.need_create_optimizer = False
        
        # load Drag coefficient
        dir_path = os.path.dirname(os.path.realpath(__file__))
        config_dir = dir_path + '/../config'
        result_dir = dir_path + '/../result'
        # mesh_dir = dir_path + '/../mesh'
        NM_file = 'NM_coeff_' + self.fit_file_num
        self.result_file = os.path.join(result_dir, self.expriment + '_without_drag.txt')
        self.yaml_file = os.path.join(config_dir, quad_name + '_' + self.fit_method + '_' + self.fit_file_num + '.yaml')
        # self.drag_file = os.path.join(config_dir, NM_file + '.npz')
        self.drag_file = os.path.join(config_dir, 'NM_coeff_' + '1' + '.npz')
        self.NM_fun_val_file = os.path.join(config_dir, NM_file + '_fun_val' + '.npz')
        self.Drag_coeff_all_for_now_file = os.path.join(config_dir, NM_file + '_all_for_now' + '.npz')
        # self.mesh_file = os.path.join("file://", mesh_dir, quad_name + '.mesh')
        self.mesh_file = "file:///home/ck1201/workspace/MAS/Traj_Tracking_MPC/src/mpc_ros/mesh/hummingbird.mesh"
        if self.fit_method == "LFS":
            self.cal_type = "LFS"
            # message filter
            self.imu_sub_filter   = message_filters.Subscriber('/' + quad_name + '/ground_truth/imu', Imu)
            self.odom_sub_filter  = message_filters.Subscriber('/' + quad_name + '/ground_truth/odometry', Odometry)
            self.motor_sub_filter = message_filters.Subscriber('/' + quad_name + '/motor_speed', Actuators)
            self.TimeSynchronizer = message_filters.TimeSynchronizer([self.imu_sub_filter, self.odom_sub_filter, self.motor_sub_filter], 10)
            # message_filters.ApproximateTimeSynchronizer
            self.TimeSynchronizer.registerCallback(self.TimeSynchronizer_callback)
            self.x_data = np.array([])

            self.fit_size = 1
            self.Drag_D  = np.zeros((self.fit_size, 3, 3))
            self.Drag_kh = np.zeros((self.fit_size, 1))
            self.Drag_A  = np.zeros((self.fit_size, 3, 3))
            self.Drag_B  = np.zeros((self.fit_size, 3, 3))
            # self.experiment_time = 40
        elif self.fit_method == "NM":
            self.cal_type = "all"
            Drag = np.load(self.drag_file)
            self.Drag_D  = Drag['D']
            self.Drag_kh = Drag['kh']
            self.Drag_A  = Drag['A']
            self.Drag_B  = Drag['B']
            self.fit_size = len(self.Drag_D)
            self.fun_val = np.array([])

        self.experiment_times = self.fit_size
        self.file_num = self.fit_size

        self.pose_error = np.zeros(self.experiment_times)
        self.pose_error_square = np.zeros(self.experiment_times)
        self.pose_error_max = np.zeros(self.experiment_times)
        self.pose_error_mean = np.zeros(self.experiment_times)
        self.pose_error_rmse = np.zeros(self.experiment_times)
        self.max_v_w = np.zeros(self.experiment_times)
        self.max_a_w = np.zeros(self.experiment_times)
        self.max_motor_speed = np.zeros(self.experiment_times)
        self.reach_times = 0
        
        # load model
        self.quadrotorOptimizer = []
        for i in range(self.fit_size):
            self.quadrotorOptimizer.append(QuadrotorOptimizer(self.t_horizon, self.N, QuadrotorModel(Drag_D=self.Drag_D[i], Drag_kh=self.Drag_kh[i], Drag_A=self.Drag_A[i], Drag_B=self.Drag_B[i], need_collision_free=False, useTwoPolyhedron=False), cost_type=self.cost_type, num=i))

        # Subscribers
        self.quad_vis_pub = rospy.Publisher('/' + quad_name + '/quad_odom', Marker, queue_size=1, tcp_nodelay=True)
        self.quad_box_vis_pub = rospy.Publisher('/' + quad_name + '/quad_box', Marker, queue_size=1, tcp_nodelay=True)
        self.odom_sub = rospy.Subscriber('/' + quad_name + '/ground_truth/odometry', Odometry, self.odom_callback)
        self.imu_sub = rospy.Subscriber('/' + quad_name + '/ground_truth/imu', Imu, self.imu_callback)
        self.motor_speed_sub = rospy.Subscriber('/' + quad_name + '/motor_speed', Actuators, self.motor_speed_callback)
        self.trigger_sub = rospy.Subscriber('/' + quad_name + '/trigger', std_msgs.msg.Empty, self.trigger_callback)
        self.ap_fb_sub = rospy.Subscriber('/' + quad_name + '/autopilot/feedback', AutopilotFeedback, self.ap_fb_callback)
        
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
        self.traj_vis_pub = rospy.Publisher('/' + quad_name + '/quad_traj', Marker, queue_size=1, tcp_nodelay=True)
        self.mpc_ref_vis_pub = rospy.Publisher('/' + quad_name + '/mpc_ref', Marker, queue_size=1, tcp_nodelay=True)
        
        # Trying to unpause Gazebo for 10 seconds.
        rospy.wait_for_service('/gazebo/unpause_physics')
        unpause_gazebo = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        unpaused = unpause_gazebo.call()

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
        rospy.sleep(1.0)

        arm = Bool()
        arm.data = True
        self.arm_pub.publish(arm)

        if(self.ap_state == AutopilotFeedback().OFF):
            self.start_autopilot_pub.publish(std_msgs.msg.Empty())
            i = 0
            while (self.ap_state != AutopilotFeedback().HOVER):
                rospy.sleep(1.0)
        print("HOVER STATE")
        
        # Timer
        self.timer = rospy.Timer(rospy.Duration(1 / 50), self.QuadMPCFSM)

        rate = rospy.Rate(N / t_horizon)
        while not rospy.is_shutdown():
            if self.finish_tracking:
                rospy.sleep(3.0)
                if self.pose_error_max[self.experiment_times_current] > 2:
                    self.pose_error[self.experiment_times_current] = 0
                    self.pose_error_square[self.experiment_times_current] = 0
                    self.pose_error_max[self.experiment_times_current] = 0
                    self.pose_error_mean[self.experiment_times_current] = 0
                    self.pose_error_rmse[self.experiment_times_current] = 0
                    self.max_v_w[self.experiment_times_current] = 0
                    self.max_a_w[self.experiment_times_current] = 0
                    self.max_motor_speed[self.experiment_times_current] = 0
                    self.experiment_times_current -= 1
                self.experiment_times_current += 1
                self.finish_tracking = False

                if (self.experiment_times_current < self.experiment_times):
                    self.trigger = True
                    # self.begin_time = rospy.Time.now().to_sec()
                else:
                    print("max  error: ", self.pose_error_max)
                    print("mean error: ", self.pose_error_mean)
                    print("rmse error: ", self.pose_error_rmse)
                    print("max  v_w  : ", self.max_v_w)
                    print("max  a_w  : ", self.max_a_w)
                    # print("max  motor: ", self.max_motor_speed)
                    self.experiment_times_current -= 1
                    if self.fit_method == "LFS":
                        drag_coefficient = np.squeeze(self.fitParam(self.x_data, self.a_data, self.motor_data, self.t_data)).tolist()
                        print("LFS Result: ", drag_coefficient)
                        self.saveParam(drag_coefficient)
                        return
                    elif self.fit_method == "NM":
                        alpha = 1
                        gamma = 2
                        rho = 0.5
                        sigma = 0.5
                        print("pose rmse: ", self.pose_error_rmse)
                        if self.cal_type == "all":
                            pose_error_rmse_last = self.pose_error_rmse
                            sort_index = np.argsort(pose_error_rmse_last)
                            pose_error_rmse_last = np.sort(pose_error_rmse_last)

                            # sort Drag coeff
                            Drag_D  = np.zeros((self.fit_size, 3, 3))
                            Drag_kh = np.zeros((self.fit_size, 1))
                            Drag_A  = np.zeros((self.fit_size, 3, 3))
                            Drag_B  = np.zeros((self.fit_size, 3, 3))
                            for i in range(self.fit_size):
                                Drag_D [i] = self.Drag_D[sort_index[i]]
                                Drag_kh[i] = self.Drag_kh[sort_index[i]]
                                Drag_A [i] = self.Drag_A[sort_index[i]]
                                Drag_B [i] = self.Drag_B[sort_index[i]]
                            self.Drag_D  = Drag_D
                            self.Drag_kh = Drag_kh
                            self.Drag_A  = Drag_A
                            self.Drag_B  = Drag_B

                            # np.savez(self.NM_fun_val_file,fun_val=pose_error_rmse_last)
                            # np.savez(self.Drag_coeff_all_for_now_file,A=self.Drag_A,B=self.Drag_B,D=self.Drag_D,kh=self.Drag_kh)

                            if (pose_error_rmse_last[-1] - pose_error_rmse_last[0]) < 0.02 or pose_error_rmse_last[0] < 0.05:
                                drag_coefficient = np.concatenate((np.sum(self.Drag_D[0], axis=1), self.Drag_kh[0], np.sum(self.Drag_A[0], axis=1)[:2], np.sum(self.Drag_B[0], axis=1))).tolist()
                                print("Nelder Mead Result: ", drag_coefficient)
                                self.saveParam(drag_coefficient)
                                return

                            # calculate reflection
                            Drag_D_mean  = np.mean(self.Drag_D[:-1], axis=0)
                            Drag_kh_mean = np.mean(self.Drag_kh[:-1], axis=0)
                            Drag_A_mean  = np.mean(self.Drag_A[:-1], axis=0)
                            Drag_B_mean  = np.mean(self.Drag_B[:-1], axis=0)
                            Drag_D_reflect  = Drag_D_mean + alpha * (Drag_D_mean - self.Drag_D[-1])
                            Drag_kh_reflect = Drag_kh_mean + alpha * (Drag_kh_mean - self.Drag_kh[-1])
                            Drag_A_reflect  = Drag_A_mean + alpha * (Drag_A_mean - self.Drag_A[-1])
                            Drag_B_reflect  = Drag_B_mean + alpha * (Drag_B_mean - self.Drag_B[-1])

                            self.cal_type = "reflection"
                            self.Drag_D_tobe_created  = np.zeros((1, 3, 3))
                            self.Drag_kh_tobe_created = np.zeros((1, 1))
                            self.Drag_A_tobe_created  = np.zeros((1, 3, 3))
                            self.Drag_B_tobe_created  = np.zeros((1, 3, 3))

                            self.Drag_D_tobe_created[0]  = Drag_D_reflect
                            self.Drag_kh_tobe_created[0] = Drag_kh_reflect
                            self.Drag_A_tobe_created[0]  = Drag_A_reflect
                            self.Drag_B_tobe_created[0]  = Drag_B_reflect

                            self.need_create_optimizer = True
                        elif self.cal_type == "reflection":
                            reflection_rmse = self.pose_error_rmse[0]
                            if reflection_rmse >= pose_error_rmse_last[0] and reflection_rmse < pose_error_rmse_last[-2]:
                                self.Drag_D [-1] = Drag_D_reflect
                                self.Drag_kh[-1] = Drag_kh_reflect
                                self.Drag_A [-1] = Drag_A_reflect
                                self.Drag_B [-1] = Drag_B_reflect
                                self.pose_error_rmse = pose_error_rmse_last
                                self.pose_error_rmse[-1] = reflection_rmse
                                self.finish_tracking = True
                                self.cal_type = "all"
                                # continue
                            elif reflection_rmse < pose_error_rmse_last[0]:
                                Drag_D_expansion  = Drag_D_mean + gamma * (Drag_D_reflect - Drag_D_mean)
                                Drag_kh_expansion = Drag_kh_mean + gamma * (Drag_kh_reflect - Drag_kh_mean)
                                Drag_A_expansion  = Drag_A_mean + gamma * (Drag_A_reflect - Drag_A_mean)
                                Drag_B_expansion  = Drag_B_mean + gamma * (Drag_B_reflect - Drag_B_mean)

                                self.cal_type = "expansion"
                                self.Drag_D_tobe_created  = np.zeros((1, 3, 3))
                                self.Drag_kh_tobe_created = np.zeros((1, 1))
                                self.Drag_A_tobe_created  = np.zeros((1, 3, 3))
                                self.Drag_B_tobe_created  = np.zeros((1, 3, 3))
                                self.Drag_D_tobe_created[0]  = Drag_D_expansion
                                self.Drag_kh_tobe_created[0] = Drag_kh_expansion
                                self.Drag_A_tobe_created[0]  = Drag_A_expansion
                                self.Drag_B_tobe_created[0]  = Drag_B_expansion
                                self.need_create_optimizer = True
                            elif reflection_rmse >= pose_error_rmse_last[-2]:
                                Drag_D_contraction  = Drag_D_mean + rho * (self.Drag_D[-1] - Drag_D_mean)
                                Drag_kh_contraction = Drag_kh_mean + rho * (self.Drag_kh[-1] - Drag_kh_mean)
                                Drag_A_contraction  = Drag_A_mean + rho * (self.Drag_A[-1] - Drag_A_mean)
                                Drag_B_contraction  = Drag_B_mean + rho * (self.Drag_B[-1] - Drag_B_mean)
                                self.cal_type = "contraction"

                                self.Drag_D_tobe_created  = np.zeros((1, 3, 3))
                                self.Drag_kh_tobe_created = np.zeros((1, 1))
                                self.Drag_A_tobe_created  = np.zeros((1, 3, 3))
                                self.Drag_B_tobe_created  = np.zeros((1, 3, 3))

                                self.Drag_D_tobe_created[0]  = Drag_D_contraction
                                self.Drag_kh_tobe_created[0] = Drag_kh_contraction
                                self.Drag_A_tobe_created[0]  = Drag_A_contraction
                                self.Drag_B_tobe_created[0]  = Drag_B_contraction

                                self.need_create_optimizer = True

                                # self.quadrotorOptimizer = []
                                # self.quadrotorOptimizer.append(QuadrotorOptimizer(self.t_horizon, self.N, QuadrotorModel(Drag_D=Drag_D_contraction, Drag_kh=Drag_kh_contraction, Drag_A=Drag_A_contraction, Drag_B=Drag_B_contraction), cost_type=self.cost_type, num=self.file_num))
                                # self.file_num += 1
                                # self.experiment_times = 1

                            else:
                                self.finish_tracking = True
                                self.cal_type = "shrink"
                                # continue
                        elif self.cal_type == "expansion":
                            expansion_rmse = self.pose_error_rmse[0]
                            if expansion_rmse < reflection_rmse:
                                self.Drag_D [-1] = Drag_D_expansion
                                self.Drag_kh[-1] = Drag_kh_expansion
                                self.Drag_A [-1] = Drag_A_expansion
                                self.Drag_B [-1] = Drag_B_expansion
                                self.pose_error_rmse = pose_error_rmse_last
                                self.pose_error_rmse[-1] = expansion_rmse
                            else:
                                self.Drag_D [-1] = Drag_D_reflect
                                self.Drag_kh[-1] = Drag_kh_reflect
                                self.Drag_A [-1] = Drag_A_reflect
                                self.Drag_B [-1] = Drag_B_reflect
                                self.pose_error_rmse = pose_error_rmse_last
                                self.pose_error_rmse[-1] = reflection_rmse
                            self.finish_tracking = True
                            self.cal_type = "all"
                            # continue
                        elif self.cal_type == "contraction":
                            contraction_rmse = self.pose_error_rmse[0]
                            if contraction_rmse < pose_error_rmse_last[-1]:
                                self.Drag_D [-1] = Drag_D_contraction
                                self.Drag_kh[-1] = Drag_kh_contraction
                                self.Drag_A [-1] = Drag_A_contraction
                                self.Drag_B [-1] = Drag_B_contraction
                                self.finish_tracking = True
                                self.cal_type = "all"
                                self.pose_error_rmse = pose_error_rmse_last
                                self.pose_error_rmse[-1] = contraction_rmse
                            else:
                                self.finish_tracking = True
                                self.cal_type = "shrink"
                                # continue
                        elif self.cal_type == "shrink":
                            self.Drag_D_tobe_created  = np.zeros((self.fit_size, 3, 3))
                            self.Drag_kh_tobe_created = np.zeros((self.fit_size, 1))
                            self.Drag_A_tobe_created  = np.zeros((self.fit_size, 3, 3))
                            self.Drag_B_tobe_created  = np.zeros((self.fit_size, 3, 3))

                            self.Drag_D_tobe_created[0]  = self.Drag_D[0]
                            self.Drag_kh_tobe_created[0] = self.Drag_kh[0]
                            self.Drag_A_tobe_created[0]  = self.Drag_A[0]
                            self.Drag_B_tobe_created[0]  = self.Drag_B[0]

                            
                            for i in range(self.fit_size - 1):
                                self.Drag_D[i+1]  = self.Drag_D[0] + sigma * (self.Drag_D[i+1] - self.Drag_D[0])
                                self.Drag_kh[i+1] = self.Drag_kh[0] + sigma * (self.Drag_kh[i+1] - self.Drag_kh[0])
                                self.Drag_A[i+1]  = self.Drag_A[0] + sigma * (self.Drag_A[i+1] - self.Drag_A[0])
                                self.Drag_B[i+1]  = self.Drag_B[0] + sigma * (self.Drag_B[i+1] - self.Drag_B[0])

                                self.Drag_D_tobe_created[i+1]  = self.Drag_D[i+1]
                                self.Drag_kh_tobe_created[i+1] = self.Drag_kh[i+1]
                                self.Drag_A_tobe_created[i+1]  = self.Drag_A[i+1]
                                self.Drag_B_tobe_created[i+1]  = self.Drag_B[i+1]
                            
                            self.need_create_optimizer = True
                            self.cal_type = "all"

                        if len(self.fun_val) == 0:
                            self.fun_val = np.zeros((1, self.fit_size))
                            self.fun_val[0] = pose_error_rmse_last[np.newaxis,:]
                        else:
                            self.fun_val = np.concatenate((self.fun_val, pose_error_rmse_last[np.newaxis,:]), axis=0)
                        np.savez(self.NM_fun_val_file,fun_val=self.fun_val)
                        np.savez(self.Drag_coeff_all_for_now_file,A=self.Drag_A,B=self.Drag_B,D=self.Drag_D,kh=self.Drag_kh)
                        print(self.Drag_A[0])
                        print(self.Drag_B[0])
                        print(self.Drag_D[0])
                        print(self.Drag_kh[0])
                        print(pose_error_rmse_last)

            if self.need_create_optimizer:
                self.need_create_optimizer = False
                self.experiment_times = len(self.Drag_D_tobe_created)
                # print(self.experiment_times)
                for i in range(self.experiment_times):
                    # print(self.Drag_D_tobe_created[i])
                    # print(self.Drag_kh_tobe_created[i])
                    # print(self.Drag_A_tobe_created[i])
                    # print(self.Drag_B_tobe_created[i])
                    self.quadrotorOptimizer[i] = QuadrotorOptimizer(self.t_horizon, self.N, QuadrotorModel(Drag_D=self.Drag_D_tobe_created[i], Drag_kh=self.Drag_kh_tobe_created[i], Drag_A=self.Drag_A_tobe_created[i], Drag_B=self.Drag_B_tobe_created[i]), cost_type=self.cost_type, num=i)

                # print(self.cal_type)

                self.experiment_times_current = 0
                self.pose_error = np.zeros(self.experiment_times)
                self.pose_error_square = np.zeros(self.experiment_times)
                self.pose_error_max = np.zeros(self.experiment_times)
                self.pose_error_mean = np.zeros(self.experiment_times)
                self.pose_error_rmse = np.zeros(self.experiment_times)
                self.max_v_w = np.zeros(self.experiment_times)
                self.max_a_w = np.zeros(self.experiment_times)
                self.max_motor_speed = np.zeros(self.experiment_times)
                self.reach_times = 0
                self.pose_error_num = 0
                self.trigger = True

            rate.sleep()
        return

    def odom_callback(self, msg):
        if not(self.have_odom):
            self.have_odom = True
        self.p = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        self.q = [msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z]
        self.v_w = v_dot_q(np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]), np.array(self.q)).tolist()
        self.w = [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]
        self.x0 = np.concatenate((self.p, self.v_w, self.q, self.w))


        quad_odom = Marker()
        quad_odom.header.frame_id = 'world'
        quad_odom.header.stamp = msg.header.stamp
        quad_odom.ns = "mesh"
        quad_odom.id = 0
        quad_odom.action = Marker.ADD
        quad_odom.mesh_resource = self.mesh_file
        quad_odom.pose.position.x = self.p[0]
        quad_odom.pose.position.y = self.p[1]
        quad_odom.pose.position.z = self.p[2]
        quad_odom.pose.orientation.w = self.q[0]
        quad_odom.pose.orientation.x = self.q[1]
        quad_odom.pose.orientation.y = self.q[2]
        quad_odom.pose.orientation.z = self.q[3]
        quad_odom.scale.x = 1
        quad_odom.scale.y = 1
        quad_odom.scale.z = 1
        quad_odom.type = Marker.MESH_RESOURCE
        quad_odom.mesh_use_embedded_materials = True
        self.quad_vis_pub.publish(quad_odom)


        quad_box = Marker()
        quad_box.header.frame_id = '/world'
        quad_box.header.stamp = rospy.Time.now()
        quad_box.ns = "quad_box"
        quad_box.id = 0
        quad_box.type = Marker.SPHERE_LIST
        quad_box.action = Marker.ADD
        quad_box.pose.orientation.w = 1
        quad_box.pose.orientation.x = 0
        quad_box.pose.orientation.y = 0
        quad_box.pose.orientation.z = 0
        quad_box.color.r = 1
        quad_box.color.g = 0
        quad_box.color.b = 0
        quad_box.color.a = 1
        quad_box.scale.x = 0.05
        quad_box.scale.y = 0.05
        quad_box.scale.z = 0.05

        pos = np.array(self.p)
        # temp = min(len(self.quadrotorOptimizer)-1, self.experiment_times_current)
        boxVertexNp = self.quadrotorOptimizer[self.experiment_times_current].quadrotorModel.model.boxVertexNp
        RotationMat = quat_to_rotation_matrix(unit_quat(np.array(self.q)))
        # if self.have_polyhedron:
        #     polyhedrons = self.PolyhedronArray.polyhedrons
        # cost = 0
        for i in range(len(boxVertexNp)):
            pot = boxVertexNp[i]
            temp_pos = RotationMat.dot(pot[:,np.newaxis]) + pos[:,np.newaxis]
            point = Point()
            point.x = temp_pos[0]
            point.y = temp_pos[1]
            point.z = temp_pos[2]
            quad_box.points.append(point)

            # if self.have_polyhedron:
            #     for k in range(len(polyhedrons[0].points)):
            #         temp = polyhedrons[0].points[k]
            #         point = np.array([temp.x, temp.y, temp.z])
            #         temp = polyhedrons[0].normals[k]
            #         normal = np.array([temp.x, temp.y, temp.z])
            #         b = normal.dot(point)
            #         cost += max(normal.dot(temp_pos) - b, 0)
        # print(cost)
        self.quad_box_vis_pub.publish(quad_box)
        return

    def imu_callback(self, msg):
        if not(self.have_odom):
            return
        self.a_w = v_dot_q(np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]), np.array(self.q)).tolist()
    
    def motor_speed_callback(self, msg):
        self.have_motor_speed = True
        self.motor_speed = np.array(msg.angular_velocities)
    
    def trigger_callback(self, msg):
        self.trigger = True
        self.begin_time = rospy.Time.now().to_sec()
        self.experiment_times_current = 0
        self.pose_error = np.zeros(self.experiment_times)
        self.pose_error_square = np.zeros(self.experiment_times)
        self.pose_error_max = np.zeros(self.experiment_times)
        self.pose_error_mean = np.zeros(self.experiment_times)
        self.pose_error_rmse = np.zeros(self.experiment_times)
        self.max_v_w = np.zeros(self.experiment_times)
        self.max_a_w = np.zeros(self.experiment_times)
        self.max_motor_speed = np.zeros(self.experiment_times)
        self.reach_times = 0
        self.pose_error_num = 0

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
            # yaw = math.pi / 2 / math.pi * 180
            yaw = math.pi / 2
            # print(yaw / 2)
            # print(math.cos(yaw / 2))
            if not self.have_reach_start_point_cmd:
                print("go to start point")
                cmd = PoseStamped()
                cmd.header.stamp = rospy.Time.now()
                cmd.pose.position.x, cmd.pose.position.y, cmd.pose.position.z = self.start_point[0], self.start_point[1], self.start_point[2]
                cmd.pose.orientation.w, cmd.pose.orientation.x, cmd.pose.orientation.y, cmd.pose.orientation.z = math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)
                self.control_pose_pub.publish(cmd)
                rospy.sleep(1.0)
                # if self.ap_state != AutopilotFeedback().HOVER:
                if self.ap_state == AutopilotFeedback().TRAJECTORY_CONTROL or (np.linalg.norm(np.array(self.p) - self.start_point) < 0.1):
                    self.have_reach_start_point_cmd = True
                    # rospy.sleep(5.0)
                else:
                    return
            # if (self.ap_state == AutopilotFeedback().HOVER):
            if (self.ap_state == AutopilotFeedback().HOVER) and (np.linalg.norm(np.array(self.p) - self.start_point) < 0.1):
                self.reach_times += 1
                if self.reach_times > 50:
                    self.reach_times = 0
                    self.reach_start_point = True
                    self.begin_time = rospy.Time.now().to_sec()
                    if self.fit_method == 'LFS':
                        self.start_record = True
                    self.have_reach_start_point_cmd = False
                    print("arrive start point")
            else:
                if self.ap_state != AutopilotFeedback().TRAJECTORY_CONTROL:
                    cmd = PoseStamped()
                    cmd.header.stamp = rospy.Time.now()
                    cmd.pose.position.x, cmd.pose.position.y, cmd.pose.position.z = self.start_point[0], self.start_point[1], self.start_point[2]
                    cmd.pose.orientation.w, cmd.pose.orientation.x, cmd.pose.orientation.y, cmd.pose.orientation.z = math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)
                    self.control_pose_pub.publish(cmd)
                    rospy.sleep(1.0)
            return
        elif (not self.finish_tracking):
            p, v, q, br, u = self.getReference(experiment=self.expriment, start_point=self.start_point, hover_point=self.hover_point, time_now=rospy.Time.now().to_sec() - self.begin_time, t_horizon=self.t_horizon, N_node=self.N, velocity=self.max_vel, w_rate=self.w_rate, model=self.quadrotorOptimizer[self.experiment_times_current].quadrotorModel, plot=False)
            error_pose = np.linalg.norm(np.array(self.p) - p[0])
            error_vel  = np.linalg.norm(np.array(self.v_w) - v[0])
            error_q    = np.linalg.norm(diff_between_q_q(np.array(self.q), q[0])[1:])
            error_br   = np.linalg.norm(np.array(self.w) - br[0])
            if rospy.Time.now().to_sec() - self.begin_time > self.stable_time:
                self.pose_error[self.experiment_times_current] += error_pose
                self.pose_error_square[self.experiment_times_current] += error_pose ** 2
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
            elif self.pose_error_max[self.experiment_times_current] > 2.1:
                self.finish_tracking = True
        else:
            self.trigger = False
            self.reach_start_point = False
            self.pose_error_mean[self.experiment_times_current] = self.pose_error[self.experiment_times_current] / self.pose_error_num
            self.pose_error_rmse[self.experiment_times_current] = math.sqrt(self.pose_error_square[self.experiment_times_current] / self.pose_error_num)
            self.pose_error_num = 0
            print("max error : ", self.pose_error_max[self.experiment_times_current])
            print("max vel :   ", self.max_v_w[self.experiment_times_current])
            print("max acc :   ", self.max_a_w[self.experiment_times_current])
            print("Rmse      : ", self.pose_error_rmse[self.experiment_times_current])
            # print("max motor :  {:.2f}, {:.2f}%".format(self.max_motor_speed[self.experiment_times_current], self.max_motor_speed[self.experiment_times_current] / self.quadrotorOptimizer[self.experiment_times_current].quadrotorModel.model.RotorSpeed_max * 100))
            return
        
        self.runMPC(p=p, v=v, q=q, br=br, u=u, error_pose=error_pose, error_vel=error_vel)
        return

    def getReference(self, experiment, start_point, hover_point, time_now, t_horizon, N_node, velocity, w_rate, model, plot):
        if start_point is None:
            start_point = np.array([5, 0, 5])
        if abs(self.start_point[0]) < 1:
            r = 10
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
            timeAll = np.linalg.norm((hover_point - start_point)) / velocity
            temp_t = np.concatenate((t[np.newaxis,:] / timeAll, np.ones((1, N_node + 1))), axis=0)
            temp_t = temp_t.min(0)[:,np.newaxis]
            temp_t = np.repeat(temp_t, repeats=3, axis=1)
            temp_p = (hover_point - start_point)[np.newaxis,:]
            temp_p = np.repeat(temp_p, repeats=N_node + 1, axis=0)
            p = temp_t * temp_p + np.repeat(start_point[np.newaxis,:], repeats=N_node + 1, axis=0)
            # p[:, 0] = temp[0]
            # p[:, 1] = temp[1]
            # p[:, 2] = temp[2]
            # p = np.repeat(hover_point[np.newaxis,:], repeats=N_node + 1, axis=0)
            # yaw[:, 0] = pi
            # print(p)

        elif experiment == 'circle':
            # w = 2 # rad/s
            w = velocity / r
            phi = 0
            [p, v, a, j, s] = self.getRefCircleHor(r, w, phi, t, start_point[2])
        elif experiment == 'circle_speedup_stay':
            w = velocity / r
            phi = 0
            # stable_time = w / w_rate
            stable_time = velocity / (2 * w_rate * r)
            stable_phi = w_rate * stable_time ** 2 + phi

            for i in range(N_node + 1):
                # if w_rate * t[i] ** 2 < w * t[i]:
                if t[i] < stable_time:
                    temp_t = np.ones((1,1)) * t[i]
                    [p[i], v[i], a[i], j[i], s[i]] = self.getRefCircleSpeedup(r, w_rate, phi, temp_t, start_point[2])
                else:
                    temp_t = np.ones((1,1)) * (t[i] - stable_time)
                    [p[i], v[i], a[i], j[i], s[i]] = self.getRefCircleHor(r, w, stable_phi, temp_t, start_point[2])

        elif experiment == 'circle_speedup':
            phi = 0
            [p, v, a, j, s] = self.getRefCircleSpeedup(r, w_rate, phi, t, start_point[2])
            
        elif experiment == 'circle_vertical':
            w = 1 # rad/s
            phi = 0
            [p, v, a, j, s] = self.getRefCircleVert(r, w, phi, t, start_point[2])
            
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

        if self.heading:
            for i in range(len(p)):
                yaw[i] = math.atan2(v[i, 1], v[i, 0])
        else:
            yaw[:] = 0
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

        mpc_ref = Marker()
        mpc_ref.header.frame_id = '/world'
        mpc_ref.header.stamp = rospy.Time.now()
        mpc_ref.ns = "mpc_ref"
        mpc_ref.id = 0
        mpc_ref.type = Marker.SPHERE_LIST
        mpc_ref.action = Marker.ADD
        mpc_ref.pose.orientation.w = 1
        mpc_ref.pose.orientation.x = 0
        mpc_ref.pose.orientation.y = 0
        mpc_ref.pose.orientation.z = 0
        mpc_ref.color.r = 1
        mpc_ref.color.g = 0
        mpc_ref.color.b = 1
        mpc_ref.color.a = 1
        mpc_ref.scale.x = 0.1
        mpc_ref.scale.y = 0.1
        mpc_ref.scale.z = 0.1

        for i in range(len(p)):
            point = Point()
            point.x = p[i,0]
            point.y = p[i,1]
            point.z = p[i,2]
            mpc_ref.points.append(point)
        self.mpc_ref_vis_pub.publish(mpc_ref)

        return p, v, q, br, u

    def getRefCircleHor(self, r, w, phi, t, height):
        p = np.zeros((len(t), 3))
        v = np.zeros((len(t), 3))
        a = np.zeros((len(t), 3))
        j = np.zeros((len(t), 3))
        s = np.zeros((len(t), 3))
        p[:, 0] = r * np.cos(w * t + phi)
        p[:, 1] = r * np.sin(w * t + phi)
        p[:, 2] = height
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
        return [p, v, a, j, s]

    def getRefCircleVert(self, r, w, phi, t, height):
        p = np.zeros((len(t), 3))
        v = np.zeros((len(t), 3))
        a = np.zeros((len(t), 3))
        j = np.zeros((len(t), 3))
        s = np.zeros((len(t), 3))
        p[:, 0] = r * np.cos(w * t + phi)
        p[:, 1] = 0
        p[:, 2] = height + r * np.sin(w * t + phi)
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
        return [p, v, a, j, s]

    def getRefCircleSpeedup(self, r, w_rate, phi, t, height):
        p = np.zeros((len(t), 3))
        v = np.zeros((len(t), 3))
        a = np.zeros((len(t), 3))
        j = np.zeros((len(t), 3))
        s = np.zeros((len(t), 3))
        p[:, 0] = r * np.cos(w_rate * (t ** 2) + phi)
        p[:, 1] = r * np.sin(w_rate * (t ** 2) + phi)
        p[:, 2] = height
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
        return [p, v, a, j, s]

    def runMPC(self, p, v, q, br, u, error_pose, error_vel):
        self.setReference(p=p, v=v, q=q, br=br, u=u)
        time = rospy.Time.now().to_sec()
        self.quadrotorOptimizer[self.experiment_times_current].acados_solver.solve()
        self.lastMPCTime = rospy.Time.now().to_sec()
        self.sendCmd()
        self.recordData(p=p, v=v, q=q, br=br, u=u, error_pose=error_pose, error_vel=error_vel, time=time)
        self.visTraj()

    def setReference(self, p, v, q, br, u):
        self.quadrotorOptimizer[self.experiment_times_current].acados_solver.set(0, "lbx", self.x0)
        self.quadrotorOptimizer[self.experiment_times_current].acados_solver.set(0, "ubx", self.x0)
        for i in range(self.N):
            yref = np.concatenate((p[i], v[i], q[i], br[i], u[i]))
            if self.quadrotorOptimizer[self.experiment_times_current].ocp.cost.cost_type == "LINEAR_LS":
                self.quadrotorOptimizer[self.experiment_times_current].acados_solver.set(i, 'yref', yref)
                # self.quadrotorOptimizer[self.experiment_times_current].acados_solver.set(i, 'u', u[i])
                # self.quadrotorOptimizer[self.experiment_times_current].acados_solver.set(self.N, 'yref', xref)
                # self.quadrotorOptimizer[self.experiment_times_current].acados_solver.set(i, 'xdot_guess', xdot_guessref)
            elif self.quadrotorOptimizer[self.experiment_times_current].ocp.cost.cost_type == "EXTERNAL":
                param = yref
                self.quadrotorOptimizer[self.experiment_times_current].acados_solver.set(i, 'p', param)

        xref = np.concatenate((p[self.N], v[self.N], q[self.N], br[self.N]))
        if self.quadrotorOptimizer[self.experiment_times_current].ocp.cost.cost_type_e == "LINEAR_LS":
            self.quadrotorOptimizer[self.experiment_times_current].acados_solver.set(self.N, 'yref', xref)
        elif self.quadrotorOptimizer[self.experiment_times_current].ocp.cost.cost_type_e == "EXTERNAL":
            param = np.concatenate((xref, np.zeros(self.quadrotorOptimizer[self.experiment_times_current].quadrotorModel.model.u.size()[0])))
            self.quadrotorOptimizer[self.experiment_times_current].acados_solver.set(self.N, 'p', param)

    def sendCmd(self):
        x1 = self.quadrotorOptimizer[self.experiment_times_current].acados_solver.get(1, "x")
        u1 = self.quadrotorOptimizer[self.experiment_times_current].acados_solver.get(0, "u")
        # p  = x1[: 3]
        # v  = x1[3: 6]
        q  = x1[6:10]
        br = x1[-3:]

        cmd = ControlCommand()
        cmd.header.stamp = rospy.Time.now()
        cmd.expected_execution_time = rospy.Time.now()
        cmd.control_mode = 2 # NONE=0 ATTITUDE=1 BODY_RATES=2 ANGULAR_ACCELERATIONS=3 ROTOR_THRUSTS=4
        cmd.armed = True
        cmd.orientation.w, cmd.orientation.x, cmd.orientation.y, cmd.orientation.z = q[0], q[1], q[2], q[3]
        cmd.bodyrates.x, cmd.bodyrates.y, cmd.bodyrates.z = br[0], br[1], br[2]
        cmd.collective_thrust = np.sum(u1 ** 2 * self.quadrotorOptimizer[self.experiment_times_current].quadrotorModel.kT) / self.quadrotorOptimizer[self.experiment_times_current].quadrotorModel.mass
        cmd.rotor_thrusts = u1 ** 2 * self.quadrotorOptimizer[self.experiment_times_current].quadrotorModel.kT
        self.ap_control_cmd_pub.publish(cmd)

    def recordData(self, p, v, q, br, u, error_pose, error_vel, time):
        v_b = v_dot_q(v[0], quaternion_inverse(np.array(self.q)))
        vb_abs = np.linalg.norm(v_dot_q(self.v_w, quaternion_inverse(np.array(self.q))))
        vw_abs = np.linalg.norm(self.v_w)
        aw_abs = np.linalg.norm(self.a_w)
        desire = Odometry()
        desire.pose.pose.position.x, desire.pose.pose.position.y, desire.pose.pose.position.z = p[0, 0], p[0, 1], p[0, 2]
        desire.pose.pose.orientation.w, desire.pose.pose.orientation.x, desire.pose.pose.orientation.y, desire.pose.pose.orientation.z = q[0, 0], q[0, 1], q[0, 2], q[0, 3]
        desire.twist.twist.linear.x, desire.twist.twist.linear.y, desire.twist.twist.linear.z = v_b[0], v_b[1], v_b[2]
        desire.twist.twist.angular.x, desire.twist.twist.angular.x, desire.twist.twist.angular.x = br[0, 0], br[0, 1], br[0, 2]
        self.desire_pub.publish(desire)

        max_motor_speed_now = np.max(self.motor_speed) if self.have_motor_speed else 0 
        self.max_motor_speed[self.experiment_times_current] = max(self.max_motor_speed[self.experiment_times_current], max_motor_speed_now)
        self.max_v_w[self.experiment_times_current] = max(self.max_v_w[self.experiment_times_current], vw_abs)
        self.max_a_w[self.experiment_times_current] = max(self.max_a_w[self.experiment_times_current], aw_abs)

        desire_motor = Actuators()
        desire_motor.angular_velocities = u[0]
        if self.a_data is not(None):
            desire_motor.angles = np.array([error_pose, error_vel, max_motor_speed_now, self.max_motor_speed[self.experiment_times_current], vw_abs, self.a_data[-1, 0], self.a_data[-1, 1], self.a_data[-1, 2] - 9.81, self.v_w[0], self.v_w[1], self.v_w[2]])
        self.desire_motor_pub.publish(desire_motor)

        self.print_cnt += 1
        if self.print_cnt > 5:
            self.print_cnt = 0
            print("********", self.cal_type, "|", self.expriment, self.experiment_times_current + 1, ':', self.experiment_times, "********")
            print('rest of time: ', self.experiment_time - (rospy.Time.now().to_sec() - self.begin_time))
            print('runtime: ', self.lastMPCTime - time)
            # print("pose:        [{:.2f}, {:.2f}, {:.2f}]".format(self.p[0], self.p[1], self.p[2]))
            print("pose error:  [{:.2f}, {:.2f}, {:.2f}], norm = {:.2f}".format(p[0, 0] - self.p[0], p[0, 1] - self.p[1], p[0, 2] - self.p[2], error_pose))
            print("pose rmse:   [{:.3f}]".format(math.sqrt(self.pose_error_square[self.experiment_times_current] / self.pose_error_num)))
            print("vel:         [{:.2f}, {:.2f}, {:.2f}], norm = {:.2f}".format(self.v_w[0], self.v_w[1], self.v_w[2], vw_abs))
            print("acc:         [{:.2f}, {:.2f}, {:.2f}], norm = {:.2f}".format(self.a_w[0], self.a_w[1], self.a_w[2], aw_abs))
            print("max motor :  {:.2f}, {:.2f}%".format(max_motor_speed_now, max_motor_speed_now / self.quadrotorOptimizer[self.experiment_times_current].quadrotorModel.model.RotorSpeed_max * 100))
            print()

    def visTraj(self):
        traj = Marker()
        traj.header.frame_id = '/world'
        traj.header.stamp = rospy.Time.now()
        traj.ns = "trajectory"
        traj.id = 0
        traj.type = Marker.SPHERE_LIST
        traj.action = Marker.ADD
        
        traj.pose.orientation.w = 1
        traj.pose.orientation.x = 0
        traj.pose.orientation.y = 0
        traj.pose.orientation.z = 0
        traj.color.r = 0
        traj.color.g = 0
        traj.color.b = 1
        traj.color.a = 1
        traj.scale.x = 0.1
        traj.scale.y = 0.1
        traj.scale.z = 0.1
        for i in range(self.N + 1):
            temp  = self.quadrotorOptimizer[self.experiment_times_current].acados_solver.get(i, "x")
            point = Point()
            point.x = temp[0]
            point.y = temp[1]
            point.z = temp[2]
            traj.points.append(point)
        self.traj_vis_pub.publish(traj)

    def fitParam(self, x_data, a_data, motor_data, t_data):
        self.need_fit_param = False
        kT = self.quadrotorOptimizer[self.experiment_times_current].quadrotorModel.kT
        mass = self.quadrotorOptimizer[self.experiment_times_current].quadrotorModel.mass
        # g = self.quadrotorOptimizer[self.experiment_times_current].quadrotorModel.g[2]
        b_drag = np.zeros((3 * len(x_data), 1))
        A_drag = np.zeros((3 * len(x_data), 4))
        d_drag = np.zeros((3 * len(x_data), 1))
        C_drag = np.zeros((3 * len(x_data), 5))
        brDotset = np.gradient(x_data[:, 10:13], t_data, axis=0)
        Inertia = self.quadrotorOptimizer[self.experiment_times_current].quadrotorModel.Inertia
        for i in range(len(x_data)):
            # p = x_data[i, :3]
            v = x_data[i, 3:6]
            a = a_data[i]
            # a[2] += g
            q = x_data[i, 6:10]
            br = x_data[i, 10:13]
            brDot = brDotset[i, :]
            temp_input = self.quadrotorOptimizer[self.experiment_times_current].quadrotorModel.G.dot(motor_data[i] ** 2)
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

    def saveParam(self, drag_coefficient):
        if os.path.exists(self.yaml_file) == False:
            # open(self.yaml_file, 'w')
            file = open(self.yaml_file, 'w', encoding='utf-8')
            config = {
                'D_dx': drag_coefficient[0],
                'D_dy': drag_coefficient[1],
                'D_dz': drag_coefficient[2],
                'kh': drag_coefficient[3],
                'D_ax': drag_coefficient[4],
                'D_ay': drag_coefficient[5],
                'D_bx': drag_coefficient[6],
                'D_by': drag_coefficient[7],
                'D_bz': drag_coefficient[8],
                }
            yaml.dump(config, file)
            file.close()
        else:
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

    def shutdown_node(self):
        print("closed")

def main():
    QuadMPC()
    
if __name__ == "__main__":
    main()