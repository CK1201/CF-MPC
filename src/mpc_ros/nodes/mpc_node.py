#!/usr/bin/env python3.6
import rospy, std_msgs.msg, message_filters, os, yaml, math
import numpy as np
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry, Path
from mav_msgs.msg import Actuators
from sensor_msgs.msg import Imu
from quadrotor_msgs.msg import ControlCommand, AutopilotFeedback
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker
from std_srvs.srv import Empty
from src.utils.utils import *
from src.quad_mpc.quad_optimizer import QuadrotorOptimizer
from src.quad_mpc.quad_model import QuadrotorModel
from decomp_ros_msgs.msg import PolyhedronArray
from matplotlib.font_manager import FontProperties  # 导入FontPropertie
import matplotlib as mpl
import matplotlib.pyplot as plt

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
        self.drag_coeff_type = rospy.get_param('~drag_coeff_type', default='LFS')
        self.fit_file_num = str(rospy.get_param('~fit_file_num', default='1'))
        self.result_file_num = str(rospy.get_param('~result_file_num', default='1'))
        self.expriment = rospy.get_param('~expriment', default='hover')
        self.record_data_time = rospy.get_param('~record_data_time', default='35')
        self.experiment_times = rospy.get_param('~experiment_times', default='1')
        self.start_point = np.zeros(3)
        self.hover_point = np.zeros(3)
        self.start_point[0] = rospy.get_param('~start_x', default='0')
        self.start_point[1] = rospy.get_param('~start_y', default='0')
        self.start_point[2] = rospy.get_param('~start_z', default='1')
        self.hover_point[0] = rospy.get_param('~hover_x', default='0')
        self.hover_point[1] = rospy.get_param('~hover_y', default='0')
        self.hover_point[2] = rospy.get_param('~hover_z', default='2')
        self.w_rate = rospy.get_param('~w_rate', default='0.05')
        self.max_vel = rospy.get_param('~max_vel', default='0.1')
        self.heading = rospy.get_param('~heading', default='false')
        plot = rospy.get_param('~plot', default='true')
        self.cost_type = rospy.get_param('~cost_type', default='EXTERNAL')
        self.need_collision_free = rospy.get_param('~need_collision_free', default='false')
        self.use_prior = rospy.get_param('~use_prior', default='true')
        self.poly_offset = rospy.get_param('~poly_offset', default='0.0')
        self.useTwoPolyhedron = False
        self.selectStrategy = "one" # all one pred_one (inside)
        self.need_sim = False
        self.vel_traversal = False
        self.wrate_traversal = False
        
        self.r_circle = 5 if self.start_point[0] < 1 else self.start_point[0]

        if self.vel_traversal:
            self.wrate_traversal = False
            self.expriment = 'circle_speedup_stay'
            # self.vel_list = np.arange(1, self.max_vel + 1)
            self.vel_list = np.arange(2, self.max_vel + 0.0001, 2)
            self.experiment_times = len(self.vel_list)
            self.wrate_list = np.ones(self.experiment_times) * self.w_rate
            self.stable_time = self.vel_list[0] / (2 * self.wrate_list[0] * self.r_circle) + 5
            self.experiment_time = self.record_data_time + self.stable_time
        elif self.wrate_traversal:
            self.expriment = 'circle_speedup'
            self.wrate_list = np.arange(0.05, self.w_rate + 0.0001, 0.05)
            self.experiment_times = len(self.wrate_list)
            self.vel_list = np.ones(self.experiment_times) * self.max_vel
            self.stable_time = 0
            self.experiment_time = self.vel_list[0] / (2 * self.wrate_list[0] * self.r_circle)
        else:
            self.vel_list = np.ones(self.experiment_times) * self.max_vel
            self.wrate_list = np.ones(self.experiment_times) * self.w_rate
            if self.expriment == 'circle_speedup_stay':
                self.stable_time = self.max_vel / (2 * self.w_rate * self.r_circle) + 5
                self.experiment_time = self.record_data_time + self.stable_time
            elif self.expriment == 'circle_speedup':
                self.stable_time = 0
                self.experiment_time = self.max_vel / (2 * self.w_rate * self.r_circle)
                # self.experiment_time = 2 / self.w_rate
            else:
                self.stable_time = 0
                self.experiment_time = self.record_data_time

        if not(self.with_drag):
            self.fit_file_num = '0'
        if self.cost_type == "LINEAR_LS" and self.need_collision_free:
            rospy.logwarn("cost type: LINEAR_LS can not be collision free!!! set cost type as EXTERNAL.")
            rospy.logwarn("cost type: LINEAR_LS can not be collision free!!! set cost type as EXTERNAL.")
            rospy.logwarn("cost type: LINEAR_LS can not be collision free!!! set cost type as EXTERNAL.")
            self.cost_type = "EXTERNAL"
        self.t_horizon = t_horizon # prediction horizon
        self.N = N # number of discretization steps'
        self.pose_error_num = 0
        self.print_cnt = 0
        self.experiment_times_current = 0
        self.pose_error = np.zeros(self.experiment_times)
        self.pose_error_square = np.zeros(self.experiment_times)
        self.pose_error_max = np.zeros(self.experiment_times)
        self.pose_error_mean = np.zeros(self.experiment_times)
        self.pose_error_rmse = np.zeros(self.experiment_times)
        self.yaw_error_square = np.zeros(self.experiment_times)
        self.yaw_rmse = np.zeros(self.experiment_times)
        self.max_v_w = np.zeros(self.experiment_times)
        self.max_a_w = np.zeros(self.experiment_times)
        self.max_motor_speed = np.zeros(self.experiment_times)
        self.crash_times = np.zeros(self.experiment_times)
        # self.crash_times = 0
        self.reach_times = 0
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
        self.last_poly_time = rospy.Time.now().to_sec()
        font = FontProperties(fname="SimHei.ttf", size=14)  # 设置字体
        mpl.rcParams['font.family'] = 'SimHei'
        plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）


        if not(self.need_collision_free):
            self.have_polyhedron = True
        # if self.expriment != "hover":
        #     self.have_polyhedron = True
        #     # rospy.logwarn("expriment: "+ self.expriment +" can not be Collision Free!!! set Collision Free as False.")
        #     # rospy.logwarn("expriment: "+ self.expriment +" can not be Collision Free!!! set Collision Free as False.")
        #     # rospy.logwarn("expriment: "+ self.expriment +" can not be Collision Free!!! set Collision Free as False.")
        #     self.need_collision_free = False
        # elif not(self.need_collision_free):
        #     self.have_polyhedron = True
        
        # load Drag coefficient
        dir_path = os.path.dirname(os.path.realpath(__file__))
        config_dir = dir_path + '/../config'
        result_dir = dir_path + '/../result'
        # mesh_dir = dir_path + '/../mesh' self.result_file_num
        # exp_name = self.expriment
        if self.with_drag:
            drag_name = self.drag_coeff_type + '_' + self.fit_file_num
        else:
            drag_name = 'NONE_' + self.fit_file_num
        self.result_file = os.path.join(result_dir, self.expriment + '_' + self.result_file_num + '_' + drag_name + '.txt')
        self.yaml_file = os.path.join(config_dir, quad_name + '_' + drag_name + '.yaml')
        # self.mesh_file = os.path.join("file://", mesh_dir, quad_name + '.mesh')
        self.mesh_file = "file:///home/ck1201/workspace/MAS/Traj_Tracking_MPC/src/mpc_ros/mesh/hummingbird.mesh"
        if not(self.with_drag):
            Drag_D = np.zeros((3, 3))
            Drag_kh = 0
            Drag_A = np.zeros((3, 3))
            Drag_B = np.zeros((3, 3))
        else:
            try:
                with open(self.yaml_file) as file:
                    config = yaml.load(file)
                    Drag_D = np.diag([config["D_dx"], config["D_dy"], config["D_dz"]])
                    Drag_kh = config["kh"]
                    Drag_A = np.zeros((3, 3))
                    Drag_A[0, 1] = config["D_ax"]
                    Drag_A[1, 0] = config["D_ay"]
                    Drag_B = np.diag([config["D_bx"], config["D_by"], config["D_bz"]])
                    # self.result_file = os.path.join(result_dir, self.expriment + '_with_drag.txt')
            except FileNotFoundError:
                warn_msg = "Tried to load Drag coefficient but the file was not found. Using default Drag coefficient."
                rospy.logwarn(warn_msg)
                Drag_D = np.zeros((3, 3))
                Drag_kh = 0
                Drag_A = np.zeros((3, 3))
                Drag_B = np.zeros((3, 3))
        # Drag_A = np.zeros((3, 3))
        # Drag_B = np.zeros((3, 3))
        print(Drag_D)
        print(Drag_kh)
        print(Drag_A)
        print(Drag_B)
        # load model
        self.quadrotorOptimizer = QuadrotorOptimizer(self.t_horizon, self.N, QuadrotorModel(Drag_D=Drag_D, Drag_kh=Drag_kh, Drag_A=Drag_A, Drag_B=Drag_B, need_collision_free=self.need_collision_free, useTwoPolyhedron=self.useTwoPolyhedron), cost_type=self.cost_type)

        # Subscribers
        if self.need_collision_free:
            self.PolyhedronArray_sub = rospy.Subscriber('/flight_corridor_node/polyhedron_array', PolyhedronArray, self.PolyhedronArray_callback)
            self.path_sub_ = rospy.Subscriber('/flight_corridor_node/path', Path, self.path_callback)
            self.polyhedron_array_in_use_pub = rospy.Publisher('/' + quad_name + '/polyhedron_array_in_use_pub', PolyhedronArray, queue_size=1, tcp_nodelay=True)
            self.path_in_use_pub = rospy.Publisher('/' + quad_name + '/path_in_use_pub', Path, queue_size=1, tcp_nodelay=True)
        self.quad_vis_pub = rospy.Publisher('/' + quad_name + '/quad_odom', Marker, queue_size=1, tcp_nodelay=True)
        self.quad_box_vis_pub = rospy.Publisher('/' + quad_name + '/quad_box', Marker, queue_size=1, tcp_nodelay=True)
        self.odom_sub = rospy.Subscriber('/' + quad_name + '/ground_truth/odometry', Odometry, self.odom_callback)
        self.imu_sub = rospy.Subscriber('/' + quad_name + '/ground_truth/imu', Imu, self.imu_callback)
        self.motor_speed_sub = rospy.Subscriber('/' + quad_name + '/motor_speed', Actuators, self.motor_speed_callback)
        self.trigger_sub = rospy.Subscriber('/' + quad_name + '/trigger', std_msgs.msg.Empty, self.trigger_callback)
        self.ap_fb_sub = rospy.Subscriber('/' + quad_name + '/autopilot/feedback', AutopilotFeedback, self.ap_fb_callback)
        
        # message filter
        if self.need_sim:
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
        self.traj_vis_pub = rospy.Publisher('/' + quad_name + '/quad_traj', Marker, queue_size=1, tcp_nodelay=True)
        self.mpc_ref_vis_pub = rospy.Publisher('/' + quad_name + '/mpc_ref', Marker, queue_size=1, tcp_nodelay=True)
        # self.call_for_poly_pub = rospy.Publisher('/' + quad_name + '/call_for_poly', Bool, queue_size=1, tcp_nodelay=True)
        
        # Trying to unpause Gazebo for 10 seconds.
        rospy.wait_for_service('/gazebo/unpause_physics')
        unpause_gazebo = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        unpaused = unpause_gazebo.call()

        if plot:
            self.getReference(experiment=self.expriment, start_point=self.start_point, hover_point=self.hover_point, time_now=0.1, t_horizon=35, N_node=350, velocity=10, model=self.quadrotorOptimizer.quadrotorModel, plot=True)
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
                rospy.sleep(1.0)
        print("HOVER STATE")
        
        # Timer
        self.timer = rospy.Timer(rospy.Duration(1 / 50), self.QuadMPCFSM)

        rate = rospy.Rate(N / t_horizon)
        while not rospy.is_shutdown():
            if self.finish_tracking:
                # if self.need_collision_free:
                #     # may have bug (long time no use)
                #     x_sim = self.Simulation(self.x_data, self.motor_data, self.t_data - self.begin_time)
                #     draw_data_sim(self.x_data, x_sim, self.motor_data, self.t_data - self.begin_time)
                rospy.sleep(3.0)
                if self.pose_error_max[self.experiment_times_current] > 2:
                    self.crash_times[self.experiment_times_current] += 1

                    self.pose_error[self.experiment_times_current] = 0
                    self.pose_error_max[self.experiment_times_current] = 0
                    self.pose_error_mean[self.experiment_times_current] = 0
                    self.pose_error_square[self.experiment_times_current] = 0
                    self.pose_error_rmse[self.experiment_times_current] = 0

                    self.yaw_error_square[self.experiment_times_current] = 0
                    self.yaw_rmse[self.experiment_times_current] = 0

                    self.max_v_w[self.experiment_times_current] = 0
                    self.max_a_w[self.experiment_times_current] = 0
                    self.max_motor_speed[self.experiment_times_current] = 0
                    self.experiment_times_current -= 1
                self.experiment_times_current += 1
                self.finish_tracking = False

                if (self.experiment_times_current < self.experiment_times):
                    self.trigger = True
                    self.begin_time = rospy.Time.now().to_sec()
                    if self.expriment != "hover":
                        self.stable_time = 0
                        self.experiment_time = self.vel_list[self.experiment_times_current] / (2 * self.wrate_list[self.experiment_times_current] * self.r_circle)
                        if not(self.wrate_traversal):
                            self.stable_time = self.experiment_time + 5
                            self.experiment_time = self.record_data_time + self.stable_time
                        
                    
                    # np.savetxt(self.result_file, np.concatenate((self.pose_error_max[np.newaxis,:], self.pose_error_rmse[np.newaxis,:], self.yaw_rmse[np.newaxis,:], self.max_v_w[np.newaxis,:], self.max_a_w[np.newaxis,:], self.max_motor_speed[np.newaxis,:], np.ones((1,self.experiment_times)) * self.crash_times), axis=0), fmt='%0.8f', delimiter=',')
                else:
                    print("max error : ", self.pose_error_max)
                    print("mean error: ", self.pose_error_mean)
                    print("pose rmse : ", self.pose_error_rmse)
                    print("yaw  rmse : ", self.yaw_rmse)
                    print("max v_w   : ", self.max_v_w)
                    print("max a_w   : ", self.max_a_w)
                    print("max motor : ", self.max_motor_speed)

                    
                    # np.savetxt(self.result_file, np.concatenate((self.pose_error_max[np.newaxis,:], self.pose_error_rmse[np.newaxis,:], self.yaw_rmse[np.newaxis,:], self.max_v_w[np.newaxis,:], self.max_a_w[np.newaxis,:], self.max_motor_speed[np.newaxis,:], np.ones((1,self.experiment_times)) * self.crash_times), axis=0), fmt='%0.8f', delimiter=',')
                    
                    # if self.vel_traversal:
                    #     print()
                    # else:
                    #     fig=plt.figure()
                    #     ax1=fig.add_subplot(2,2,1)
                    #     # ax1.set_title("pose_error_max")
                    #     # ax1.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
                    #     # ax1.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
                    #     ax1.boxplot([self.pose_error_rmse], labels=['Pose RMSE'], showmeans=True)
                    #     ax1.set_ylabel("m")
                    #     ax1.grid()

                    #     ax2=fig.add_subplot(2,2,2)
                    #     # ax2.set_title("pose_error_mean")
                    #     # ax2.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
                    #     # ax2.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
                    #     ax2.boxplot([self.yaw_rmse], labels=['Yaw RMSE'], showmeans=True)
                    #     ax2.set_ylabel("m")
                    #     ax2.grid()

                    #     ax3=fig.add_subplot(2,2,3)
                    #     # ax3.set_title("max_motor_speed_percent")
                    #     # ax3.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
                    #     # ax3.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
                    #     ax3.boxplot([self.max_v_w], labels=['Max Velocity'], showmeans=True)
                    #     ax3.set_ylabel("m/s")
                    #     ax3.grid()

                    #     ax4=fig.add_subplot(2,2,4)
                    #     # ax4.set_title("max_v_w")
                    #     # ax4.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
                    #     # ax4.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
                    #     ax4.boxplot([self.max_motor_speed / self.quadrotorOptimizer.quadrotorModel.model.RotorSpeed_max], labels=['Max Rotor Speed'], showmeans=True)
                    #     ax4.set_ylabel("deg°")
                    #     ax4.grid()

                    #     plt.show()
                np.savetxt(self.result_file, np.concatenate((self.pose_error_max[np.newaxis,:], self.pose_error_rmse[np.newaxis,:], self.yaw_rmse[np.newaxis,:], self.max_v_w[np.newaxis,:], self.max_a_w[np.newaxis,:], self.max_motor_speed[np.newaxis,:], self.crash_times[np.newaxis,:]), axis=0), fmt='%0.8f', delimiter=',')
                # return
            rate.sleep()

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

        boxVertexNp = self.quadrotorOptimizer.quadrotorModel.model.boxVertexNp
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
        self.a_w[2] -= 9.81

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
        self.yaw_error_square = np.zeros(self.experiment_times)
        self.yaw_rmse = np.zeros(self.experiment_times)
        self.max_v_w = np.zeros(self.experiment_times)
        self.max_a_w = np.zeros(self.experiment_times)
        self.max_motor_speed = np.zeros(self.experiment_times)
        self.reach_times = 0

    def ap_fb_callback(self, msg):
        self.have_ap_fb = True
        self.ap_state =  msg.autopilot_state

    def PolyhedronArray_callback(self, msg):
        if not(self.have_odom):
            return
        if not(self.use_prior):
        # if rospy.Time.now().to_sec() - self.last_poly_time > 2:
            self.max_select = 0
        self.have_polyhedron = True
        self.PolyhedronArray = msg

    def path_callback(self, msg):
        self.Path = msg

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
        if not(self.have_odom) or not(self.trigger) or not(self.have_polyhedron):
            return

        if not self.reach_start_point:
            # get to initial point
            if not self.have_reach_start_point_cmd:
                print("go to start point")
                cmd = PoseStamped()
                cmd.header.stamp = rospy.Time.now()
                cmd.pose.position.x, cmd.pose.position.y, cmd.pose.position.z = self.start_point[0], self.start_point[1], self.start_point[2]
                if self.expriment != "hover":
                    theta = math.pi / 2
                    cmd.pose.orientation.w, cmd.pose.orientation.x, cmd.pose.orientation.y, cmd.pose.orientation.z = math.cos(theta / 2), 0, 0, math.sin(theta / 2)
                self.control_pose_pub.publish(cmd)
                rospy.sleep(1.0)
                if self.ap_state == AutopilotFeedback().TRAJECTORY_CONTROL or (np.linalg.norm(np.array(self.p) - self.start_point) < 0.1):
                    self.have_reach_start_point_cmd = True
                else:
                    return
            if (self.ap_state == AutopilotFeedback().HOVER) and (np.linalg.norm(np.array(self.p) - self.start_point) < 0.1):
                self.reach_times += 1
                if self.reach_times > 50:
                    self.reach_times = 0
                    self.reach_start_point = True
                    self.begin_time = rospy.Time.now().to_sec()
                    self.have_reach_start_point_cmd = False
                    if self.need_collision_free:
                        self.polyhedrons = self.PolyhedronArray.polyhedrons
                        self.polyhedron_array_in_use_pub.publish(self.PolyhedronArray)
                        self.path_in_use_pub.publish(self.Path)
                        self.max_select = 0
                    if self.need_sim:
                        self.start_record = True
                        self.start_record_time = rospy.Time.now().to_sec()
                    print("arrive start point")
            else:
                if self.ap_state != AutopilotFeedback().TRAJECTORY_CONTROL:
                    cmd = PoseStamped()
                    cmd.header.stamp = rospy.Time.now()
                    cmd.pose.position.x, cmd.pose.position.y, cmd.pose.position.z = self.start_point[0], self.start_point[1], self.start_point[2]
                    if self.expriment != "hover":
                        theta = math.pi / 2
                        cmd.pose.orientation.w, cmd.pose.orientation.x, cmd.pose.orientation.y, cmd.pose.orientation.z = math.cos(theta / 2), 0, 0, math.sin(theta / 2)
                    self.control_pose_pub.publish(cmd)
                    rospy.sleep(1.0)
            return
        elif (not self.finish_tracking):
            p, v, q, br, u = self.getReference(experiment=self.expriment, start_point=self.start_point, hover_point=self.hover_point, time_now=rospy.Time.now().to_sec() - self.begin_time, t_horizon=self.t_horizon, N_node=self.N, velocity=self.vel_list[self.experiment_times_current], model=self.quadrotorOptimizer.quadrotorModel, plot=False)
            error_pose = np.linalg.norm(np.array(self.p) - p[0])
            error_vel  = np.linalg.norm(np.array(self.v_w) - v[0])
            euler_now = quaternion_to_euler(np.array(self.q)) / math.pi * 180
            euler_ref = quaternion_to_euler(q[0]) / math.pi * 180
            phi = round(abs(euler_now[2] - euler_ref[2])) % 360
            self.error_yaw = 360 - phi if phi > 180 else phi
            # print(error_yaw)
            error_q    = np.linalg.norm(diff_between_q_q(np.array(self.q), q[0]))
            error_br   = np.linalg.norm(np.array(self.w) - br[0])
            if rospy.Time.now().to_sec() - self.begin_time > self.stable_time:
                self.pose_error[self.experiment_times_current] += error_pose
                self.pose_error_square[self.experiment_times_current] += error_pose ** 2
                self.yaw_error_square[self.experiment_times_current] += self.error_yaw ** 2
                self.pose_error_num += 1
                self.pose_error_max[self.experiment_times_current] = max(self.pose_error_max[self.experiment_times_current], error_pose)
            # if self.expriment == 'hover':
            #     if (np.linalg.norm(np.array(self.p) - self.hover_point) < 0.05):
            #         self.reach_times += 1
            #         if self.reach_times > 500:
            #             self.finish_tracking = True
            #             self.reach_times = 0    
            if rospy.Time.now().to_sec() - self.begin_time >= self.experiment_time:
                self.finish_tracking = True
            elif (error_pose > 2.1) and self.expriment != 'hover':
                self.pose_error_max[self.experiment_times_current] = error_pose
                self.finish_tracking = True
        else:
            self.trigger = False
            self.reach_start_point = False
            self.pose_error_mean[self.experiment_times_current] = self.pose_error[self.experiment_times_current] / self.pose_error_num
            self.pose_error_rmse[self.experiment_times_current] = math.sqrt(self.pose_error_square[self.experiment_times_current] / self.pose_error_num)
            self.yaw_rmse[self.experiment_times_current] = math.sqrt(self.yaw_error_square[self.experiment_times_current] / self.pose_error_num)
            self.pose_error_num = 0
            print("max error : ", self.pose_error_max[self.experiment_times_current])
            # print("mean error: ", self.pose_error_mean[self.experiment_times_current])
            print("pose rmse : ", self.pose_error_rmse[self.experiment_times_current])
            print("yaw rmse  : ", self.yaw_rmse[self.experiment_times_current])
            print("max vel   : ", self.max_v_w[self.experiment_times_current])
            print("max acc   : ", self.max_a_w[self.experiment_times_current])
            print("max motor :  {:.2f}, {:.2f}%".format(self.max_motor_speed[self.experiment_times_current], self.max_motor_speed[self.experiment_times_current] / self.quadrotorOptimizer.quadrotorModel.model.RotorSpeed_max * 100))
            return
        
        self.runMPC(p=p, v=v, q=q, br=br, u=u, error_pose=error_pose, error_vel=error_vel) 
        return

    def getReference(self, experiment, start_point, hover_point, time_now, t_horizon, N_node, velocity, model, plot):
        if start_point is None:
            start_point = np.array([5, 0, 5])
        if abs(self.start_point[0]) < 1:
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
            dist = np.linalg.norm((hover_point - start_point))
            vel = (hover_point - start_point) / dist * velocity
            timeAll = dist / velocity
            temp_t = np.concatenate((t[np.newaxis,:] / timeAll, np.ones((1, N_node + 1))), axis=0)
            temp_t = temp_t.min(0)[:,np.newaxis]
            temp_t = np.repeat(temp_t, repeats=3, axis=1)
            temp_p = (hover_point - start_point)[np.newaxis,:]
            temp_p = np.repeat(temp_p, repeats=N_node + 1, axis=0)
            p = temp_t * temp_p + np.repeat(start_point[np.newaxis,:], repeats=N_node + 1, axis=0)

            for i in range(N_node + 1):
                if t[i] < timeAll:
                    v[i] = vel
                else:
                    v[i] = 0
            # p[:, 0] = temp[0]
            # p[:, 1] = temp[1]
            # p[:, 2] = temp[2]
            # p = np.repeat(hover_point[np.newaxis,:], repeats=N_node + 1, axis=0)

        elif experiment == 'circle':
            # w = 2 # rad/s
            w = velocity / r
            phi = 0
            [p, v, a, j, s] = self.getRefCircleHor(r, w, phi, t, start_point[2])

        elif experiment == 'circle_speedup_stay':
            w = velocity / r
            w_rate = self.wrate_list[self.experiment_times_current]
            phi = 0
            stable_time = self.stable_time - 5
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
            w_rate = self.wrate_list[self.experiment_times_current]
            phi = 0
            [p, v, a, j, s] = self.getRefCircleSpeedup(r, w_rate, phi, t, start_point[2])
            
        elif experiment == 'circle_vertical':
            w = velocity / r # rad/s
            phi = 0
            [p, v, a, j, s] = self.getRefCircleVert(r, w, phi, t, start_point[2])
        
        elif experiment == 'circle_vertical_speedup_stay':
            w = velocity / r
            w_rate = self.wrate_list[self.experiment_times_current]
            phi = 0
            stable_time = self.stable_time - 5
            stable_phi = w_rate * stable_time ** 2 + phi
            for i in range(N_node + 1):
                # if w_rate * t[i] ** 2 < w * t[i]:
                if t[i] < stable_time:
                    temp_t = np.ones((1,1)) * t[i]
                    [p[i], v[i], a[i], j[i], s[i]] = self.getRefCircleSpeedup(r, w_rate, phi, temp_t, start_point[2])
                else:
                    temp_t = np.ones((1,1)) * (t[i] - stable_time)
                    [p[i], v[i], a[i], j[i], s[i]] = self.getRefCircleVert(r, w, stable_phi, temp_t, start_point[2])

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
            if experiment == 'hover':
                yaw[:] = math.atan2(hover_point[1] - start_point[1], hover_point[0] - start_point[0])
            else:
                for i in range(len(p)):
                    yaw[i] = math.atan2(v[i, 1], v[i, 0])
        else:
            yaw[:] = math.pi * 0
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
            # yawdot[i] = (a[i, 1] * v[i, 0] - v[i, 1] * a[i, 0]) / (v[i, 0] ** 2 + v[i, 1] ** 2)
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

        point = Point()
        for i in range(len(p)):
            point.x, point.y, point.z = p[i,0], p[i,1], p[i,2]
            mpc_ref.points.append(point)
        self.mpc_ref_vis_pub.publish(mpc_ref)

        return p, v, q, br, u

    def getRefCircleHor(self, r, w, phi, t, height):
        """
        This function returns the position, velocity, acceleration, jerk, and snap of a reference circle
        in the horizontal plane
        
        :param r: radius of the circle
        :param w: angular velocity
        :param phi: phase angle
        :param t: time
        :param height: height of the circle
        :return: the position, velocity, acceleration, jerk, and snap of a reference circle.
        """
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
        """
        This function returns the position, velocity, acceleration, jerk, and snap of a reference circle
        with radius r, angular velocity w, phase phi, and height height
        
        :param r: radius of the circle
        :param w: angular velocity
        :param phi: the phase of the circle
        :param t: time
        :param height: the height of the circle
        :return: the position, velocity, acceleration, jerk, and snap of a reference circle.
        """
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
        """
        The function takes in a radius, angular velocity, phase, time, and height, and returns the
        position, velocity, acceleration, jerk, and snap of a reference circle
        
        :param r: radius of the circle
        :param w_rate: the angular velocity of the circle
        :param phi: the initial angle of the circle
        :param t: time
        :param height: height of the circle
        :return: a list of 5 elements, each element is a numpy array of size (len(t), 3).
        """
        p = np.zeros((len(t), 3))
        v = np.zeros((len(t), 3))
        a = np.zeros((len(t), 3))
        j = np.zeros((len(t), 3))
        s = np.zeros((len(t), 3))
        p[:, 0] = r * np.cos(w_rate * t ** 2 + phi)
        p[:, 1] = r * np.sin(w_rate * t ** 2 + phi)
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
        """
        The function takes in the current state of the quadrotor, the desired state, and the error
        between the current and desired state. It then sets the reference state to the desired state,
        solves the MPC problem, sends the command to the quadrotor, records the data, and visualizes the
        trajectory
        
        :param p: position
        :param v: velocity
        :param q: the current state of the quadrotor
        :param br: body rate
        :param u: the desired acceleration
        :param error_pose: the error between the current pose and the desired pose
        :param error_vel: velocity error
        """
        self.setReference(p=p, v=v, q=q, br=br, u=u)
        time = rospy.Time.now().to_sec()
        self.quadrotorOptimizer.acados_solver.solve()
        self.lastMPCTime = rospy.Time.now().to_sec()
        self.sendCmd()
        self.recordData(p=p, v=v, q=q, br=br, u=u, error_pose=error_pose, error_vel=error_vel, time=time)
        self.visTraj()

    def selectPolyhedron(self):
        """
        The function takes in the current position and orientation of the quadrotor, and the list of
        polyhedrons, and returns the index of the polyhedron that the quadrotor is currently in. 
        
        The function first checks if the quadrotor is in the first polyhedron. If it is, it returns 0. If
        not, it checks if the quadrotor is in the second polyhedron. If it is, it returns 1. If not, it
        checks if the quadrotor is in the third polyhedron. If it is, it returns 2. And so on. 
        
        The function stops checking if the quadrotor is in the next polyhedron if it finds that the
        quadrotor is not in the current polyhedron. 
        
        The function also takes in a parameter called `selectStrategy`. This parameter determines how the
        function checks if the quadrotor is in a polyhedron.
        :return: The index of the polyhedron that is selected.
        """
        boxVertexNp = self.quadrotorOptimizer.quadrotorModel.model.boxVertexNp
        RotationMat = quat_to_rotation_matrix(unit_quat(np.array(self.q)))
        pos = np.array(self.p)

        select = self.max_select
        if not(self.use_prior):
            # if rospy.Time.now().to_sec() - self.last_poly_time > 2:
                # self.max_select = 0
                self.last_poly_time = rospy.Time.now().to_sec()
                self.polyhedrons = self.PolyhedronArray.polyhedrons
                self.polyhedron_array_in_use_pub.publish(self.PolyhedronArray)
                self.path_in_use_pub.publish(self.Path)

        if self.selectStrategy == "all":
            # only all inside 
            for i in range(self.max_select, len(self.polyhedrons) - 1):
                inside = True
                for j in range(len(self.polyhedrons[i + 1].points)):
                    temp = self.polyhedrons[i + 1].points[j]
                    point = np.array([temp.x, temp.y, temp.z])
                    temp = self.polyhedrons[i + 1].normals[j]
                    normal = np.array([temp.x, temp.y, temp.z])
                    b = normal.dot(point)
                    for k in range(len(boxVertexNp)):
                        pot = boxVertexNp[k]
                        if normal.dot(RotationMat.dot(pot[:,np.newaxis]) + pos[:,np.newaxis]) - b > 0:
                            inside = False
                            break
                    if inside == False:
                        break
                if inside == False:
                    break
                else:
                    select = i + 1
                    # if self.use_prior:
                    self.max_select = select
        elif self.selectStrategy == "one":
            # as long as one inside
            for i in range(self.max_select, len(self.polyhedrons) - 1):
                for k in range(len(boxVertexNp)):
                    inside = True
                    pot = boxVertexNp[k]
                    pot = RotationMat.dot(pot[:,np.newaxis]) + pos[:,np.newaxis]
                    for j in range(len(self.polyhedrons[i + 1].points)):
                        temp = self.polyhedrons[i + 1].points[j]
                        point = np.array([temp.x, temp.y, temp.z])
                        temp = self.polyhedrons[i + 1].normals[j]
                        normal = np.array([temp.x, temp.y, temp.z])
                        b = normal.dot(point)
                        
                        if normal.dot(pot) - b > 0:
                            inside = False
                            break
                    if inside == True:
                        break
                if inside == False:
                    break
                else:
                    select = i + 1
                    # if self.use_prior:
                    self.max_select = select
        elif self.selectStrategy == "pred_one":
            posPred = np.array([self.quadrotorOptimizer.acados_solver.get(int(self.N * 0.3), "x")[:3]])
            for i in range(self.max_select, len(self.polyhedrons) - 1):
                inside = True
                for j in range(len(self.polyhedrons[i + 1].points)):
                    temp = self.polyhedrons[i + 1].points[j]
                    point = np.array([temp.x, temp.y, temp.z])
                    temp = self.polyhedrons[i + 1].normals[j]
                    normal = np.array([temp.x, temp.y, temp.z])
                    b = normal.dot(point)
                    if posPred.dot(normal[:,np.newaxis]) - b > 0:
                        inside = False
                        break
                if inside == False:
                    break
                else:
                    select = i + 1
                    # if self.use_prior:
                    self.max_select = select
        elif self.selectStrategy == "near":
            # to-do
            posPred = np.array([self.quadrotorOptimizer.acados_solver.get(0, "x")[:3]])
            for i in range(self.max_select, len(self.polyhedrons) - 1):
                inside = True
                for j in range(len(self.polyhedrons[i + 1].points)):
                    temp = self.polyhedrons[i + 1].points[j]
                    point = np.array([temp.x, temp.y, temp.z])
                    temp = self.polyhedrons[i + 1].normals[j]
                    normal = np.array([temp.x, temp.y, temp.z])
                    b = normal.dot(point)
                    if posPred.dot(normal[:,np.newaxis]) - b > 0:
                        inside = False
                        break
                if inside == False:
                    break
                else:
                    select = i + 1
                    # if self.use_prior:
                    self.max_select = select
        return select

    def setReference(self, p, v, q, br, u):
        """
        It sets the reference trajectory for the quadrotor
        
        :param p: position
        :param v: velocity
        :param q: position
        :param br: body rate
        :param u: control input
        """
        self.quadrotorOptimizer.acados_solver.set(0, "lbx", self.x0)
        self.quadrotorOptimizer.acados_solver.set(0, "ubx", self.x0)
        if self.need_collision_free:
            paramPolyhedron = self.getPolyhedronParam()
        for i in range(self.N):
            yref = np.concatenate((p[i], v[i], q[i], br[i], u[i]))
            if self.quadrotorOptimizer.ocp.cost.cost_type == "LINEAR_LS":
                self.quadrotorOptimizer.acados_solver.set(i, 'yref', yref)
                # self.quadrotorOptimizer.acados_solver.set(i, 'u', u[i])
                # self.quadrotorOptimizer.acados_solver.set(self.N, 'yref', xref)
                # self.quadrotorOptimizer.acados_solver.set(i, 'xdot_guess', xdot_guessref)
            elif self.quadrotorOptimizer.ocp.cost.cost_type == "EXTERNAL":
                param = yref
                if self.need_collision_free:
                    param = np.concatenate((param, paramPolyhedron))
                self.quadrotorOptimizer.acados_solver.set(i, 'p', param)

        xref = np.concatenate((p[self.N], v[self.N], q[self.N], br[self.N]))
        if self.quadrotorOptimizer.ocp.cost.cost_type_e == "LINEAR_LS":
            self.quadrotorOptimizer.acados_solver.set(self.N, 'yref', xref)
        elif self.quadrotorOptimizer.ocp.cost.cost_type_e == "EXTERNAL":
            param = np.concatenate((xref, np.zeros(self.quadrotorOptimizer.quadrotorModel.model.u.size()[0])))
            if self.need_collision_free:
                param = np.concatenate((param, paramPolyhedron))
            self.quadrotorOptimizer.acados_solver.set(self.N, 'p', param)
        
    def getPolyhedronParam(self):
        select_num = self.selectPolyhedron()
        selectedPolyhedron = self.polyhedrons[select_num]
        paramPolyhedron = np.array([])
        for i in range(self.quadrotorOptimizer.quadrotorModel.model.MaxNumOfPolyhedrons):
            if i < len(selectedPolyhedron.points):
                temp = selectedPolyhedron.points[i]
                point = np.array([temp.x, temp.y, temp.z])
                temp = selectedPolyhedron.normals[i]
                normal = np.array([temp.x, temp.y, temp.z])
                # b = normal.dot(point)
                b = normal.dot(point - normal / np.linalg.norm(normal) * self.poly_offset)
                paramPolyhedron = np.concatenate((paramPolyhedron, normal, np.array([b])))
            else:
                paramPolyhedron = np.concatenate((paramPolyhedron, np.array([0, 0, -1, 0])))

        if self.useTwoPolyhedron:
            selectedPolyhedron_next = self.polyhedrons[min(select_num + 1, len(self.polyhedrons) - 1)]
            for i in range(self.quadrotorOptimizer.quadrotorModel.model.MaxNumOfPolyhedrons):
                if i < len(selectedPolyhedron_next.points):
                    temp = selectedPolyhedron_next.points[i]
                    point = np.array([temp.x, temp.y, temp.z])
                    temp = selectedPolyhedron_next.normals[i]
                    normal = np.array([temp.x, temp.y, temp.z])
                    # b = normal.dot(point)
                    b = normal.dot(point - normal / np.linalg.norm(normal) * self.poly_offset)
                    paramPolyhedron = np.concatenate((paramPolyhedron, normal, np.array([b])))
                else:
                    paramPolyhedron = np.concatenate((paramPolyhedron, np.array([0, 0, -1, 0])))
        return paramPolyhedron

    def sendCmd(self):
        x1 = self.quadrotorOptimizer.acados_solver.get(1, "x")
        u1 = self.quadrotorOptimizer.acados_solver.get(0, "u")
        # p  = x1[: 3]
        # v  = x1[3: 6]
        q  = x1[6: 10]
        br = x1[-3:]

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
        if rospy.Time.now().to_sec() - self.begin_time > self.stable_time:
            self.max_motor_speed[self.experiment_times_current] = max(self.max_motor_speed[self.experiment_times_current], max_motor_speed_now)
            self.max_v_w[self.experiment_times_current] = max(self.max_v_w[self.experiment_times_current], vw_abs)
            self.max_a_w[self.experiment_times_current] = max(self.max_a_w[self.experiment_times_current], aw_abs)

        desire_motor = Actuators()
        desire_motor.angular_velocities = u[0]
        if self.a_data is not(None):
            desire_motor.angles = np.array([error_pose, error_vel, max_motor_speed_now, self.max_motor_speed[self.experiment_times_current], vw_abs, aw_abs, self.a_w[0], self.a_w[1], self.a_w[2], self.v_w[0], self.v_w[1], self.v_w[2]])
        self.desire_motor_pub.publish(desire_motor)

        self.print_cnt += 1
        if self.print_cnt > 5:
            self.print_cnt = 0
            print("********",self.expriment, self.experiment_times_current + 1, ':', self.experiment_times, "********")
            print('rest of time: ', self.experiment_time - (rospy.Time.now().to_sec() - self.begin_time))
            print('runtime: ', self.lastMPCTime - time)
            print("pose:        [{:.2f}, {:.2f}, {:.2f}]".format(self.p[0], self.p[1], self.p[2]))
            print("vel:         [{:.2f}, {:.2f}, {:.2f}], norm = {:.2f}".format(self.v_w[0], self.v_w[1], self.v_w[2], vw_abs))
            print("pose error:  [{:.2f}, {:.2f}, {:.2f}], norm = {:.2f}".format(p[0, 0] - self.p[0], p[0, 1] - self.p[1], p[0, 2] - self.p[2], error_pose))
            pose_error_num = 1 if self.pose_error_num == 0 else self.pose_error_num
            print("pose rmse:   [{:.3f}]".format(math.sqrt(self.pose_error_square[self.experiment_times_current] / pose_error_num)))
            print("yaw rmse:    [{:.3f}], {:.3f}".format(math.sqrt(self.yaw_error_square[self.experiment_times_current] / pose_error_num), self.error_yaw))
            print("max motor :  {:.2f}, {:.2f}%".format(max_motor_speed_now, max_motor_speed_now / self.quadrotorOptimizer.quadrotorModel.model.RotorSpeed_max * 100))
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
        for i in range(self.N):
            temp  = self.quadrotorOptimizer.acados_solver.get(i + 1, "x")
            point = Point()
            point.x = temp[0]
            point.y = temp[1]
            point.z = temp[2]
            traj.points.append(point)
        self.traj_vis_pub.publish(traj)

    def Simulation(self, x_data, motor_data, t):
        model = self.quadrotorOptimizer.quadrotorModel
        x_sim = np.zeros((len(motor_data), 13))
        x_sim[0] = x_data[0]
        for i in range(len(motor_data) - 1):
            x_sim[i + 1] = np.concatenate((model.Simulation(x_sim[i, :3], x_sim[i, 3:6], x_sim[i, 6:10], x_data[i, 10:], np.abs(motor_data[i]), t[i + 1] - t[i])))
        return x_sim

def main():
    QuadMPC()
    
if __name__ == "__main__":
    main()