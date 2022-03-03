#!/usr/bin/env python3.6
import rospy, std_msgs.msg
import numpy as np
from std_msgs.msg import Int16, Bool
from nav_msgs.msg import Odometry
from mav_msgs.msg import Actuators
from quadrotor_msgs.msg import ControlCommand, AutopilotFeedback
from geometry_msgs.msg import PoseStamped, TwistStamped
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
    def __init__(self, t_horizon = 1, N = 10, plot = False) -> None:
        rospy.init_node("mpc_node")
        quad_name = rospy.get_param('~quad_name', default='hummingbird')
        self.expriment = rospy.get_param('~expriment', default='hover')
        self.start_point = np.zeros(3)
        self.start_point[0] = rospy.get_param('~hover_x', default='0')
        self.start_point[1] = rospy.get_param('~hover_y', default='0')
        self.start_point[2] = rospy.get_param('~hover_z', default='5')
        self.t_horizon = t_horizon # prediction horizon
        self.N = N # number of discretization steps
        self.have_odom = False
        self.have_ap_fb = False
        self.trigger = False
        self.reach_start_point = False
        # load model
        self.quadrotorOptimizer = QuadrotorOptimizer(self.t_horizon, self.N)
        # Subscribers
        self.odom_sub = rospy.Subscriber('/' + quad_name + '/ground_truth/odometry', Odometry, self.odom_callback)
        self.trigger_sub = rospy.Subscriber('/' + quad_name + '/trigger', std_msgs.msg.Empty, self.trigger_callback)
        self.ap_fb_sub = rospy.Subscriber('/' + quad_name + '/autopilot/feedback', AutopilotFeedback, self.ap_fb_callback)
        # Publishers
        self.arm_pub = rospy.Publisher('/' + quad_name + '/bridge/arm', Bool, queue_size=1, tcp_nodelay=True)
        self.start_autopilot_pub = rospy.Publisher('/' + quad_name + '/autopilot/start', std_msgs.msg.Empty, queue_size=1, tcp_nodelay=True)
        # self.control_motor_pub = rospy.Publisher('/' + quad_name + '/command/motor_speed', Actuators, queue_size=1, tcp_nodelay=True)
        self.desire_pub = rospy.Publisher('/' + quad_name + '/desire', Odometry, queue_size=1, tcp_nodelay=True)
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
        self.reach_last = False
        self.pose_error = 0
        self.pose_error_max = 0
        self.pose_error_num = 0

        if plot:
            self.getReference(experiment=self.expriment, start_point=self.start_point, time_now=0, t_horizon=70, N_node=500, model=self.quadrotorOptimizer.quadrotorModel, plot=True)

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

        while not self.have_ap_fb:
            rospy.sleep(0.5)
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

    def QuadMPCFSM(self, event):
        if not(self.have_odom) or not(self.trigger):
            return

        if not self.reach_start_point:
            # get to initial point
            p, v, q, br, u = self.getReference(experiment='hover', start_point=self.start_point, time_now=rospy.Time.now().to_sec() - self.begin_time, t_horizon=self.t_horizon, N_node=self.N, model=self.quadrotorOptimizer.quadrotorModel, plot=False)
            if (np.linalg.norm(np.array(self.p) - self.start_point) < 0.05):
                if not self.reach_last:
                    self.reach_last = True
                else:
                    self.reach_start_point = True
                    self.begin_time = rospy.Time.now().to_sec()
            else:
                self.reach_last = False
        elif (rospy.Time.now().to_sec() - self.begin_time <= 30):
            p, v, q, br, u = self.getReference(experiment=self.expriment,start_point=self.start_point, time_now=rospy.Time.now().to_sec() - self.begin_time, t_horizon=self.t_horizon, N_node=self.N, model=self.quadrotorOptimizer.quadrotorModel, plot=False)
            error = np.linalg.norm(np.array(self.p) - p[0])
            self.pose_error += error
            self.pose_error_num += 1
            if error > self.pose_error_max:
                self.pose_error_max = error
        else:
            self.trigger = False
            self.pose_error_mean = self.pose_error / self.pose_error_num
            self.pose_error = 0
            self.pose_error_num = 0
            print("error max : ", self.pose_error_max)
            print("error mean: ", self.pose_error_mean)
            return

        desire = Odometry()
        desire.pose.pose.position.x, desire.pose.pose.position.y, desire.pose.pose.position.z = p[0, 0], p[0, 1], p[0, 2]
        desire.pose.pose.orientation.w, desire.pose.pose.orientation.x, desire.pose.pose.orientation.y, desire.pose.pose.orientation.z = u[0, 0], u[0, 1], u[0, 2], u[0, 3]
        desire.twist.twist.linear.x, desire.twist.twist.linear.y, desire.twist.twist.linear.z = v[0, 0], v[0, 1], v[0, 2]

        self.desire_pub.publish(desire)

        self.quadrotorOptimizer.acados_solver.set(0, "lbx", self.x0)
        self.quadrotorOptimizer.acados_solver.set(0, "ubx", self.x0)
        for i in range(self.N):
            xref = np.concatenate((p[i], v[i], q[i], br[i], ))
            # self.quadrotorOptimizer.acados_solver.set(i, 'x', xref)
            self.quadrotorOptimizer.acados_solver.set(i, 'yref', np.concatenate((xref, u[i])))
            self.quadrotorOptimizer.acados_solver.set(i, 'u', u[i])
        xref = np.concatenate((p[self.N], v[self.N], q[self.N], br[self.N]))
        # self.quadrotorOptimizer.acados_solver.set(self.N, 'x', xref)
        self.quadrotorOptimizer.acados_solver.set(self.N, 'yref', xref)

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
        angle = rotation_matrix_to_euler(quat_to_rotation_matrix(q))

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
            start_point = np.array([0, 0, 3])
        p = np.zeros((N_node + 1, 3))
        v = np.zeros((N_node + 1, 3))
        a = np.zeros((N_node + 1, 3))
        j = np.zeros((N_node + 1, 3))
        s = np.zeros((N_node + 1, 3))
        yaw = np.zeros((N_node + 1, 1))
        yawdot = np.zeros((N_node + 1, 1))
        yawdotdot = np.zeros((N_node + 1, 1))
        delta_t = np.linspace(0, t_horizon, N_node + 1)
        t = time_now + delta_t
        # print(t)
        if experiment == 'hover':
            p[:, 0] = start_point[0]
            p[:, 1] = start_point[1]
            p[:, 2] = start_point[2]
            # u = math.sqrt(self.quadrotorOptimizer.quadrotorModel.g[-1] * self.quadrotorOptimizer.quadrotorModel.mass / self.quadrotorOptimizer.quadrotorModel.kT / 4)
            # u = np.ones((self.N, 4)) * u

        elif experiment == 'circle':
            r = 5
            w = 0.5 # rad/s
            phi = 0
            p[:, 0] = r * np.cos(w * t + phi) - r
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
            r = 5
            w_rate = 0.03
            w = t * w_rate # rad/s
            phi = 0
            p[:, 0] = r * np.cos(w * t + phi) - r
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

        q, euler_angle, br, u = getReference_Quaternion_Bodyrates_RotorSpeed(v=v, a=a, j=j, s=s, yaw=yaw, yawdot=yawdot, yawdotdot=yawdotdot, model=model)

        if plot:
            fig=plt.figure(num=1)# ,figsize=(9,9)

            ax1=fig.add_subplot(331)
            ax1.set_title("pose")
            ax1.plot(p[:, 0],p[:, 1], label='pose')

            ax2=fig.add_subplot(332)
            ax2.set_title("velocity")
            ax2.plot(t,v[:,0], label='x')
            ax2.plot(t,v[:,1], label='y')
            ax2.plot(t,v[:,2], label='z')

            ax3=fig.add_subplot(333)
            ax3.set_title("euler angle")
            ax3.plot(t,euler_angle[:, 0], label='x')
            ax3.plot(t,euler_angle[:, 1], label='y')
            ax3.plot(t,euler_angle[:, 2], label='z')

            ax4=fig.add_subplot(334)
            ax4.set_title("bodyrate")
            ax4.plot(t,br[:,0], label='z')
            ax4.plot(t,br[:,1], label='y')
            ax4.plot(t,br[:,2], label='z')
            
            ax5=fig.add_subplot(335)
            ax5.plot(t,u[:, 1])
            ax6=fig.add_subplot(336)
            ax6.plot(t,u[:, 2])
            ax7=fig.add_subplot(337)
            ax7.plot(t,u[:, 3])
            plt.show()

        return p, v, q, br, u

    def shutdown_node(self):
        print("closed")

def main():
    QuadMPC()
    
if __name__ == "__main__":
    main()