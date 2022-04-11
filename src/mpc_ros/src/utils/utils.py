import math, pyquaternion
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import xml.etree.ElementTree as XMLtree

def flatten(Mat):
    if isinstance(Mat, np.ndarray):
        row, col = np.size(Mat, 0), np.size(Mat, 1)
        vector = np.zeros(row * col)
    else:
        [row, col] = Mat.size()
        vector = ca.SX.zeros(row * col)
    for i in range(row):
        for j in range(col):
            vector[i * col + j] = Mat[i, j]
    return vector

def crossmat(vec):
    if isinstance(vec, np.ndarray):
        return np.array([
            [0      , -vec[2], vec[1] ],
            [vec[2] , 0      , -vec[0]],
            [-vec[1], vec[0] , 0      ]])
    else:
        return ca.vertcat(
            ca.horzcat(0      , -vec[2], vec[1] ),
            ca.horzcat(vec[2] , 0      , -vec[0]),
            ca.horzcat(-vec[1], vec[0] , 0      ))

def v1_cross_v2(vec1, vec2):
    return np.dot(crossmat(vec1), vec2.T)

def unit_quat(q):
    if isinstance(q, np.ndarray):
        q_norm = np.sqrt(np.sum(q ** 2))
    else:
        q_norm = ca.sqrt(ca.sumsqr(q))
    return q / q_norm

def v_dot_q(v, q):
    rot_mat = quat_to_rotation_matrix(q)
    if isinstance(q, np.ndarray):
        return rot_mat.dot(v)

    return ca.mtimes(rot_mat, v)

def quaternion_inverse(q):
    # q = unit_quat(q)
    w, x, y, z = q[0], q[1], q[2], q[3]
    if isinstance(q, np.ndarray):
        return np.array([w, -x, -y, -z])
    else:
        return ca.vertcat(w, -x, -y, -z)

def q_dot_q(q, r):
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    rw, rx, ry, rz = r[0], r[1], r[2], r[3]

    t0 = rw * qw - rx * qx - ry * qy - rz * qz
    t1 = rw * qx + rx * qw - ry * qz + rz * qy
    t2 = rw * qy + rx * qz + ry * qw - rz * qx
    t3 = rw * qz - rx * qy + ry * qx + rz * qw

    if isinstance(q, np.ndarray):
        return np.array([t0, t1, t2, t3])
    else:
        return ca.vertcat(t0, t1, t2, t3)

def diff_between_q_q(q, r):
    return q_dot_q(q, quaternion_inverse(r))

def quat_to_rotation_matrix(quaternions):
    q0, q1, q2, q3 = quaternions[0], quaternions[1], quaternions[2], quaternions[3]
    if isinstance(quaternions, np.ndarray):
        return np.array([
            [1 - 2 * (q2 ** 2 + q3 ** 2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
            [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 ** 2 + q3 ** 2), 2 * (q2 * q3 - q0 * q1)],
            [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1 ** 2 + q2 ** 2)]])
    else:
        return ca.vertcat(
            ca.horzcat(1 - 2 * (q2 ** 2 + q3 ** 2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)),
            ca.horzcat(2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 ** 2 + q3 ** 2), 2 * (q2 * q3 - q0 * q1)),
            ca.horzcat(2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1 ** 2 + q2 ** 2)))

def quaternion_to_euler(q):
    q = pyquaternion.Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])
    yaw, pitch, roll = q.yaw_pitch_roll
    return np.array([roll, pitch, yaw])

def euler_to_quaternion(roll, pitch, yaw):
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return np.array([qw, qx, qy, qz])

def skew_symmetric(v):
    if isinstance(v, np.ndarray):
        return np.array([
            [0   , -v[0], -v[1], -v[2]],
            [v[0], 0    , v[2] , -v[1]],
            [v[1], -v[2], 0    , v[0] ],
            [v[2], v[1] , -v[0], 0    ]])
    else:
        return ca.vertcat(
            ca.horzcat(0   , -v[0], -v[1], -v[2]),
            ca.horzcat(v[0], 0    , v[2] , -v[1]),
            ca.horzcat(v[1], -v[2], 0    , v[0] ),
            ca.horzcat(v[2], v[1] , -v[0], 0    ))

def rotation_matrix_to_euler(r_mat):
    sy = math.sqrt(r_mat[0, 0] * r_mat[0, 0] + r_mat[1, 0] * r_mat[1, 0])

    singular = (sy < 1e-6)

    if not singular:
        x = math.atan2(r_mat[2, 1], r_mat[2, 2])
        y = math.atan2(-r_mat[2, 0], sy)
        z = math.atan2(r_mat[1, 0], r_mat[0, 0])
    else:
        x = math.atan2(-r_mat[1, 2], r_mat[1, 1])
        y = math.atan2(-r_mat[2, 0], sy)
        z = 0

    return np.array([x, y, z])

def rotation_matrix_to_quat(rot):
    """
    Calculate a quaternion from a 3x3 rotation matrix.

    :param rot: 3x3 numpy array, representing a valid rotation matrix
    :return: a quaternion corresponding to the 3D rotation described by the input matrix. Quaternion format: wxyz
    """

    q = pyquaternion.Quaternion(matrix=rot)
    return np.array([q.w, q.x, q.y, q.z])

def getReference_Quaternion_Bodyrates_RotorSpeed(v, a, j, s, yaw, yawdot, yawdotdot, model, dt):
    N_reference = len(v)
    q = np.zeros((N_reference, 4))
    euler_angle = np.zeros((N_reference, 3))
    br = np.zeros((N_reference, 3))
    brdot = np.zeros((N_reference, 3))
    u = np.zeros((N_reference, 4))
    thrust = np.zeros((N_reference,))
    thrustdot = np.zeros((N_reference,))
    m = model.mass
    g = np.array([0, 0, 9.81])
    Ginv = np.linalg.inv(model.G)
    Inertia = model.Inertia
    d = model.D
    Drad_A = model.A
    Drad_B = model.B
    dx, dy, dz = d[0, 0], d[1, 1], d[2, 2]
    kh = model.kh
    for i in range(N_reference):
        # Quaternions
        xc = np.array([math.cos(yaw[i]), math.sin(yaw[i]), 0])
        yc = np.array([-math.sin(yaw[i]), math.cos(yaw[i]), 0])
        alpha = m * (a[i] + g) + dx * v[i]
        beta = m * (a[i] + g) + dy * v[i]
        
        xb = np.cross(yc, alpha)
        # xb = v1_cross_v2(yc, alpha)
        xb = xb / np.linalg.norm(xb)
        yb = np.cross(beta, xb)
        # yb = v1_cross_v2(beta, xb)
        yb = yb / np.linalg.norm(yb)
        zb = np.cross(xb, yb)
        # zb = v1_cross_v2(xb, yb)
        rotation_mat = np.concatenate((xb.reshape(3, 1), yb.reshape(3, 1), zb.reshape(3, 1)), axis=1)
        q[i] = rotation_matrix_to_quat(rotation_mat)
        euler_angle[i] = quaternion_to_euler(q[i])
        # q[i] = euler_to_quaternion(euler_angle[i, 0], euler_angle[i, 1], euler_angle[i, 2])
        drag = kh * (np.dot(xb, v[i].T) ** 2 + np.dot(yb, v[i].T) ** 2)
        thrust[i] = np.dot(zb, m * a[i].T + m * g.T + dz * v[i].T) - drag
        
        # Bodyrates
        A_br = np.zeros((3, 3))
        b_br = np.zeros((3,))
        A_br[0, 1] = thrust[i] + (dx - dz) * np.dot(zb, v[i].T) + drag
        A_br[0, 2] = -(dx - dy) * np.dot(yb, v[i].T)
        A_br[1, 0] = thrust[i] + (dy - dz) * np.dot(zb, v[i].T) + drag
        A_br[1, 2] = (dx - dy) * np.dot(xb, v[i].T)
        A_br[2, 1] = -np.dot(yc, zb.T)
        A_br[2, 2] = np.linalg.norm(np.cross(yc, zb))
        b_br[0] = m * np.dot(xb, j[i].T) + dx * np.dot(xb, a[i].T)
        b_br[1] = -m * np.dot(yb, j[i].T) - dy * np.dot(yb, a[i].T)
        b_br[2] = yawdot[i] * np.dot(xc, xb.T)
        br[i] = np.linalg.solve(A_br, b_br)
        # br[i, 0] = -m * np.dot(yb, j[i].T) / thrust[i]
        # br[i, 1] = m * np.dot(xb, j[i].T) / thrust[i]
        # br[i, 2] = (yawdot[i] * np.dot(xc, xb.T) + br[i, 1] * np.dot(yc, zb.T)) / np.linalg.norm(np.cross(yc, zb))
        brx, bry, brz = br[i, 0], br[i, 1], br[i, 2]
        thrustdot[i] = m * np.dot(zb, j[i].T) + dz * np.dot(zb, a[i].T) + brx * np.dot(yb, v[i].T) * (dy + dz - 2 * kh * np.dot(zb, v[i].T)) - bry * np.dot(xb, v[i].T) * (dx + dz - 2 * kh * np.dot(zb, v[i].T))
        # thrustdot = np.gradient(thrust) / dt

        # Bodyratesdot
        b_brdot = np.zeros((3,))
        b_brdot[0] = m * np.dot(xb, s[i].T) + m * np.dot(brz * yb - bry * zb, j[i].T) + dx * (brz * np.dot(yb, a[i].T) - bry * np.dot(zb, a[i].T) + np.dot(xb, j[i].T)) - bry * (thrustdot[i] + (dx - dz) * (np.dot(bry * xb - brx * yb, v[i].T) + np.dot(zb, a[i].T)) + 2 * kh) + brz * (dx - dy) * (np.dot(-brz * xb + brx * zb, v[i].T + np.dot(yb, a[i].T)))
        b_brdot[1] = -m * np.dot(yb, s[i].T) - m * np.dot(-brz * xb + brx * zb, j[i].T) - dy * (-brz * np.dot(xb, a[i].T) + brx * np.dot(zb, a[i].T) + np.dot(yb, j[i].T)) - brx * (thrustdot[i] + (dy - dz)) - brz * (dx - dy)
        b_brdot[2] = yawdotdot[i] * np.dot(xc, xb.T) + (yawdot[i] ** 2 + bry ** 2) * np.dot(yc, xb.T) - 2 * yawdot[i] * bry * np.dot(xc, zb.T) - brx * bry * np.dot(yc, yb.T) + yawdot[i] * brz * np.dot(xc, yb.T)
        # b_brdot[0] = m * np.dot(xb, s[i].T) - 2 * thrustdot[i] * bry - thrust[i] * brx * brz
        # b_brdot[1] = -m * np.dot(yb, s[i].T) - 2 * thrustdot[i] * brx + thrust[i] * bry * brz
        # b_brdot[2] = yawdotdot[i] * np.dot(xc, xb.T) + 2 * yawdot[i] * (brz * np.dot(xc, yb.T) - bry * np.dot(xc, zb)) - brx * np.dot(yc, bry * yb.T + brz * zb.T)
        brdot[i] = np.linalg.solve(A_br, b_brdot)

        # u
        # tao = np.dot(Inertia, brdot[i].T) + np.dot(np.cross(br[i], Inertia), br[i].T)
        tao_d = -Drad_A.dot(rotation_mat.T.dot(v[i].T)) - Drad_B.dot(br[i].T)
        tao = np.dot(Inertia, brdot[i].T) + np.dot(np.dot(crossmat(br[i]), Inertia), br[i].T) - tao_d
        # print(tao)
        u[i] = np.sqrt(np.dot(Ginv, np.array([thrust[i], tao[0], tao[1], tao[2]]).T))

    return q, euler_angle, br, u

def draw_data_sim(x_data, x_sim, motor_data, t):
    euler_angle = np.zeros((len(x_data), 3))
    euler_angle_sim = np.zeros((len(x_data), 3))
    for i in range(len(x_data)):
        euler_angle[i] = quaternion_to_euler(x_data[i, 6:10])
        euler_angle_sim[i] = quaternion_to_euler(x_sim[i, 6:10])

    fig=plt.figure(figsize=(27,18))# ,figsize=(9,9)

    ax1=fig.add_subplot(2,6,1) # , projection='3d'
    ax1.set_title("pose")
    ax1.plot(t,x_data[:,0], label='x')
    ax1.plot(t,x_data[:,1], label='y')
    ax1.plot(t,x_data[:,2], label='z')
    ax1.legend()
    ax1.grid()

    ax2=fig.add_subplot(2,6,2)
    ax2.set_title("velocity")
    ax2.plot(t,x_data[:,3], label='x')
    ax2.plot(t,x_data[:,4], label='y')
    ax2.plot(t,x_data[:,5], label='z')
    ax2.legend()
    ax2.grid()

    ax3=fig.add_subplot(2,6,3)
    ax3.set_title("euler angle")
    ax3.plot(t,euler_angle[:, 0], label='x')
    ax3.plot(t,euler_angle[:, 1], label='y')
    ax3.plot(t,euler_angle[:, 2], '.', label='z')
    ax3.legend()
    ax3.grid()

    ax4=fig.add_subplot(2,6,4)
    ax4.set_title("quat")
    ax4.plot(t,x_data[:,6], label='w')
    ax4.plot(t,x_data[:,7], label='x')
    ax4.plot(t,x_data[:,8], label='y')
    ax4.plot(t,x_data[:,9], label='z')
    ax4.legend()
    ax4.grid()

    ax5=fig.add_subplot(2,6,5)
    ax5.set_title("bodyrate")
    ax5.plot(t,x_data[:,10], label='x')
    ax5.plot(t,x_data[:,11], label='y')
    ax5.plot(t,x_data[:,12], label='z')
    ax5.legend()
    ax5.grid()
    
    ax6=fig.add_subplot(2,6,6)
    ax6.set_title("motor speed")
    ax6.plot(t,motor_data[:, 0], label='u1')
    ax6.plot(t,motor_data[:, 1], label='u2')
    ax6.plot(t,motor_data[:, 2], label='u3')
    ax6.plot(t,motor_data[:, 3], label='u4')
    ax6.legend()
    ax6.grid()
    
    ax7=fig.add_subplot(2,6,7)
    ax7.set_title("sim pose")
    ax7.plot(t,x_sim[:,0], label='x')
    ax7.plot(t,x_sim[:,1], label='y')
    ax7.plot(t,x_sim[:,2], label='z')
    ax7.legend()
    ax7.grid()

    ax8=fig.add_subplot(2,6,8)
    ax8.set_title("sim velocity")
    ax8.plot(t,x_sim[:,3], label='x')
    ax8.plot(t,x_sim[:,4], label='y')
    ax8.plot(t,x_sim[:,5], label='z')
    ax8.legend()
    ax8.grid()

    ax9=fig.add_subplot(269)
    ax9.set_title("sim euler angle")
    ax9.plot(t,euler_angle_sim[:, 0], label='x')
    ax9.plot(t,euler_angle_sim[:, 1], label='y')
    ax9.plot(t,euler_angle_sim[:, 2], label='z')
    ax9.legend()
    ax9.grid()

    ax10=fig.add_subplot(2,6,10)
    ax10.set_title("sim quat")
    ax10.plot(t,x_sim[:,6], label='w')
    ax10.plot(t,x_sim[:,7], label='x')
    ax10.plot(t,x_sim[:,8], label='y')
    ax10.plot(t,x_sim[:,9], label='z')
    ax10.legend()
    ax10.grid()

    ax11=fig.add_subplot(2,6,11)
    ax11.set_title("sim bodyrate")
    ax11.plot(t,x_sim[:,10], label='x')
    ax11.plot(t,x_sim[:,11], label='y')
    ax11.plot(t,x_sim[:,12], label='z')
    ax11.legend()
    ax11.grid()

    ax12=fig.add_subplot(2,6,12, projection='3d')
    ax12.set_title("sim pose")
    ax12.plot(x_sim[:, 0],x_sim[:, 1], x_sim[:, 2], label='pose')
    ax12.legend()
    ax12.grid()
    plt.show()
    return

def parse_xacro_file(xacro):
    """
    Reads a .xacro file describing a robot for Gazebo and returns a dictionary with its properties.
    :param xacro: full path of .xacro file to read
    :return: a dictionary of robot attributes
    """

    tree = XMLtree.parse(xacro)

    attrib_dict = {}

    for node in tree.getroot().getchildren():
        # Get attributes
        attributes = node.attrib

        if 'value' in attributes.keys():
            attrib_dict[attributes['name']] = attributes['value']

        if node.getchildren():
            try:
                attrib_dict[attributes['name']] = [child.attrib for child in node.getchildren()]
            except:
                continue

    return attrib_dict