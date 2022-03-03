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
    vec_mat = crossmat(vec1)
    return np.dot(vec_mat, vec2.T)

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
    w, x, y, z = q[0], q[1], q[2], q[3]

    if isinstance(q, np.ndarray):
        return np.array([w, -x, -y, -z])
    else:
        return ca.vertcat(w, -x, -y, -z)

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

def getReference_Quaternion_Bodyrates_RotorSpeed(v, a, j, s, yaw, yawdot, yawdotdot, model):
    N_reference = len(v)
    q = np.zeros((N_reference, 4))
    euler_angle = np.zeros((N_reference, 3))
    br = np.zeros((N_reference, 3))
    brdot = np.zeros((N_reference, 3))
    u = np.zeros((N_reference, 4))
    thrust = np.zeros((N_reference,))
    m = model.mass
    g = np.array([0, 0, 9.81])
    Ginv = np.linalg.inv(model.G)
    Inertia = model.Inertia
    d = model.D
    dx, dy, dz = d[0, 0], d[1, 1], d[2, 2]
    kh = model.kh
    for i in range(N_reference):
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
        euler_angle[i] = rotation_matrix_to_euler(rotation_mat)
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
        # print(np.dot(Ginv, np.array([thrust[i], tao[0], tao[1], tao[2]]).T))
        u[i] = np.sqrt(np.dot(Ginv, np.array([thrust[i], tao[0], tao[1], tao[2]]).T))

    return q, euler_angle, br, u

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