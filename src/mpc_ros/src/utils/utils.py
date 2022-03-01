import math, pyquaternion
import numpy as np
import casadi as ca
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