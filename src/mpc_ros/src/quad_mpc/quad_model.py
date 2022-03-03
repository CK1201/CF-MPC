import os
from re import I
import casadi as ca
import numpy as np
from src.utils.utils import *

class QuadrotorModel:
    def __init__(self, quad_name = 'hummingbird', configuration = '+') -> None:

        this_path = os.path.dirname(os.path.realpath(__file__))
        params_file = os.path.join(this_path, '..', '..', 'config', quad_name + '.xacro')

        # Get parameters for drone
        attrib = parse_xacro_file(params_file)

        # quad.max_thrust = float(attrib["max_rot_velocity"]) ** 2 * float(attrib["motor_constant"])
        # quad.c = float(attrib['moment_constant'])
        self.model = ca.types.SimpleNamespace()
        self.constraint = ca.types.SimpleNamespace()
        # params = ca.types.SimpleNamespace()
        self.configuration = configuration
        self.g = ca.vertcat(0, 0, 9.81) # gravity
        self.mass = float(attrib['mass']) + float(attrib['mass_rotor']) * 4
        self.arm_length = float(attrib['arm_length'])
        
        self.kT = float(attrib["motor_constant"])
        self.kM = float(attrib['moment_constant'])
        self.D = np.zeros((3, 3))
        self.kh = 0
        self.Inertia = np.array([
            [float(attrib['body_inertia'][0]['ixx']), float(attrib['body_inertia'][0]['ixy']), float(attrib['body_inertia'][0]['ixz'])],
            [float(attrib['body_inertia'][0]['ixy']), float(attrib['body_inertia'][0]['iyy']), float(attrib['body_inertia'][0]['iyz'])],
            [float(attrib['body_inertia'][0]['ixz']), float(attrib['body_inertia'][0]['iyz']), float(attrib['body_inertia'][0]['izz'])]
        ])
        
        # print(self.Inertia)

        # rotor
        RotorSpeed_min = 0
        RotorSpeed_max = float(attrib["max_rot_velocity"]) # rad/s
        RotorDrag_coefficient = float(attrib["rotor_drag_coefficient"])

        self.thrust_max = self.kT * (RotorSpeed_max ** 2)

        if self.configuration == 'x':
            arm_length_act = self.arm_length * np.cos(np.pi/4)
            self.G = np.array([[1, 1, 1, 1],
                          [arm_length_act , -arm_length_act, -arm_length_act, arm_length_act],
                          [-arm_length_act, -arm_length_act, arm_length_act, arm_length_act],
                          [1, -1, 1, -1]])
        elif self.configuration == '+':
            self.G = np.array([[1, 1, 1, 1],
                          [0, self.arm_length , 0, -self.arm_length],
                          [-self.arm_length, 0 , self.arm_length, 0],
                          [-1, 1, -1, 1]])
        self.G = ca.mtimes(np.diag([self.kT, self.kT, self.kT, self.kT * self.kM]), self.G)

        # x
        p = ca.SX.sym("P", 3, 1)
        v = ca.SX.sym("V", 3, 1)
        Orientation = ca.SX.sym("Orientation", 4, 1)
        BodyRate = ca.SX.sym("BodyRate", 3, 1)
        x = ca.vertcat(p, v, Orientation, BodyRate)

        # xdot
        pDot = ca.SX.sym("PDot", 3, 1)
        vDot = ca.SX.sym("VDot", 3, 1)
        OrientationDot = ca.SX.sym("OrientationDot", 4, 1)
        BodyRateDot = ca.SX.sym("BodyRateDot", 3, 1)
        xdot = ca.vertcat(pDot, vDot, OrientationDot, BodyRateDot)

        # u
        RotorSpeed = ca.SX.sym("RotorSpeed", 4, 1)
        temp_input = self.G @ (RotorSpeed ** 2)
        # print(temp_input[0])
        
        # algebraic variables
        # z = ca.vertcat([])

        # parameters
        # p = ca.vertcat([])

        # f_expl
        BodyRateHat = crossmat(BodyRate)
        # Orientation = unit_quat(Orientation)
        # RotationMat = quat_to_rotation_matrix(Orientation)
        # xb = RotationMat[:, 0]
        # yb = RotationMat[:, 1]
        # zb = RotationMat[:, 2]
        # zb = zb / ca.norm_2(zb)
        # vh = RotationMat.T @ v
        # vh[2] = 0
        # print(tempBodyRate)
        f_expl = ca.vertcat(
            v,
            # -g * e3 + (temp_input[0] * zb - RotationMat @ (self.D @ RotationMat.T @ v - self.kh * vh.T @ vh * e3)) / self.mass,
            # -g * e3 + (f_thrust - RotationMat @ (self.D @ RotationMat.T @ v - self.kh * vh.T @ vh * e3)) / self.mass,
            v_dot_q(ca.vertcat(0, 0, temp_input[0] / self.mass), Orientation) - self.g,
            1 / 2 * skew_symmetric(BodyRate) @ Orientation,
            ca.inv(self.Inertia) @ (-BodyRateHat @ self.Inertia @ BodyRate + temp_input[1:])
        )
        
        # con_h
        con_h = None

        self.model.name = quad_name
        self.model.x = x
        self.model.xdot = xdot
        self.model.u = RotorSpeed
        # self.model.p = p
        # self.model.z = z
        self.model.f_expl_expr = f_expl
        self.model.f_impl_expr = xdot - f_expl
        self.model.con_h_expr = con_h
        # self.model.params = params
        self.model.x0 = np.concatenate((np.zeros(6), np.array([1, 0, 0, 0]), np.zeros(3)))
        # self.model.x0 = np.concatenate((np.zeros(6), flatten(np.eye(3)), np.zeros(3)),axis=0)
        # Model bounds


        # state bounds
        self.model.BodyratesX = 2 * np.pi
        self.model.BodyratesY = 2 * np.pi
        self.model.BodyratesZ = np.pi / 4

        # input bounds
        self.model.RotorSpeed_min = RotorSpeed_min
        self.model.RotorSpeed_max = RotorSpeed_max

        # define constraints struct
        # self.constraint.expr = ca.vertcat([])
    
    def Simulation(self, p0, v0, q0, br0, u, dt):
        x = [p0, v0, q0, br0]
        k1 = [self.f_p(x), self.f_v(x, u), self.f_q(x), self.f_br(x, u)]
        x_aux = [x[i] + dt / 2 * k1[i] for i in range(4)]
        k2 = [self.f_p(x_aux), self.f_v(x_aux, u), self.f_q(x_aux), self.f_br(x_aux, u)]
        x_aux = [x[i] + dt / 2 * k2[i] for i in range(4)]
        k3 = [self.f_p(x_aux), self.f_v(x_aux, u), self.f_q(x_aux), self.f_br(x_aux, u)]
        x_aux = [x[i] + dt * k3[i] for i in range(4)]
        k4 = [self.f_p(x_aux), self.f_v(x_aux, u), self.f_q(x_aux), self.f_br(x_aux, u)]
        x = [x[i] + dt * (1.0 / 6.0 * k1[i] + 2.0 / 6.0 * k2[i] + 2.0 / 6.0 * k3[i] + 1.0 / 6.0 * k4[i]) for i in range(4)]
        x[2] = unit_quat(x[2])
        return x

    def f_p(self, x):
        return x[1]

    def f_v(self, x, u):
        rotation_mat0 = quat_to_rotation_matrix(x[2])
        temp_input = np.dot(self.G, u.T ** 2)
        return -np.array([0, 0, 9.81]) + temp_input[0] / self.mass * rotation_mat0[:, 2]
        
    def f_q(self, x):
        return 1 / 2 * np.dot(skew_symmetric(x[3]), x[2].T)

    def f_br(self, x, u):
        temp_input = np.dot(self.G, u.T ** 2)
        return np.dot(np.linalg.inv(self.Inertia), (temp_input[1:] - np.dot(v1_cross_v2(x[3], self.Inertia), x[3].T)).T)
