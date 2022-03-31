import os
import casadi as ca
import numpy as np
from src.utils.utils import *

class QuadrotorModel:
    def __init__(self, quad_name = 'hummingbird', configuration = '+', Drag_D = np.zeros((3, 3)), Drag_kh = 0, Drag_A = np.zeros((3, 3)), Drag_B = np.zeros((3, 3))) -> None:

        this_path = os.path.dirname(os.path.realpath(__file__))
        params_file = os.path.join(this_path, '..', '..', 'xacro', quad_name + '.xacro')

        # Get parameters for drone
        attrib = parse_xacro_file(params_file)

        # quad.max_thrust = float(attrib["max_rot_velocity"]) ** 2 * float(attrib["motor_constant"])
        self.model = ca.types.SimpleNamespace()
        self.constraint = ca.types.SimpleNamespace()
        # params = ca.types.SimpleNamespace()
        self.configuration = configuration
        self.g = ca.vertcat(0, 0, 9.81) # gravity
        self.mass = float(attrib['mass']) + float(attrib['mass_rotor']) * 4
        self.body_width = float(attrib['body_width'])
        self.body_height = float(attrib['body_height'])
        self.arm_length = float(attrib['arm_length'])
        
        self.kT = float(attrib["motor_constant"])
        self.kM = float(attrib['moment_constant'])
        self.D = Drag_D
        self.kh = Drag_kh
        self.A = Drag_A
        self.B = Drag_B
        self.Inertia = np.array([
            [float(attrib['body_inertia'][0]['ixx']), float(attrib['body_inertia'][0]['ixy']), float(attrib['body_inertia'][0]['ixz'])],
            [float(attrib['body_inertia'][0]['ixy']), float(attrib['body_inertia'][0]['iyy']), float(attrib['body_inertia'][0]['iyz'])],
            [float(attrib['body_inertia'][0]['ixz']), float(attrib['body_inertia'][0]['iyz']), float(attrib['body_inertia'][0]['izz'])]
        ])

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
        self.G = np.diag([self.kT, self.kT, self.kT, self.kT * self.kM]) @ self.G

        # x
        p = ca.SX.sym("P", 3, 1)
        v = ca.SX.sym("V", 3, 1)
        Orientation = ca.SX.sym("Orientation", 4, 1)
        BodyRate = ca.SX.sym("BodyRate", 3, 1)
        x = ca.vertcat(p, v, Orientation, BodyRate)

        RotationMat = quat_to_rotation_matrix(unit_quat(Orientation))

        boxVertexNp = np.array([[self.arm_length, 0, 0],
                              [self.arm_length, 0, self.body_height],
                              [-self.arm_length, 0, 0],
                              [-self.arm_length, 0, self.body_height],
                              [0, self.arm_length, 0],
                              [0, self.arm_length, self.body_height],
                              [0, -self.arm_length, 0],
                              [0, -self.arm_length, self.body_height]])
        self.model.boxVertex = ca.SX.zeros(3, 8)
        # Rotation = ca.SX.sym("Rotation", 3, 3)
        for i in range (len(boxVertexNp)):
            self.model.boxVertex[:,i] = RotationMat @ boxVertexNp[i] + p
        # self.boxVertex[0] = (ca.SX.sym("Rotation", 3, 3) @ self.boxVertex[0]).T
        # print(Rotation)
        # print(self.boxVertex)
        # print(ca.SX.sym("Rotation", 3, 3))
        # print((ca.SX.sym("Rotation", 3, 3) @ self.boxVertex[1]).size())

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
        self.model.MaxNumOfPolyhedrons = 10
        param = ca.SX.sym("param", x.size()[0] + RotorSpeed.size()[0] + self.model.MaxNumOfPolyhedrons * 4, 1)
        # print(param[4:6].T @ param[4:6])
        # print(param[:3].T @ self.boxVertex[1].T)
        # f_expl
        # Orientation = ca.fabs(Orientation)
        BodyRateHat = crossmat(BodyRate)
        
        # xb = RotationMat[:, 0]
        # yb = RotationMat[:, 1]
        # zb = RotationMat[:, 2]
        # zb = zb / ca.norm_2(zb)
        vh = RotationMat.T @ v
        vh[2] = 0
        tao_d = - self.A @ RotationMat.T @ v - self.B @ BodyRate
        # print(tempBodyRate)
        f_expl = ca.vertcat(
            v,
            v_dot_q(ca.vertcat(0, 0, temp_input[0] / self.mass), Orientation) - self.g - RotationMat @ (self.D @ RotationMat.T @ v - ca.vertcat(0, 0, self.kh * vh.T @ vh)) / self.mass,
            1 / 2 * skew_symmetric(BodyRate) @ Orientation,
            ca.inv(self.Inertia) @ (-BodyRateHat @ self.Inertia @ BodyRate + temp_input[1:] + tao_d)
        )
        
        # con_h
        
        # print(RotationMat @ boxVertex[0].T + p)
        # con_h1 = bodyFrame + boxVertex
        # RotationMat @ self.body_width @ self.body_height
        # con_h = ca.vertcat([[RotationMat @ boxVertex[i].T + p] for i in range(len(boxVertex))])
        # print(con_h)
        con_h = None

        self.model.name = quad_name
        self.model.x = x
        self.model.xdot = xdot
        self.model.u = RotorSpeed
        self.model.p = param
        # self.model.z = z
        self.model.f_expl_expr = f_expl
        self.model.f_impl_expr = xdot - f_expl
        self.model.con_h_expr = con_h
        # self.model.params = params
        self.model.x0 = np.concatenate((np.zeros(6), np.array([1, 0, 0, 0]), np.zeros(3)))
        # self.model.x0 = np.concatenate((np.zeros(6), flatten(np.eye(3)), np.zeros(3)),axis=0)
        # Model bounds

        # state bounds
        self.model.BodyratesX = np.pi * 2
        self.model.BodyratesY = np.pi * 2
        self.model.BodyratesZ = np.pi * 0.5
        # input bounds
        self.model.RotorSpeed_min = RotorSpeed_min
        self.model.RotorSpeed_max = RotorSpeed_max

        # define constraints struct
        # self.constraint.expr = ca.vertcat([])
    
    def Simulation(self, p0, v0, q0, br0, u, dt):
        x = [p0, v0, q0, br0]
        k1 = [self.f_p(x), self.f_v(x, u), self.f_q(x), self.f_br(x, u)]
        x_aux = [x[i] + dt / 2 * k1[i] for i in range(4)]
        x_aux = self.unit_x(x_aux)
        k2 = [self.f_p(x_aux), self.f_v(x_aux, u), self.f_q(x_aux), self.f_br(x_aux, u)]
        x_aux = [x[i] + dt / 2 * k2[i] for i in range(4)]
        x_aux = self.unit_x(x_aux)
        k3 = [self.f_p(x_aux), self.f_v(x_aux, u), self.f_q(x_aux), self.f_br(x_aux, u)]
        x_aux = [x[i] + dt * k3[i] for i in range(4)]
        x_aux = self.unit_x(x_aux)
        k4 = [self.f_p(x_aux), self.f_v(x_aux, u), self.f_q(x_aux), self.f_br(x_aux, u)]
        x = [x[i] + dt / 6 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) for i in range(4)]
        # x[2] = unit_quat(x[2])
        return self.unit_x(x)

    def unit_x(self, x):
        return [x[0], x[1], unit_quat(x[2]), x[3]]

    def f_p(self, x):
        return x[1]

    def f_v(self, x, u):
        a_thrust = np.array([0, 0, np.dot(self.G, u.T ** 2)[0]]) / self.mass
        rotMat = quat_to_rotation_matrix(np.array(x[2]))
        vh = rotMat.T.dot(np.array(x[1]).T)
        vh[2] = 0
        drag = rotMat.dot(self.D).dot(rotMat.T).dot(np.array(x[1]).T) - self.kh * vh.dot(vh) * rotMat[:,2]
        return -np.array([0, 0, 9.81]) + v_dot_q(a_thrust, np.array(x[2])) - drag / self.mass
        
    def f_q(self, x):
        return 1 / 2 * skew_symmetric(x[3]).dot(x[2])

    def f_br(self, x, u):
        rotMat = quat_to_rotation_matrix(np.array(x[2]))
        tao = np.dot(self.G, u.T ** 2)[1:]
        tao_d = -self.A.dot(rotMat.T).dot(np.array(x[1]).T) - self.B.dot(u.T)
        # return np.dot(np.linalg.inv(self.Inertia), tao.T - np.dot(np.dot(crossmat(x[3]), self.Inertia), np.array(x[3]).T))
        return np.dot(np.linalg.inv(self.Inertia), tao.T + tao_d.T - crossmat(x[3]).dot(self.Inertia).dot(np.array(x[3]).T))