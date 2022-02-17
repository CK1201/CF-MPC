import os, scipy.linalg
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSimSolver
# from acados_template import utils


def acados_setting(Tf, N):
    
    ocp = AcadosOcp()
    # acados_solver = AcadosOcpSolver(ocp)
    acados_source_path = os.environ['ACADOS_SOURCE_DIR']
    ocp.acados_include_path = acados_source_path + '/include'
    ocp.acados_lib_path = acados_source_path + '/lib'


    acModel = AcadosModel()
    model, constraint = QuadrotorModel()

    # define acados ODE
    acModel.name = model.name
    acModel.x = model.x
    acModel.xdot = model.xdot
    acModel.u = model.u
    acModel.p = model.p
    acModel.z = model.z
    acModel.f_expl_expr = model.f_expl_expr
    acModel.f_impl_expr = model.f_impl_expr
    ocp.model = acModel

    # define constraint
    acModel.con_h_expr = constraint.expr

    # dimensions
    nx = acModel.x.size()[0]
    nu = acModel.u.size()[0]
    ny = nx + nu
    ny_e = nx
    np_ = acModel.p.size()[0]
    # print(nx)

    nsbx = 0
    nh = constraint.expr.shape[0]
    nsh = nh
    ns = nsh + nsbx
    # print(nh)

    # discretization
    ocp.dims.N = N
    ocp.dims.nbu = nu
    ocp.dims.nbx = nx
    ocp.dims.nbx_0 = nx
    ocp.dims.nbx_e = nx
    ocp.dims.nh = nh
    ocp.dims.np = np_

    # set cost
    # Q = np.diag([ 1e-1, 1e-8, 1e-8, 1e-8, 1e-3, 5e-3 ])
    Q = np.zeros((nx, nx))
    R = np.zeros((nu, nu))
    Q[:3, :3] = np.eye(3)
    # R[0, 0] = 1e-3
    # R[1, 1] = 5e-3

    # Qe = np.diag([ 5e0, 1e1, 1e-8, 1e-8, 5e-3, 2e-3 ])
    Qe = np.zeros((ny_e, ny_e))
    Qe[:3, :3] = np.eye(3)

    ocp.cost.cost_type = "LINEAR_LS" # EXTERNAL, LINEAR_LS, NONLINEAR_LS
    ocp.cost.cost_type_e = "LINEAR_LS"
    # unscale = N / Tf
    ocp.cost.W_0 = scipy.linalg.block_diag(Q, R)
    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Qe
    # print(ocp.cost.W)

    Vx = np.zeros((ny, nx))
    Vx[:nx, :nx] = np.eye(nx)
    ocp.cost.Vx = Vx

    Vu = np.zeros((ny, nu))
    Vu[-nu:, -nu:] = np.eye(nu)
    ocp.cost.Vu = Vu

    Vx_e = np.zeros((ny_e, nx))
    Vx_e[:nx, :nx] = np.eye(nx)
    ocp.cost.Vx_e = Vx_e

    # ocp.cost.zl = 100 * np.ones(ns)
    # ocp.cost.zu = 100 * np.ones(ns)
    # ocp.cost.Zl = 1 * np.ones(ns)
    # ocp.cost.Zu = 1 * np.ones(ns)

    # set intial condition
    ocp.constraints.x0 = model.x0

    # set intial references
    # ocp.cost.yref = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # ocp.cost.yref_e = np.array([3, 3, 5, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    ocp.cost.yref = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    ocp.cost.yref_e = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0])

    # setting constraints
    # state constraints
    # ocp.constraints.lbx_0 = np.array([])
    # ocp.constraints.ubx_0 = np.array([])
    # ocp.constraints.idxbx_0 = np.array([])

    # ocp.constraints.lbx = np.array([-100])
    # ocp.constraints.ubx = np.array([100])
    # ocp.constraints.idxbx = np.array([0])

    # ocp.constraints.lbx_e = np.array([])
    # ocp.constraints.ubx_e = np.array([])
    # ocp.constraints.idxbx_e = np.array([])

    # ocp.constraints.lsbx_e
    # ocp.constraints.idxbxe_0 = np.array([])

    # input constraints
    ocp.constraints.lbu = np.ones(nu) * model.RotorSpeed_min
    ocp.constraints.ubu = np.ones(nu) * model.RotorSpeed_max
    ocp.constraints.idxbu = np.array(range(nu))

    # ocp.constraints.lsbx = np.zeros([nsbx])
    # ocp.constraints.usbx = np.zeros([nsbx])
    # ocp.constraints.idxsbx = np.array(range(nsbx))

    ocp.constraints.lh = np.array(
        [
    #         constraint.along_min,
    #         constraint.alat_min,
    #         model.n_min,
    #         model.throttle_min,
    #         model.delta_min,
        ]
    )
    ocp.constraints.uh = np.array(
        [
    #         constraint.along_max,
    #         constraint.alat_max,
    #         model.n_max,
    #         model.throttle_max,
    #         model.delta_max,
        ]
    )
    # ocp.constraints.lsh = np.zeros(nsh)
    # ocp.constraints.ush = np.zeros(nsh)
    # ocp.constraints.idxsh = np.array(range(nsh))

    # set QP solver and integration
    ocp.solver_options.tf = Tf
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM" # 'PARTIAL_CONDENSING_HPIPM', 'FULL_CONDENSING_QPOASES', 'FULL_CONDENSING_HPIPM', 'PARTIAL_CONDENSING_QPDUNES', 'PARTIAL_CONDENSING_OSQP'
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON" # 'GAUSS_NEWTON', 'EXACT'
    ocp.solver_options.integrator_type = "ERK"
    # ocp.solver_options.print_level = 0
    # ocp.solver_options.sim_method_num_stages = 4
    # ocp.solver_options.sim_method_num_steps = 3
    # ocp.solver_options.nlp_solver_step_length = 0.05
    # ocp.solver_options.nlp_solver_max_iter = 200
    # ocp.solver_options.tol = 1e-4
    # ocp.solver_options.nlp_solver_tol_comp = 1e-1
    

    # create solver
    json_file = os.path.join('./' + model.name + '_acados_ocp.json')
    if os.path.exists(json_file):
        print("remove "+ json_file)
        os.remove(json_file)
    acados_solver = AcadosOcpSolver(ocp, json_file = json_file)
    acados_integrator = AcadosSimSolver(ocp, json_file = json_file)

    return model, constraint, acados_solver, acados_integrator

def QuadrotorModel():
    model = ca.types.SimpleNamespace()
    constraint = ca.types.SimpleNamespace()
    params = ca.types.SimpleNamespace()
    configuration = '+'
    g = 9.81 # gravity
    m = 0.68 # mass
    l = 0.17 # arm_length
    
    kT = 0.00000854858
    kM = 0.016
    D = np.zeros((3, 3))
    kh = 0
    # Inertia = np.eye(3) * 0.007
    # Inertia[2, 2] = 0.012
    Inertia = np.array([
        [0.007, 0    , 0    ],
        [0    , 0.007, 0    ],
        [0    , 0    , 0.012]
    ])

    # rotor
    RotorMass = 0.009
    RotorSpeed_min = 0
    RotorSpeed_max = 838 # rad/s
    RotorDrag_coefficient = 8.06428e-05
    RotorInertia = 10 * RotorMass
    RotorRadius = 0.1

    # e1 = np.eye(3)[:, 0]
    # e2 = np.eye(3)[:, 1]
    e3 = np.eye(3)[:, 2]
    # print(Inertia)

    K = np.zeros((4, 4))
    K[0, 0] = kT
    K[1, 1] = kT * l
    K[2, 2] = kT * l
    K[3, 3] = kT * kM

    if configuration == 'x':
        beta = np.pi/4
        G = np.array([[1            , 1            , 1            , 1           ],
                      [np.sin(beta) , -np.sin(beta), -np.sin(beta), np.sin(beta)],
                      [-np.cos(beta), -np.cos(beta), np.cos(beta) , np.cos(beta)],
                      [1            , -1           , 1            , -1          ]])
    elif configuration == '+':
        beta = 0
        G = np.array([[1            , 1            , 1            , 1           ],
                      [np.sin(beta) , -np.sin(beta), -np.sin(beta), np.sin(beta)],
                      [-np.cos(beta), -np.cos(beta), np.cos(beta) , np.cos(beta)],
                      [1            , -1           , 1            , -1          ]])
    
    G = K @ G
    # print(G)


    # name
    ModelName = "QuadrotorModel"

    # x
    p = ca.SX.sym("P", 3, 1)
    v = ca.SX.sym("V", 3, 1)
    Orientation = ca.SX.sym("Orientation", 4, 1)
    BodyRate = ca.SX.sym("BodyRate", 3, 1)
    x = ca.vertcat(p, v, Orientation, BodyRate)
    
    Orientation2 = ca.SX.sym("Orientation2", 9, 1)
    x2 = ca.vertcat(p, v, Orientation2, BodyRate)
    # print(x2)

    # xdot
    pDot = ca.SX.sym("PDot", 3, 1)
    vDot = ca.SX.sym("VDot", 3, 1)
    OrientationDot = ca.SX.sym("OrientationDot", 4, 1)
    BodyRateDot = ca.SX.sym("BodyRateDot", 3, 1)
    xdot = ca.vertcat(pDot, vDot, OrientationDot, BodyRateDot)

    Orientation2Dot = ca.SX.sym("Orientation2Dot", 9, 1)
    xdot2 = ca.vertcat(pDot, vDot, Orientation2Dot, BodyRateDot)

    # u
    RotorSpeed = ca.SX.sym("RotorSpeed", 4, 1)
    temp_input = G @ (ca.diag(RotorSpeed) @ RotorSpeed)
    # print(RotorSpeed)
    # print(ca.norm_2(RotorSpeed))

    # algebraic variables
    z = ca.vertcat([])

    # parameters
    p = ca.vertcat([])

    # f_expl
    BodyRateHat = vec2mat(BodyRate)
    RotationMat = quat2mat(Orientation)
    # xb = RotationMat[:, 0]
    # yb = RotationMat[:, 1]
    zb = RotationMat[:, 2]
    zb = zb / ca.norm_2(zb)
    vh = RotationMat.T @ v
    vh[2] = 0
    # print(tempBodyRate)
    f_expl = ca.vertcat(
        v,
        -g * e3 + (temp_input[0] * zb - RotationMat @ (D @ RotationMat.T @ v - kh * vh.T @ vh * e3)) / m,
        # -g * e3 + temp_input[0] * zb / m, 
        quatDot(Orientation, BodyRate),
        ca.inv(Inertia) @ (-BodyRateHat @ Inertia @ BodyRate + temp_input[1:])
    )


    RotationMat2 = ca.SX.zeros((3, 3))
    RotationMat2[0, 0] = Orientation2[0]
    RotationMat2[0, 1] = Orientation2[1]
    RotationMat2[0, 2] = Orientation2[2]
    RotationMat2[1, 0] = Orientation2[3]
    RotationMat2[1, 1] = Orientation2[4]
    RotationMat2[1, 2] = Orientation2[5]
    RotationMat2[2, 0] = Orientation2[6]
    RotationMat2[2, 1] = Orientation2[7]
    RotationMat2[2, 2] = Orientation2[8]
    zb2 = RotationMat2[:, 2]
    zb2 = zb2 / ca.norm_2(zb2)
    vh2 = RotationMat2.T @ v
    vh2[2] = 0

    f_expl2 = ca.vertcat(
        v,
        -g * e3 + (temp_input[0] * zb2 - RotationMat2 @ (D @ RotationMat2.T @ v - kh * vh2.T @ vh2 * e3)) / m,
        # -g * e3 + temp_input[0] * zb / m,
        ca_flatten(RotationMat2 @ BodyRateHat),
        ca.inv(Inertia) @ (-BodyRateHat @ Inertia @ BodyRate + temp_input[1:])
    )
    # print(temp_input[0])
    # print(temp_input[1:])

    # con_h
    con_h = None

    model.name = ModelName
    # model.x = x
    model.x = x2
    # model.xdot = xdot
    model.xdot = xdot2
    model.u = RotorSpeed
    model.p = p
    model.z = z
    # model.f_expl_expr = f_expl
    model.f_expl_expr = f_expl2
    # model.f_impl_expr = xdot - f_expl
    model.f_impl_expr = xdot2 - f_expl2
    model.con_h_expr = con_h
    model.params = params
    # model.x0 = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    model.x0 = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0])

    # Model bounds


    # state bounds


    # input bounds
    model.RotorSpeed_min = RotorSpeed_min
    model.RotorSpeed_max = RotorSpeed_max

    # define constraints struct
    # constraint.expr = ca.vertcat(RotorSpeed)
    constraint.expr = ca.vertcat([])
    
    return model, constraint

def quat2mat(quaternions):
    # quaternions = quaternions / ca.norm_2(quaternions)
    Mat = ca.SX.zeros(3, 3)
    q0 = quaternions[0]
    q1 = quaternions[1]
    q2 = quaternions[2]
    q3 = quaternions[3]
    Mat[0, 0] = 1 - 2 * q2 ** 2 - 2 * q3 ** 2
    Mat[0, 1] = 2 * q1 * q2 - 2 * q0 * q3
    Mat[0, 2] = 2 * q1 * q3 + 2 * q0 * q2
    Mat[1, 0] = 2 * q1 * q2 + 2 * q0 * q3
    Mat[1, 1] = 1 - 2 * q1 ** 2 - 2 * q3 ** 2
    Mat[1, 2] = 2 * q2 * q3 - 2 * q0 * q1
    Mat[2, 0] = 2 * q1 * q3 - 2 * q0 * q2
    Mat[2, 1] = 2 * q2 * q3 + 2 * q0 * q1
    Mat[2, 2] = 1 - 2 * q1 ** 2 - 2 * q2 ** 2
    # print(Mat)
    return Mat

def vec2mat(vec):
    Mat = ca.SX.zeros(3, 3)
    Mat[0, 1] = -vec[2]
    Mat[0, 2] = vec[1]
    Mat[1, 0] = vec[2]
    Mat[1, 2] = -vec[0]
    Mat[2, 0] = -vec[1]
    Mat[2, 1] = vec[0]
    return Mat

def quatDot(quat, bodyrate):
    temp = ca.SX.zeros(4, 4)
    temp[1:, 0] = bodyrate
    temp[0, 1] = -bodyrate[0]
    temp[0, 2] = -bodyrate[1]
    temp[0, 3] = -bodyrate[2]
    temp[1, 2] =  bodyrate[2]
    temp[1, 3] = -bodyrate[1]
    temp[2, 1] = -bodyrate[2]
    temp[2, 3] =  bodyrate[0]
    temp[3, 1] =  bodyrate[1]
    temp[3, 2] = -bodyrate[0]
    # quat = quat / ca.norm_2(quat)
    # print(ca.norm_2(quat))
    return temp @ quat / 2

def plotRes(simX,simU,t):
    # plot results
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.step(t, simU[:,0], color='r')
    plt.step(t, simU[:,1], color='g')
    plt.title('closed-loop simulation')
    # plt.legend(['dD','ddelta'])
    plt.ylabel('u')
    plt.xlabel('t')
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(t, simX[:,:])
    plt.ylabel('x')
    plt.xlabel('t')
    # plt.legend(['s','n','alpha','v','D','delta'])
    plt.grid(True)

def ca_flatten(Mat):
    [row, col] = Mat.size()
    vector = ca.SX.zeros((row * col, 1))
    for i in range(row):
        for j in range(col):
            vector[i * col + j] = Mat[i, j]
    # print(vector)
    return vector