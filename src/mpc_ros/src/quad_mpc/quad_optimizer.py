import os, scipy.linalg
import numpy as np
import casadi as ca
from src.utils.utils import *
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

class QuadrotorOptimizer:
    def __init__(self, Tf, N, quadrotorModel, cost_type, num=0) -> None:
        self.ocp = AcadosOcp()
        # acados_solver = AcadosOcpSolver(ocp)
        acados_source_path = os.environ['ACADOS_SOURCE_DIR']
        self.ocp.acados_include_path = acados_source_path + '/include'
        self.ocp.acados_lib_path = acados_source_path + '/lib'


        acModel = AcadosModel()
        self.quadrotorModel = quadrotorModel
        model, constraint   = self.quadrotorModel.model, self.quadrotorModel.constraint

        # define acados ODE
        acModel.name        = model.name
        acModel.x           = model.x
        acModel.xdot        = model.xdot
        acModel.u           = model.u
        # acModel.z           = model.z
        acModel.f_expl_expr = model.f_expl_expr
        acModel.f_impl_expr = model.f_impl_expr
        self.ocp.model      = acModel

        # define constraint
        # acModel.con_h_expr = constraint.expr

        # dimensions
        nx   = acModel.x.size()[0]
        nu   = acModel.u.size()[0]
        ny   = nx + nu
        ny_e = nx
        if cost_type == "EXTERNAL":
            acModel.p = model.p
            np_  = acModel.p.size()[0]
            self.ocp.dims.np = np_
            self.ocp.parameter_values = np.zeros(np_)

        # nsbx = 0
        # nh = constraint.expr.shape[0]
        # nsh = nh
        # ns = nsh + nsbx
        # print(cost_type)

        # discretization
        self.ocp.dims.N     = N
        self.ocp.dims.nbu   = nu
        self.ocp.dims.nbx   = nx
        self.ocp.dims.nbx_0 = nx
        self.ocp.dims.nbx_e = nx
        # self.ocp.dims.nh = nh

        # self.ocp.cost.cost_type_0 = "NONLINEAR_LS" # EXTERNAL, LINEAR_LS, NONLINEAR_LS
        self.ocp.cost.cost_type = cost_type
        self.ocp.cost.cost_type_e = cost_type
        if cost_type == "LINEAR_LS":
            Q = np.diag(np.concatenate((np.ones(3) * 10, np.ones(3) * 0.02, np.ones(4) * 0.2, np.ones(3) * 0.01)))
            R = np.eye(nu) * 1 / model.RotorSpeed_max
            self.ocp.cost.W   = scipy.linalg.block_diag(Q, R)
            self.ocp.cost.W_e = scipy.linalg.block_diag(Q)

            self.ocp.cost.yref   = np.concatenate((model.x0, np.zeros(nu)))
            self.ocp.cost.yref_e = model.x0

            Vx = np.zeros((ny, nx))
            Vx[:nx, :nx] = np.eye(nx)
            self.ocp.cost.Vx = Vx

            Vu = np.zeros((ny, nu))
            Vu[-nu:, -nu:] = np.eye(nu)
            self.ocp.cost.Vu = Vu

            Vx_e = np.zeros((ny_e, nx))
            Vx_e[:nx, :nx] = np.eye(nx)
            self.ocp.cost.Vx_e = Vx_e

        elif cost_type == "EXTERNAL":
            # Q = np.diag(np.concatenate((np.ones(3) * 10, np.ones(3) * 0.02, np.ones(3) * 0.5, np.ones(3) * 0.01)))
            # R = np.eye(nu) / model.RotorSpeed_max

            Q = np.diag(np.concatenate((np.ones(3) * 200, np.ones(3) * 1, np.ones(3) * 5, np.ones(3) * 1)))
            # Q[2,2] = 500 # z
            Q[8,8] = 200 # yaw
            R = np.eye(nu) * 6 / model.RotorSpeed_max

            diff_q = diff_between_q_q(acModel.x[6:10], acModel.p[6:10])[1:]
            diff_state = ca.vertcat(acModel.x[:6] - acModel.p[:6], diff_q, acModel.x[10:13] - acModel.p[10:13])
            diff_input = acModel.u - acModel.p[nx:ny]

            # self.ocp.model.cost_expr_ext_cost = (ca.vertcat(acModel.x, acModel.u) - acModel.p[:ny]).T @ scipy.linalg.block_diag(Q, R) @ (ca.vertcat(acModel.x, acModel.u)  - acModel.p[:ny]) + SafetyCost * SafetyWeight
            # self.ocp.model.cost_expr_ext_cost_e = (acModel.x - acModel.p[:nx]).T @ Q @ (acModel.x - acModel.p[:nx]) + SafetyCost * SafetyWeight
            self.ocp.model.cost_expr_ext_cost = ca.vertcat(diff_state, diff_input).T @ scipy.linalg.block_diag(Q, R) @ ca.vertcat(diff_state, diff_input)
            self.ocp.model.cost_expr_ext_cost_e = diff_state.T @ Q @ diff_state

            if self.quadrotorModel.need_collision_free:
                SafetyWeight = Q[2,2] * 10
                SafetyCost = 0
                if self.quadrotorModel.useTwoPolyhedron:
                    for i in range(model.boxVertex.size()[1]):
                        cost1 = 0
                        cost2 = 0
                        for j in range (model.MaxNumOfPolyhedrons):
                            APolyhedron = acModel.p[ny + j * 4: ny + j * 4 + 3]
                            bPolyhedron = acModel.p[ny + j * 4 + 3]
                            cost1 += self.LossFunction(APolyhedron.T @ model.boxVertex[:,i] - bPolyhedron)
                            APolyhedron2 = acModel.p[ny + (model.MaxNumOfPolyhedrons + j) * 4: ny + (model.MaxNumOfPolyhedrons + j) * 4 + 3]
                            bPolyhedron2 = acModel.p[ny + (model.MaxNumOfPolyhedrons + j) * 4 + 3]
                            cost2 += self.LossFunction(APolyhedron2.T @ model.boxVertex[:,i] - bPolyhedron2)
                        # inside 1 corridor
                        # SafetyCost += cost1
                        # inside either 2 corridor
                        SafetyCost += ca.fmin(cost1, cost2)
                else:
                    for i in range (model.MaxNumOfPolyhedrons):
                        APolyhedron = acModel.p[ny + i * 4: ny + i * 4 + 3]
                        bPolyhedron = acModel.p[ny + i * 4 + 3]
                        for j in range(model.boxVertex.size()[1]):
                            SafetyCost += self.LossFunction(APolyhedron.T @ model.boxVertex[:,j] - bPolyhedron)
                self.ocp.model.cost_expr_ext_cost += SafetyCost * SafetyWeight
                self.ocp.model.cost_expr_ext_cost_e += SafetyCost * SafetyWeight

        # set intial condition
        self.ocp.constraints.x0 = model.x0

        # set intial references


        # setting constraints
        # state constraints
        # self.ocp.constraints.lbx_0 = np.array([])
        # self.ocp.constraints.ubx_0 = np.array([])
        # self.ocp.constraints.idxbx_0 = np.array([])

        # bodyrate constraint
        self.ocp.constraints.lbx   = np.array([-model.BodyratesX, -model.BodyratesY, -model.BodyratesZ])
        self.ocp.constraints.ubx   = np.array([model.BodyratesX,  model.BodyratesY,  model.BodyratesZ])
        self.ocp.constraints.idxbx = np.array(range(3)) + nx - 3

        # self.ocp.constraints.lbx_e = np.array([-model.BodyratesX, -model.BodyratesY, -model.BodyratesZ])
        # self.ocp.constraints.ubx_e = np.array([ model.BodyratesX,  model.BodyratesY,  model.BodyratesZ])
        # self.ocp.constraints.idxbx_e = np.array(range(3)) + nx - 3

        # self.ocp.constraints.lsbx_e
        # self.ocp.constraints.idxbxe_0 = np.array([])

        # input constraints
        self.ocp.constraints.lbu   = np.ones(nu) * model.RotorSpeed_min
        self.ocp.constraints.ubu   = np.ones(nu) * model.RotorSpeed_max
        self.ocp.constraints.idxbu = np.array(range(nu))

        # ocp.constraints.lsbx = np.zeros([nsbx])
        # ocp.constraints.usbx = np.zeros([nsbx])
        # ocp.constraints.idxsbx = np.array(range(nsbx))

        # self.ocp.constraints.lh = np.array(
        #     [
        #     ]
        # )
        # self.ocp.constraints.uh = np.array(
        #     [
        #     ]
        # )
        # ocp.constraints.lsh = np.zeros(nsh)
        # ocp.constraints.ush = np.zeros(nsh)
        # ocp.constraints.idxsh = np.array(range(nsh))

        # set QP solver and integration
        self.ocp.solver_options.tf              = Tf
        self.ocp.solver_options.qp_solver       = "FULL_CONDENSING_HPIPM" # 'PARTIAL_CONDENSING_HPIPM', 'FULL_CONDENSING_HPIPM', 'FULL_CONDENSING_QPOASES', 'PARTIAL_CONDENSING_QPDUNES', 'PARTIAL_CONDENSING_OSQP'
        self.ocp.solver_options.nlp_solver_type = "SQP" # SQP SQP_RTI
        self.ocp.solver_options.hessian_approx  = "GAUSS_NEWTON" # 'GAUSS_NEWTON', 'EXACT'
        self.ocp.solver_options.integrator_type = "ERK"
        self.ocp.solver_options.print_level     = 0
        # self.ocp.solver_options.sim_method_num_stages = 4
        # self.ocp.solver_options.sim_method_num_steps = 3
        # self.ocp.solver_options.nlp_solver_step_length = 0.05 # default: 1
        # self.ocp.solver_options.nlp_solver_max_iter = 200 # default: 100
        # self.ocp.solver_options.tol = 1e-4 
        # self.ocp.solver_options.nlp_solver_tol_comp = 1e-1 
        self.ocp.solver_options.qp_solver_warm_start = 1
        

        # create solver
        json_file = os.path.join('./' + model.name + '_' + str(num) + '_acados_ocp.json')
        if os.path.exists(json_file):
            print("remove " + json_file)
            os.remove(json_file)
        self.acados_solver = AcadosOcpSolver(self.ocp, json_file = json_file)
        print("create " + json_file)
        print()
        # self.acados_integrator = AcadosSimSolver(self.ocp, json_file = json_file)

    def LossFunction(self, x):
        # return ca.fabs(x ** 3)
        return ca.fmax(x ** 3, 0)