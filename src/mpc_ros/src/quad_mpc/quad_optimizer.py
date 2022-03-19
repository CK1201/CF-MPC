import os, scipy.linalg
import numpy as np
import casadi as ca
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

class QuadrotorOptimizer:
    def __init__(self, Tf, N, quadrotorModel) -> None:
        self.ocp = AcadosOcp()
        # acados_solver = AcadosOcpSolver(ocp)
        acados_source_path = os.environ['ACADOS_SOURCE_DIR']
        self.ocp.acados_include_path = acados_source_path + '/include'
        self.ocp.acados_lib_path = acados_source_path + '/lib'


        acModel = AcadosModel()
        self.quadrotorModel = quadrotorModel
        model, constraint = self.quadrotorModel.model, self.quadrotorModel.constraint

        # define acados ODE
        acModel.name = model.name
        acModel.x = model.x
        acModel.xdot = model.xdot
        acModel.u = model.u
        acModel.p = model.p
        # acModel.z = model.z
        acModel.f_expl_expr = model.f_expl_expr
        acModel.f_impl_expr = model.f_impl_expr
        self.ocp.model = acModel

        # define constraint
        # acModel.con_h_expr = constraint.expr

        # dimensions
        nx = acModel.x.size()[0]
        nu = acModel.u.size()[0]
        ny = nx + nu
        ny_e = nx
        np_ = acModel.p.size()[0]

        # nsbx = 0
        # nh = constraint.expr.shape[0]
        # nsh = nh
        # ns = nsh + nsbx
        # print(nh)

        # discretization
        self.ocp.dims.N = N
        self.ocp.dims.nbu = nu
        self.ocp.dims.nbx = nx
        self.ocp.dims.nbx_0 = nx
        self.ocp.dims.nbx_e = nx
        # self.ocp.dims.nh = nh
        self.ocp.dims.np = np_
        self.ocp.parameter_values = np.array([0])
        # set cost
        Q = np.diag(np.concatenate((np.ones(3) * 100, np.ones(3) * 0.02, np.ones(4) * 0.2, np.ones(3) * 0.01)))
        # Q = np.diag(np.concatenate((np.ones(3) * 10, np.ones(3) * 0.02, np.ones(4) * 0.2, np.ones(3) * 0.01)))
        # Q[-1,-1] *=  0.1
        # Q = np.diag(np.concatenate((np.ones(3) * 100, np.ones(3) * 0.00, np.ones(4) * 0.0, np.ones(3) * 0.00)))
        R = np.eye(nu) * 1 / model.RotorSpeed_max
 
        # self.ocp.cost.cost_type_0 = "NONLINEAR_LS" # EXTERNAL, LINEAR_LS, NONLINEAR_LS
        self.ocp.cost.cost_type = "LINEAR_LS"
        self.ocp.cost.cost_type_e = "LINEAR_LS"

        if self.ocp.cost.cost_type == "LINEAR_LS":
            self.ocp.cost.W = scipy.linalg.block_diag(Q, R)
            self.ocp.cost.W_e = scipy.linalg.block_diag(Q)

            self.ocp.cost.yref = np.concatenate((model.x0, np.zeros(nu)))
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


        elif self.ocp.cost.cost_type == "NONLINEAR_LS":

            # self.ocp.model.cost_y_expr_0 = ca.vertcat(model.x, acModel.p, model.u)
            self.ocp.model.cost_y_expr = ca.vertcat(acModel.x, acModel.p, acModel.u)
            self.ocp.model.cost_y_expr_e = ca.vertcat(acModel.x, acModel.p)

            # self.ocp.cost.W_0 = scipy.linalg.block_diag(Q, R)
            self.ocp.cost.W = scipy.linalg.block_diag(Q, np.zeros((np_, np_)), R)
            self.ocp.cost.W_e = scipy.linalg.block_diag(Q, np.zeros((np_, np_)))

            self.ocp.cost.yref = np.concatenate((model.x0, np.zeros(np_), np.zeros(nu)))
            self.ocp.cost.yref_e = np.concatenate((model.x0, np.zeros(np_)))
        # print(self.ocp.cost.W)



        # set intial condition
        self.ocp.constraints.x0 = model.x0

        # set intial references


        # setting constraints
        # state constraints
        # self.ocp.constraints.lbx_0 = np.array([])
        # self.ocp.constraints.ubx_0 = np.array([])
        # self.ocp.constraints.idxbx_0 = np.array([])

        # bodyrate constraint
        self.ocp.constraints.lbx = np.array([-model.BodyratesX, -model.BodyratesY, -model.BodyratesZ])
        self.ocp.constraints.ubx = np.array([model.BodyratesX,  model.BodyratesY,  model.BodyratesZ])
        self.ocp.constraints.idxbx = np.array(range(3)) + nx - 3

        # self.ocp.constraints.lbx = np.array([0, -model.BodyratesX, -model.BodyratesY, -model.BodyratesZ])
        # self.ocp.constraints.ubx = np.array([99999, model.BodyratesX,  model.BodyratesY,  model.BodyratesZ])
        # self.ocp.constraints.idxbx = np.concatenate((np.array([2]), np.array(range(3)) + nx - 3))
        # temp = np.array(range(4))
        # temp[0] = 2
        # temp[1:] = temp[1:] + nx - 3
        # self.ocp.constraints.idxbx = temp

        # self.ocp.constraints.lbx_e = np.array([-model.BodyratesX, -model.BodyratesY, -model.BodyratesZ])
        # self.ocp.constraints.ubx_e = np.array([ model.BodyratesX,  model.BodyratesY,  model.BodyratesZ])
        # self.ocp.constraints.idxbx_e = np.array(range(3)) + nx - 3

        # self.ocp.constraints.lsbx_e
        # self.ocp.constraints.idxbxe_0 = np.array([])

        # input constraints
        self.ocp.constraints.lbu = np.ones(nu) * model.RotorSpeed_min
        self.ocp.constraints.ubu = np.ones(nu) * model.RotorSpeed_max
        self.ocp.constraints.idxbu = np.array(range(nu))

        # ocp.constraints.lsbx = np.zeros([nsbx])
        # ocp.constraints.usbx = np.zeros([nsbx])
        # ocp.constraints.idxsbx = np.array(range(nsbx))

        # self.ocp.constraints.lh = np.array(
        #     [
        #         constraint.along_min,
        #         constraint.alat_min,
        #         model.n_min,
        #         model.throttle_min,
        #         model.delta_min,
        #     ]
        # )
        # self.ocp.constraints.uh = np.array(
        #     [
        #         constraint.along_max,
        #         constraint.alat_max,
        #         model.n_max,
        #         model.throttle_max,
        #         model.delta_max,
        #     ]
        # )
        # ocp.constraints.lsh = np.zeros(nsh)
        # ocp.constraints.ush = np.zeros(nsh)
        # ocp.constraints.idxsh = np.array(range(nsh))

        # set QP solver and integration
        self.ocp.solver_options.tf = Tf
        self.ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM" # 'PARTIAL_CONDENSING_HPIPM', 'FULL_CONDENSING_HPIPM', 'FULL_CONDENSING_QPOASES', 'PARTIAL_CONDENSING_QPDUNES', 'PARTIAL_CONDENSING_OSQP'
        self.ocp.solver_options.nlp_solver_type = "SQP" # SQP SQP_RTI
        self.ocp.solver_options.hessian_approx = "GAUSS_NEWTON" # 'GAUSS_NEWTON', 'EXACT'
        self.ocp.solver_options.integrator_type = "ERK"
        self.ocp.solver_options.print_level = 0
        # self.ocp.solver_options.sim_method_num_stages = 4
        # self.ocp.solver_options.sim_method_num_steps = 3
        # self.ocp.solver_options.nlp_solver_step_length = 0.05 # default: 1
        # self.ocp.solver_options.nlp_solver_max_iter = 200 # default: 100
        # self.ocp.solver_options.tol = 1e-4 
        # self.ocp.solver_options.nlp_solver_tol_comp = 1e-1 
        self.ocp.solver_options.qp_solver_warm_start = 1
        

        # create solver
        json_file = os.path.join('./' + model.name + '_acados_ocp.json')
        if os.path.exists(json_file):
            print("remove "+ json_file)
            os.remove(json_file)
        self.acados_solver = AcadosOcpSolver(self.ocp, json_file = json_file)
        # self.acados_integrator = AcadosSimSolver(self.ocp, json_file = json_file)