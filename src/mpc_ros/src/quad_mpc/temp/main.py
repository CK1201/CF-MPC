import time, os, scipy.linalg
# from acados_template import AcadosOcp,AcadosOcpSolver
from quadrotor_acados import *
import numpy as np


def main():

    Tf = 1.0  # prediction horizon
    N = 20  # number of discretization steps
    T = 2.00  # maximum simulation time[s]
    # sref_N = 3  # reference for final reference progress

    # load model
    model, constraint, acados_solver, acados_integrator = acados_setting(Tf, N)

    # dimensions
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ny = nx + nu
    Nsim = int(T * N / Tf)

    # initialize data structs
    simX = np.ndarray((Nsim, nx))
    simU = np.ndarray((Nsim, nu))
    # print(model.x0)
    s0 = model.x0[:3]
    sref_final = np.array([3, 3, 5])
    sref_N = np.array([3, 3, 5]) / 5
    tcomp_sum = 0
    tcomp_max = 0
    x0 = model.x0

    # simulate
    for i in range(Nsim):
        # print(x0[:3])
        # update reference
        sref = s0 + sref_N
        for j in range(N):
            if j == 0:
                continue
            # yref = np.array([s0 + (sref - s0) * j / N, 0, 0, 0, 0, 0, 0, 0])
            # yref=np.array([1,0,0,1,0,0,0,0])
            # yref = np.array([3, 3, 5, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            # yref = np.array([3 / N * j, 3 / N * j, 5 / N * j, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            yref = np.array([s0[0] + (sref[0] - s0[0]) * j / N, s0[1] + (sref[1] - s0[1]) * j / N, s0[2] + (sref[2] - s0[2]) * j / N, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            
            acados_solver.set(j, "yref", yref)
        # yref_N = np.array([sref, 0, 0, 0, 0, 0])
        yref_N=np.array([sref[0], sref[1], sref[2], 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        acados_solver.set(N, "yref", yref_N)

        # solve ocp
        t = time.time()
        status = acados_solver.solve()
        # print(acados_solver.acados_ocp.constraints.ubx)
        # print(status)
        if status != 0:
            print("acados returned status {} in closed loop iteration {}.".format(status, i))
        elapsed = time.time() - t

        # manage timings
        tcomp_sum += elapsed
        if elapsed > tcomp_max:
            tcomp_max = elapsed

        # get solution
        x0 = acados_solver.get(0, "x")
        u0 = acados_solver.get(0, "u")
        # print(u0)
        # for j in range(nx):
        simX[i, :] = x0
        # for j in range(nu):
        simU[i, :] = u0
        # print(u0)
        # print(x0)
        # print(u0)

        # acados_integrator.set('x', x0)
        # acados_integrator.set('u', u0)
        # # 仿真器计算结果
        # status_s = acados_integrator.solve()
        # if status_s != 0:
        #     raise Exception('acados integrator returned status {}. Exiting.'.format(status))

        # # 将仿真器计算的小车位置作为下一个时刻的初始值
        # x0 = acados_integrator.get('x')
        # print(x0)
        # print(u0)

        # update initial condition
        x0 = acados_solver.get(1, "x")
        acados_solver.set(0, "lbx", x0)
        acados_solver.set(0, "ubx", x0)
        s0 = x0[:3]

        # check if one lap is done and break and remove entries beyond
        # if x0[0] > Sref[-1] + 0.1:
        #     # find where vehicle first crosses start line
        #     N0 = np.where(np.diff(np.sign(simX[:, 0])))[0][0]
        #     Nsim = i - N0  # correct to final number of simulation steps for plotting
        #     simX = simX[N0:i, :]
        #     simU = simU[N0:i, :]
        #     break

    # Plot Results
    # t = np.linspace(0.0, Nsim * Tf / N, Nsim)
    # print(simX[:,:3])
    print(simX)
    print(simU)
    # plotRes(simX, simU, t)
    # plotTrackProj(simX, track)
    # plotalat(simX, simU, constraint, t)

    # Print some stats
    # print("Average computation time: {}".format(tcomp_sum / Nsim))
    # print("Maximum computation time: {}".format(tcomp_max))
    # print("Average speed:{}m/s".format(np.average(simX[:, 3: 6], 2)))
    # print("Lap time: {}s".format(Tf * Nsim / N))

    # avoid plotting when running on Travis
    # if os.environ.get("ACADOS_ON_TRAVIS") is None:
        # plt.show()

if __name__ == '__main__':
    main()