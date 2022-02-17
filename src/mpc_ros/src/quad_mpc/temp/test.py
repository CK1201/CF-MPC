import time, os, scipy.linalg
# from acados_template import AcadosOcp,AcadosOcpSolver
from quadrotor_acados import *
import numpy as np


def main():

    Tf = 4.0  # prediction horizon
    N = 50  # number of discretization steps
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
    simX = np.ndarray((N, nx))
    simU = np.ndarray((N, nu))
    # print(model.x0)
    s0 = model.x0[:3]
    sref_final = np.array([3, 3, 5])
    sref_N = np.array([3, 3, 5]) / 5
    tcomp_sum = 0
    tcomp_max = 0
    x0 = model.x0

    # simulate
    # update reference
    sref = sref_final
    for j in range(N):
        if j == 0:
            continue
        # yref = np.array([s0[0] + (sref[0] - s0[0]) * j / N, s0[1] + (sref[1] - s0[1]) * j / N, s0[2] + (sref[2] - s0[2]) * j / N, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        yref = np.array([s0[0] + (sref[0] - s0[0]) * j / N, s0[1] + (sref[1] - s0[1]) * j / N, s0[2] + (sref[2] - s0[2]) * j / N, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        # print("{}, {}, {}".format(s0[0] + (sref[0] - s0[0]) * j / N, s0[1] + (sref[1] - s0[1]) * j / N, s0[2] + (sref[2] - s0[2]) * j / N))
        acados_solver.set(j, "yref", yref)
    # yref_N = np.array([sref, 0, 0, 0, 0, 0])
    # yref_N=np.array([sref[0], sref[1], sref[2], 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    yref_N=np.array([sref[0], sref[1], sref[2], 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0])
    acados_solver.set(N, "yref", yref_N)

    # solve ocp
    t = time.time()
    status = acados_solver.solve()
    # print(acados_solver.acados_ocp.constraints.ubx)
    # print(status)
    if status != 0:
        print("acados returned status {} in closed loop iteration {}.".format(status, 1))
    elapsed = time.time() - t

    # manage timings
    tcomp_sum += elapsed
    if elapsed > tcomp_max:
        tcomp_max = elapsed

    # get solution
    
    # print(u0)
    for j in range(N):
        x0 = acados_solver.get(j, "x")
        u0 = acados_solver.get(j, "u")
        simX[j, :] = x0
        simU[j, :] = u0
    

    # check if one lap is done and break and remove entries beyond
    # if x0[0] > Sref[-1] + 0.1:
    #     # find where vehicle first crosses start line
    #     N0 = np.where(np.diff(np.sign(simX[:, 0])))[0][0]
    #     Nsim = i - N0  # correct to final number of simulation steps for plotting
    #     simX = simX[N0:i, :]
    #     simU = simU[N0:i, :]
    #     break

    # Plot Results
    for i in range(N):
        print("{:.2f}, {:.2f}, {:.2f}".format(simX[i, 0], simX[i, 1], simX[i, 2]))
        # print("{:.2f}, {:.2f}, {:.2f}, {:.2f}".format(simX[i, 6], simX[i, 7], simX[i, 8], simX[i, 9]))
        print("  {:.2f}, {:.2f}, {:.2f}\n  {:.2f}, {:.2f}, {:.2f}\n  {:.2f}, {:.2f}, {:.2f}".format(simX[i, 6], simX[i, 7], simX[i, 8], simX[i, 9], simX[i, 10], simX[i, 11], simX[i, 12], simX[i, 13], simX[i, 14]))
        rotation = np.array([[simX[i, 6], simX[i, 7], simX[i, 8]],[simX[i, 9], simX[i, 10], simX[i, 11]],[simX[i, 12], simX[i, 13], simX[i, 14]]])
        # rotation
        # print("  ",np.linalg.det(rotation))
        print("{}, {}, {}, {}".format(round(simU[i, 0]), round(simU[i, 1]), round(simU[i, 2]), round(simU[i, 3])))
        print()
    # t = np.linspace(0.0, Nsim * Tf / N, Nsim)
    
    # print(simU)
    print(acados_solver.get_cost())
    # print(acados_solver.get_residuals())
    # plotRes(simX, simU, t)
    # plotTrackProj(simX, track)
    # plotalat(simX, simU, constraint, t)

    # avoid plotting when running on Travis
    # if os.environ.get("ACADOS_ON_TRAVIS") is None:
        # plt.show()

if __name__ == '__main__':
    main()