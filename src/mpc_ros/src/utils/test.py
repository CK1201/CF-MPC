import numpy as np
import matplotlib.pyplot as plt
# a = np.arange(12).reshape((4,3))
# print(a)
# print(np.gradient(a, axis=0))
# print(np.gradient(a, np.array([0.1, 0.3, 0.6, 1]), axis=0))

# a = np.array([0.1,0.2,0.3, 0.4, 0.5,0.5]) * 2
# a = np.concatenate((a[np.newaxis,:] / 2, np.ones((1,6)) * 0.5), axis=0)
# # print(a)
# a = a.min(0)[:,np.newaxis]
# # print(a)

# p1 = np.array([1,1,1])
# p2 = np.array([3,4,5])
# dp = np.repeat((p2 - p1)[np.newaxis,:], repeats=6, axis=0)
# a = np.repeat(a, repeats=3, axis=1)
# print(a)
# print(dp)
# print(dp * a)

# 4 5m/s
# 5 10m/s
# 6 15m/s

drag_file = '../../config/NM_coeff_1.npz'
fit_size = 5
A = np.zeros((fit_size,3,3))
B = np.zeros((fit_size,3,3))
D = np.zeros((fit_size,3,3))
A_coeff = np.random.rand(fit_size,2) * 0.01
B_coeff = np.random.rand(fit_size,3) * 0.01
D_coeff = np.random.rand(fit_size,3) * 0.3
D_coeff[:,2] = D_coeff[:,2] * 0.1
kh = np.random.rand(fit_size,1) * 0.01
for i in range(fit_size):
    A[i, 0, 1] = A_coeff[i, 0]
    A[i, 1, 0] = A_coeff[i, 1]
    B[i] = np.diag(B_coeff[i])
    D[i] = np.diag(D_coeff[i])
np.savez(drag_file,A=A,B=B,D=D,kh=kh)


# Drag = np.load('../../config/NM_coeff_2_all_for_now_.npz')
# Drag_D  = Drag['D']
# Drag_kh = Drag['kh']
# Drag_A  = Drag['A']
# Drag_B  = Drag['B']
# print(Drag_D[0])
# print(Drag_kh[0])
# print(Drag_A[0])
# print(Drag_B[0])
# print(np.sum(Drag_A[0], axis=1)[:2])
 
# Drag = np.load('../../config/Drag_coeff.npz')
# print(Drag['A'])
# print(Drag['B'])
# print(Drag['D'])
# print(Drag['kh'])
# print(np.diag(Drag['B'][0]))
# print(np.diag(Drag['D'][0]))
# D_ax: -0.0010074589818424833
# D_ay: 0.0008514445780673497
# D_bx: -0.0007469443535864694
# D_by: -0.0019832280422668073
# D_bz: -0.0020781549801982503
# D_dx: 0.1743045615872555
# D_dy: 0.16908576711100184
# D_dz: 0.01851075588266621
# kh: 0.003518680371287564


# a1 = np.array([[0],[2],[3]])
# a2 = np.array([[1],[1],[4]])
# a = np.concatenate((a1[:,0][:,np.newaxis],a2[:,0][:,np.newaxis]),axis=1)
# print(a)
# print(np.min(a,axis=1))




# print(np.mean(np.random.rand(10,3,3), axis=0))


# Data = np.load('../../config/NM_coeff_6_fun_val.npz')
# fun_val = Data['fun_val']
# print(fun_val[:,0])
# fig=plt.figure()
# plt.plot(fun_val[:,0])
# plt.show()