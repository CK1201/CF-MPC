import imp
import matplotlib.pyplot as plt
import numpy as np
import os

expriment_name = 'circle_speedup'

dir_path = os.path.dirname(os.path.realpath(__file__))
result_dir = dir_path + '/../../result'
result_without_drag_file = os.path.join(result_dir, expriment_name + '_without_drag.txt')
result_with_drag_file = os.path.join(result_dir, expriment_name + '_with_drag.txt')

result_without_drag = np.loadtxt(result_without_drag_file, dtype=np.float32, delimiter=',')
result_with_drag = np.loadtxt(result_with_drag_file, dtype=np.float32, delimiter=',')
# print(b)

# pose_error_max_without_drag = np.array([0.74809107, 0.74100097, 0.79902218, 0.76305571, 0.84650872, 0.76334308, 0.79035327, 0.77098438, 0.83471673, 0.83832674])
# pose_error_mean_without_drag = np.array([0.28662606, 0.28535913, 0.28739616, 0.28575097, 0.28657789, 0.28688645, 0.28459639, 0.28757265, 0.28565073, 0.28441621])
# max_v_w_without_drag = np.array([10.12084314, 10.33189263, 10.10125251, 10.25014062, 10.1946716,  10.30134494, 10.26589725, 10.17294444, 10.06803158, 10.13178454])
# max_motor_speed_without_drag = np.array([738.74439186, 747.81930548, 743.21437812, 755.94499222, 742.47719225, 752.41686837, 749.72340215, 747.78475654, 742.72735193, 750.55435501])


# pose_error_max_with_drag = np.array([0.57056934, 0.63014374, 0.54378736, 0.55965151, 0.5329468,  0.64251292, 0.56450603, 0.58245595, 0.59730979, 0.58949428])
# pose_error_mean_with_drag = np.array([0.1571584,  0.1574115,  0.15467583, 0.15239329, 0.1540452,  0.15175052, 0.16317574, 0.15493355, 0.15214415, 0.15371897])
# max_v_w_with_drag = np.array([10.41131621, 10.6055555,  10.30476591, 10.48492457, 10.34499053, 10.51230155, 10.37245198, 10.39032776, 10.37406237, 10.38229301])
# max_motor_speed_with_drag = np.array([742.96301493, 747.10923808, 746.62171299, 752.38161284, 744.19778587, 748.46916278, 749.85824336, 748.65010096, 750.76663813, 750.58794057])

# print(np.mean(result_without_drag, axis=1, keepdims=True))
# print()
# print(np.mean(result_with_drag, axis=1, keepdims=True))

plot = 'both'
fig=plt.figure(figsize=(10,10))
ax1=fig.add_subplot(2,2,1)
ax2=fig.add_subplot(2,2,2)
ax3=fig.add_subplot(2,2,3)
ax4=fig.add_subplot(2,2,4)
ax1.set_title("Max Position Error")
ax1.set_ylabel("m")
ax1.grid(linestyle='--', alpha=0.8)

ax2.set_title("Mean Position Error")
ax2.set_ylabel("m")
ax2.grid(linestyle='--', alpha=0.8)

ax3.set_title("Max Motor Speed")
ax3.grid(linestyle='--', alpha=0.8)

ax4.set_title("Max Velocity")
ax4.set_ylabel("m/s")
ax4.grid(linestyle='--', alpha=0.8)
if plot == 'both':
    
    # ax1.set_title("Max Position Error")
    ax1.boxplot([result_without_drag[0], result_with_drag[0]], labels=['without drag', 'with drag'], showmeans=True, boxprops={'linewidth':'1.5'})
    # ax1.set_ylabel("m")
    # ax1.grid(linestyle='--', alpha=0.8)

    # ax2.set_title("Mean Position Error")
    ax2.boxplot([result_without_drag[1], result_with_drag[1]], labels=['without drag', 'with drag'], showmeans=True, boxprops={'linewidth':'1.5'})
    # ax2.set_ylabel("m")
    # ax2.grid(linestyle='--', alpha=0.8)

    # ax3.set_title("Max Motor Speed")
    ax3.boxplot([result_without_drag[3] / 828, result_with_drag[3] / 838], labels=['without drag', 'with drag'], showmeans=True, boxprops={'linewidth':'1.5'})
    # ax3.grid(linestyle='--', alpha=0.8)

    # ax4.set_title("Max Velocity")
    ax4.boxplot([result_without_drag[2], result_with_drag[2]], labels=['without drag', 'with drag'], showmeans=True, boxprops={'linewidth':'1.5'})
    # ax4.set_ylabel("m/s")
    # ax4.grid(linestyle='--', alpha=0.8)
elif plot == 'without_drag':
    # ax1.set_title("Max Position Error")
    ax1.boxplot([result_without_drag[0]], labels=['without drag'], showmeans=True, boxprops={'linewidth':'1.5'})
    # ax1.set_ylabel("m")
    # ax1.grid(linestyle='--', alpha=0.8)

    # ax2.set_title("Mean Position Error")
    ax2.boxplot([result_without_drag[1]], labels=['without drag'], showmeans=True, boxprops={'linewidth':'1.5'})
    # ax2.set_ylabel("m")
    # ax2.grid(linestyle='--', alpha=0.8)

    # ax3.set_title("Max Motor Speed")
    ax3.boxplot([result_without_drag[3] / 828], labels=['without drag'], showmeans=True, boxprops={'linewidth':'1.5'})
    # ax3.grid(linestyle='--', alpha=0.8)

    # ax4.set_title("Max Velocity")
    ax4.boxplot([result_without_drag[2]], labels=['without drag'], showmeans=True, boxprops={'linewidth':'1.5'})
    # ax4.set_ylabel("m/s")
    # ax4.grid(linestyle='--', alpha=0.8)
elif plot == 'with_drag':
    # ax1.set_title("Max Position Error")
    ax1.boxplot([result_with_drag[0]], labels=['with drag'], showmeans=True, boxprops={'linewidth':'1.5'})
    # ax1.set_ylabel("m")
    # ax1.grid(linestyle='--', alpha=0.8)

    # ax2.set_title("Mean Position Error")
    ax2.boxplot([result_with_drag[1]], labels=['with drag'], showmeans=True, boxprops={'linewidth':'1.5'})
    # ax2.set_ylabel("m")
    # ax2.grid(linestyle='--', alpha=0.8)

    # ax3.set_title("Max Motor Speed")
    ax3.boxplot([result_with_drag[3] / 828], labels=['with drag'], showmeans=True, boxprops={'linewidth':'1.5'})
    # ax3.grid(linestyle='--', alpha=0.8)

    # ax4.set_title("Max Velocity")
    ax4.boxplot([result_with_drag[2]], labels=['with drag'], showmeans=True, boxprops={'linewidth':'1.5'})
    # ax4.set_ylabel("m/s")
    # ax4.grid(linestyle='--', alpha=0.8)

plt.show()