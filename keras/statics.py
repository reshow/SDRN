import numpy as np
import scipy.io as sio
import os


def getAngleList(data_dir):
    angle_list = []
    i = 0
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            file_name = file.split('.')[0]
            file_type = file.split('.')[1]
            if file_type == 'mat':
                i += 1
                print('\r', root + file_name, i, end='      ')
                m = sio.loadmat(root + '/' + file)
                pose_para = m['Pose_Para'].T.astype(np.float32)
                angles = pose_para[:3, 0]
                angle_list.append(angles)
    return angle_list


# l1 = getAngleList('../data/back/300W-3D/')
# l2 = getAngleList('../data/back/300W_LP/')
# l3 = getAngleList('../data/back/AFLW2000/')
#
# lnp1 = np.array(l1)
# lnp2 = np.array(l2)
# lnp3 = np.array(l3)
lnp1=np.load('l1.npy')
lnp2=np.load('l2.npy')
lnp3=np.load('l3.npy')

for i in range(2000):
    for j in range(3):
        while lnp3[i, j] < -np.pi:
            lnp3[i, j] += (2 * np.pi)
        while lnp3[i, j] > np.pi:
            lnp3[i, j] -= (2 * np.pi)

lx1 = lnp1[:, 0]
lx2 = lnp2[:, 0]
lx3 = lnp3[:, 0]

ly1 = lnp1[:, 1]
ly2 = lnp2[:, 1]
ly3 = lnp3[:, 1]

lz1 = lnp1[:, 2]
lz2 = lnp2[:, 2]
lz3 = lnp3[:, 2]
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

group = [-np.pi + i * np.pi / 40 for i in range(81)]

plt.hist(lx1, group, histtype='bar', rwidth=0.8)
plt.show()
plt.hist(lx2, group, histtype='bar', rwidth=0.8)
plt.show()
plt.hist(lx3, group, histtype='bar', rwidth=0.8)
plt.show()

plt.hist(ly1, group, histtype='bar', rwidth=0.8)
plt.show()
plt.hist(ly2, group, histtype='bar', rwidth=0.8)
plt.show()
plt.hist(ly3, group, histtype='bar', rwidth=0.8)
plt.show()

plt.hist(lz1, group, histtype='bar', rwidth=0.8)
plt.show()
plt.hist(lz2, group, histtype='bar', rwidth=0.8)
plt.show()
plt.hist(lz3, group, histtype='bar', rwidth=0.8)
plt.show()
