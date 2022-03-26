import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

v_ilue = np.array([0.12 ,0 ,0.96])
r = 80

pad = int((r + r//5) * 2)
margin  = pad // 2 - r

image_pad = np.zeros((pad + 1, pad + 1))


def cal_norm(x, y, r):
    if r < (x ** 2 + y ** 2) ** (1/2):
        raise ValueError("norm(x, y) over the radiatin")
    z = (r ** 2 - x ** 2 - y ** 2) ** (1/2)
    if z == 0:
        return np.array([x, y, 1]) / (x ** 2 + y ** 2) ** (1/2)
    else:
        return np.array([x/z, y/z, 1]) / ((x/z) ** 2 + (y/z) ** 2 + 1) ** (1/2)
    

def cal_energy(v_norm, v_ilue):
    energy = np.dot(v_norm, v_ilue)
    if energy < 0:
        energy = 0
    
    return energy


for _i in range(-r, r, 1):
    for _j in range(-r, r, 1):
        if r >= (_i ** 2 + _j ** 2) ** (1/2):
            v_norm = cal_norm(_i, _j, r)
            energy = cal_energy(v_norm, v_ilue)
            image_pad[margin + r + _j, margin + r + _i] = energy

plt.imshow(image_pad, cmap='gray')
plt.show()