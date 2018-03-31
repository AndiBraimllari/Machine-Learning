import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d


def graph3dpoints(points, m, b):
    E = np.linspace(0, 0, 25)
    for point in points:
       E = E + (point[1]-(m*point[0]+b))**2
    return E


P = np.array([
    [1, 1],
    [2, 2],
    [3, 4]
])

m = np.linspace(-10, 10, 25)
b = np.linspace(-5, 5, 25)

M, B = np.meshgrid(m, b)
E = graph3dpoints(P, M, B)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(M, B, E, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title("error of random lines")
ax.set_xlabel('m')
ax.set_ylabel('b')
ax.set_zlabel('error')
plt.show()
