import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d


def graph3dpoints(points, m, b):
    error = np.linspace(0, 0, 200)
    for point in points:
        error = error + (point[1]-(m*point[0]+b))**2
    return error


points = np.array([
    [1, 1],
    [2, 2],
    [3, 4]
])

m = np.linspace(-10, 10, 200)
b = np.linspace(-5, 5, 200)

M, B = np.meshgrid(m, b)
E = graph3dpoints(points, M, B)
# print(np.amin(E)) current minimal error
a, b = np.where(E == np.amin(E))
i = int(a)
j = int(b)
print(M[i][j])  # a more accurate value than that showed in the title
print(B[i][j])  # a more accurate value than that showed in the title
slope = str((M[i][j]))
slope = slope[0:4]
ycut = str(B[i][j])
ycut = ycut[0:5]
titlestr = slope+" and "+ycut
# current best fit line equation: y=M[i][j]*x+B[i][j]
# can be easily plotted by assigning x a linspace of our choice and creating y
# according to the formula just shown
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(M, B, E, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title("error from 200 random lines, m and b: %s" % titlestr)
ax.set_xlabel('m')
ax.set_ylabel('b')
ax.set_zlabel('error')
plt.show()
