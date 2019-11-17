import numpy as np
import matplotlib.pyplot as plt

#Algorithm for the approximation of a pointwise curve with a cubic spline. 
#The case of acyclic boundary conditions.


points = np.array([[0,1,0],[0.5,0.5,1],[0,-1,2],[-0.5,0.5,1],[0,1,0]], dtype="float")
##points = np.array([[1,10,1],[2,11,3],[4,11,3],[4,5,2],[2,2,5],[7,1,3],
##                   [12,3,4],[13,8,5],[8,7,4],[5,5,4]], dtype="float")    POINTS FOR BEAUTIFUL SWAN

t = [np.sqrt((points[i][0] - points[i-1][0])**2 + (points[i][1] - points[i-1][1])**2
            + (points[i][2] - points[i-1][2])**2) for i in range(1, len(points))]
t.insert(0, 1)
t = np.array(t, dtype="float")
print(t)

M = np.zeros((len(points), len(points)), dtype="float")
M[0,0] = 2*(1 + t[-1]/t[1])
M[0,1] = t[-1]/t[1]
M[0, -2] = -1

M[-1,-1] = -2*(1 + t[-1]/t[1])
M[-1,-2] = -t[-1]/t[1]
M[-1, 1] = 1

for i in range(2, len(t)):
    M[i-1, i-2] = t[i]
    M[i-1, i-1] = 2 * (t[i] + t[i - 1])
    M[i-1, i] = t[i-1]

print("Matrix M:")
print(M)

R = np.array([[0,0,0] for x in range(len(points))], dtype="float")
R[0] = 3*(t[-1]/t[1]**2)*(points[1]-points[0]) + 3/t[-1]*(points[-2] - points[-1])
R[-1] = R[0]


for i in range(2, len(t)):
    k = 3./(t[i-1] * t[i])
    R[i-1] = k * (t[i-1]**2 * (points[i] - points[i-1]) + t[i]**2 * (points[i-1] - points[i-2]))

P1 = np.linalg.inv(M) @ R

def new_points(x, k):
    k = k - 1
    F = np.array([0,0,0,0], dtype="float")
    F[0] = 2*x**3 - 3*x**2 + 1
    F[1] = -2*x**3 + 3*x**2
    F[2] = x*(x**2 - 2*x + 1)*t[k+1]
    F[3] = x*(x**2 - x)*t[k+1]
    G = np.array([points[k], points[k+1], P1[k], P1[k+1]])

    return F @ G

listPoints = np.linspace(0, 1, 100);
colors = ['blue', 'red', 'green', 'magenta']
plt.figure(figsize=(7,7))  
segments = list(range(1, len(points)))
for segment in segments:
    points1 = [new_points(slistPoints, segment) for slistPoints in listPoints]

    x = [pp[0] for pp in points1]
    y = [pp[1] for pp in points1]
    plt.plot(x, y, colors[0])

    
x_given = [pp[0] for pp in points]
y_given = [pp[1] for pp in points]
plt.plot(x_given, y_given, 'ro')
plt.show()


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(7,7))
ax = fig.gca(projection='3d')
 
ax.view_init(elev=30,azim=75)

listPoints = np.linspace(0, 1, 100);
colors = ['blue', 'red', 'green', 'magenta'] 
segments = list(range(1, len(points)))
for segment in segments:
    points1 = [new_points(slistPoints, segment) for slistPoints in listPoints]

    x = [pp[0] for pp in points1]
    y = [pp[1] for pp in points1]
    z = [pp[2] for pp in points1]
    ax.plot(x, y, z, colors[0])
    
    
x_given = [pp[0] for pp in points]
y_given = [pp[1] for pp in points]
z_given = [pp[2] for pp in points]
ax.plot(x_given, y_given,z_given, 'ro')
xx = [x_given[0]]
yy = [y_given[0]]
zz = [z_given[0]]
ax.plot(xx, yy, zz, 'yo')
xx = [x_given[-1]]
yy = [y_given[-1]]
zz = [z_given[-1]]
ax.plot(xx, yy, zz, 'go')

plt.show()
