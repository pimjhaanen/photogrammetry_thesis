import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

def estimate_shape(points):
    points = np.array(points)
    tck, _ = splprep([points[:,0], points[:,1], points[:,2]], s=2)
    u_fine = np.linspace(0, 1, 100)
    x_f, y_f, z_f = splev(u_fine, tck)
    return np.vstack([x_f, y_f, z_f]).T

def plot_shape(points, fitted):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*np.array(points).T, label='Markers')
    ax.plot(*fitted.T, label='Spline Fit')
    ax.legend()
    plt.show()
