import numpy as np
import time
from scipy.spatial import cKDTree

def laplacian_smooth(points, colors, iter=3, l=0.5, k=10):
    t0 = time.time()

    for _ in range(iter):
        new_colors = colors.copy()
        print("Smoothing iteration {0}".format(_))
        tree = cKDTree(points)
        res = {}
        t2 = time.time()
        for point in points:
            d, i = tree.query(point, k=k)
            u = (np.sum(colors[i[1:]] - colors[i[0]]) / k)
            new_colors[i[0]] = colors[i[0]] + l * u
        t3 = time.time()
        print('iteration took {0}s'.format(t3-t2))
    colors = new_colors


    t1 = time.time()
    print("Laplacian smoothing took: {0}s".format(round(t1-t0)))
    return colors