import numpy as np

# CHECKERBOARD = (6, 9)
# objectp3d = np.zeros((1, CHECKERBOARD[0]
#                       * CHECKERBOARD[1],
#                       3), np.float32)
# objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# print(objectp3d)


def getWorldCorners(edge, n_X, n_Y):
    x_vals = [edge*int(val) for val in np.linspace(0, n_X-1, n_X)]
    y_vals = [edge*int(val) for val in np.linspace(0, n_Y-1, n_Y)]
    w_corners = list()
    for yval in y_vals:
        for xval in x_vals:
            w_corners.append((xval, yval))    
    w_corners = np.stack(w_corners)
    return w_corners


A = np.array([0, 122, 333])
print(A[[2, 1]])