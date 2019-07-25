import numpy as np
from sklearn.neighbors import NearestNeighbors

"""
Iterative Closest Point
A, B: N1xm, N2xm
Return final transformation, distances to nearest neighbor, and iterations till convergence
"""
def icp(A, B, init_pose=None, max_iters=20, tolerance=0.001):

    assert A.shape == B.shape
    N1, m, N2, _ = *A.shape, *B.shape
    src, dst = np.ones((m+1, N1)), np.ones((m+1, N2))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0
    for i in range(max_iters):
        distances, indices = nearest_neighbor(src[:m, :].T, dst[:m, :].T)
        T = lsbf_transform(src[:m, :].T, dst[:m, indices].T)
        src = np.dot(T, src)

        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    T = lsbf_transform(A, src[:m, :].T)
    return T, distances, i

"""
Least Squares best fit transform mapping A to B
"""
def lsbf_transform(A, B):

    assert A.shape[1] == B.shape[1]
    N, m = A.shape

    centroid_A, centroid_B = np.mean(A, axis=0), np.mean(B, axis=0)
    AA, BB = A - centroid_A, B - centroid_B

    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    if np.linalg.det(R) < 0:
        Vt[m-1, :] *= -1
        R = np.dot(Vt.t, U.T)

    t = centroid_B.T - np.dot(R, centroid_A.T)
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t
    return T

def nearest_neighbor(src, dst):
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()