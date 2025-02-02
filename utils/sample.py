import numpy as np
import pywavefront

def sample_obj(obj, density=625):
    vertices = np.asarray(obj.vertices)
    meshes = [obj.meshes[key] for key in obj.meshes]
    indices = [[idx for face in mesh.faces for idx in face] for mesh in meshes]

    counts = [len(index) for index in indices]
    n_faces3 = np.sum(counts)
    start = 0

    faces = np.zeros((n_faces3, 3))
    for i, mesh in enumerate(meshes):
        temp = vertices[indices[i], :]
        faces[start:start+counts[i], :] = temp
        start += counts[i]

    return sample_mesh(faces, density=density)

"""
faces: (3Nx3), N = # faces
each face defined by triangle (A, B, C)
"""
def sample_mesh(faces, density=625):
    A, B, C = faces[0::3, :], faces[1::3, :], faces[2::3, :]
    cross = np.cross(A[:, 0:3] - C[:, 0:3], B[:, 0:3] - C[:, 0:3])
    areas = 0.5*(np.sqrt(np.sum(cross**2, axis=1)))
    Nsamp = (density*areas).astype(int)
    N = np.sum(Nsamp)
    if N == 0:
        return np.empty((0, 3))
    face_ids = np.zeros((N,), dtype=int)
    count = 0
    for i, n in enumerate(Nsamp):
        face_ids[count:count+n] = i
        count += n
    A, B, C = A[face_ids, :], B[face_ids, :], C[face_ids, :]
    r = np.random.uniform(0, 1, (N, 2))
    sqrt_r1 = np.sqrt(r[:, 0:1])
    samples = (1 - sqrt_r1)*A + sqrt_r1*(1 - r[:, 1:])*B + sqrt_r1*r[:, 1:]*C
    return samples