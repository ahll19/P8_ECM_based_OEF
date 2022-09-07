import numpy as np


def cutoff(mat, threshold=1e-6):
    # `mat` is a sparse matrix, and the nonzero entries in it with values less than `threshold` will removed in place.
    init_nnz = mat.nnz
    mat.data = np.where(abs(mat.data) < threshold, 0, mat.data)

    mat.eliminate_zeros()
    return init_nnz - mat.nnz
