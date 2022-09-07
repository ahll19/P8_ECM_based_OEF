from scipy.sparse.linalg import inv
from utils.timer import timer
from scipy import sparse


@timer("forming PTDF matrix")
def power_transfer_distribution_factor(y, A, Y):
    return y @ A.T @ inv(Y.tocsc())


@timer("forming LODF matrix")
def line_outage_distribution_factor(A, ptdf, epsilon=1e-6):
    lodf = ptdf @ A

    multiplier = []
    for col in range(lodf.shape[1]):
        if abs(1 - lodf[col, col]) > epsilon:
            multiplier.append(1 / (1 - lodf[col, col]))
        else:
            multiplier.append(1e10)
    multiplier = sparse.dia_matrix(([multiplier], [0]), shape=(len(multiplier), len(multiplier))).tocsr()
    lodf = lodf @ multiplier
    for col in range(lodf.shape[1]):
        lodf[col, col] = -1

    return lodf  # csr format
