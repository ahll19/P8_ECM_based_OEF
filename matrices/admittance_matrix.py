from scipy import sparse


def branch_admittance_matrix(yb: list):
    return sparse.dia_matrix(([yb], [0]), shape=(len(yb), len(yb))).tocsr()


def node_admittance_matrix(y, A):
    return A @ y @ A.T  # csr format


def generalized_node_admittance_matrix(y, u, A, Af):
    return A @ y @ A.T - A @ y @ u @ Af.T  # csr format
