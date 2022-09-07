import numpy as np
from scipy import sparse


def node_branch_incidence_matrix(branches, num_node, num_branch, reduced=None):
    incidence_matrix = sparse.lil_matrix((num_node, num_branch), dtype=int)
    for branch_id, (from_node, to_node) in enumerate(branches):
        incidence_matrix[from_node, branch_id] = 1
        incidence_matrix[to_node, branch_id] = -1

    if reduced is not None:
        assert isinstance(reduced, int) and 0 <= reduced < num_node, "`reduced` should be an integer in [0, num_node)."
        incidence_matrix.rows = np.delete(incidence_matrix.rows, reduced)
        incidence_matrix.data = np.delete(incidence_matrix.data, reduced)
        incidence_matrix._shape = (incidence_matrix._shape[0] - 1, incidence_matrix._shape[1])

    return incidence_matrix.tocsr()
