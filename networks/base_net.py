import numpy as np
from scipy import sparse


class BaseNet(object):
    def __init__(self, nodes, branches, interval, cut_off: int=None):
        self.nodes = nodes
        self.branches = branches
        self.num_node = len(nodes)
        self.num_branch = len(branches)
        self.interval = interval

        for element in (nodes if hasattr(nodes[0], "has_load") else branches):
            if not element.has_load:
                continue

            self.num_tx = len(element.load)
            self.num_th = len(element.load_his)
            break

        assert hasattr(self, "num_tx"), "Load information not found."

        self.th = np.linspace(0, self.num_th - 1, self.num_th) * interval + interval
        self.tx = np.linspace(self.num_th, self.num_th + self.num_tx - 1, self.num_tx) * interval + interval
        self.tl = interval * (self.num_th + self.num_tx)  # time length
        self.fr = 1 / self.tl  # frequency resolution
        if cut_off is None:
            self.num_f = 1 + (self.num_th + self.num_tx) // 2
        else:
            self.num_f = cut_off

    @staticmethod
    def get_Af(A):
        Af = A.copy()
        Af.data = np.where(Af.data < 0, 0, Af.data)
        Af.eliminate_zeros()
        return Af

    @staticmethod
    def get_weighted_At(A, masses):
        w_At = A.copy()
        w_At.data = np.where(w_At.data < 0, 1, 0)
        w_At.eliminate_zeros()

        masses = sparse.dia_matrix(([masses], [0]), shape=(len(masses), len(masses))).tocsr()
        w_At @= masses
        w_At = w_At.astype(float)
        for row in range(w_At.shape[0]):
            w_At[row] /= (w_At[row].sum() + 1e-6)

        return w_At

    def add_fd2td_cons(self, model):
        pass
