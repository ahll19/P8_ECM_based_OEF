from gurobipy import quicksum
import numpy as np


class BaseDevice(object):
    def __init__(self, params: dict):
        for key, value in params.items():
            self.__setattr__(key, value)

    def add_coupling_cons(self, model):
        pass

    def add_fd2td_cons(self, model):
        pass

    def add_ift_cons(self, model, var_re, var_im, var_th, var_tx):
        if var_th is None:
            var_th = var_tx
        N = len(var_re)
        Th = len(var_th)
        Tx = len(var_tx)
        T = Th + Tx
        for t in range(Th):
            model.addLConstr(
                quicksum(var_re[fi] * np.cos(2 * np.pi * fi * t / T) - var_im[fi] * np.sin(2 * np.pi * fi * t / T)
                         for fi in range(N)) == var_th[t]
            )
        for t in range(Th, T):
            model.addLConstr(
                quicksum(var_re[fi] * np.cos(2 * np.pi * fi * t / T) - var_im[fi] * np.sin(2 * np.pi * fi * t / T)
                         for fi in range(N)) == var_tx[t - Th]
            )

    def get_cost_expr(self):
        return 0

    def get_high_freq_penalty(self, rho=0):
        return 0

    @staticmethod
    def fix_to_zero(variables):
        for var in variables:
            var.setAttr("ub", 0)
            var.setAttr("lb", 0)


if __name__ == '__main__':
    # unit test
    parameters = {"max": 100, "min": 0}
    device = BaseDevice(parameters)
    print(device.max)
    print(device.min)
