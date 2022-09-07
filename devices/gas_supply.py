from devices.base_device import BaseDevice
from gurobipy import GRB, quicksum


class GasWell(BaseDevice):
    def __init__(self, id, params, nodes):
        # provide attributes: node, maxG, minG, upRamp, downRamp, quad, linear, const, his
        super().__init__(params)

        # more attributes
        self.id = id
        nodes[self.node].gas_wells.append(self)

    def add_vars(self, model, num_t, num_f):
        self.prod = model.addVars(num_t, lb=self.minG, ub=self.maxG, name=f"gWell{self.id}_prod")
        self.prod_re = model.addVars(num_f, lb=-GRB.INFINITY, name=f"gWell{self.id}_prod_re")
        self.prod_im = model.addVars(num_f, lb=-GRB.INFINITY, name=f"gWell{self.id}_prod_im")
        self.fix_to_zero([self.prod_im[0]] if num_t & 1 else [self.prod_im[0], self.prod_im[num_f - 1]])

    def add_ramp_cons(self, model):
        for t in range(len(self.prod) - 1):
            model.addLConstr(self.prod[t + 1] <= self.prod[t] + self.upRamp)
            model.addLConstr(self.prod[t + 1] >= self.prod[t] - self.downRamp)

    def add_fd2td_cons(self, model):
        self.add_ift_cons(model, self.prod_re, self.prod_im, self.his, self.prod)

    def get_cost_expr(self):
        cost_expr = quicksum(
            self.quad * prod ** 2 + self.linear * prod + self.const for prod in self.prod.values()
        )
        return cost_expr

    def get_optimal_production(self):
        return [prod.x for prod in self.prod.values()]

    def get_high_freq_penalty(self, rho=0):
        penalty_expr = quicksum(
            [(self.prod_re[i] ** 2 + self.prod_im[i] ** 2) * 10 * i ** 2 * rho for i in range(10, len(self.prod_re))]
        )
        return penalty_expr
