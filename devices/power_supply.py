import numpy as np
from devices.base_device import BaseDevice
from gurobipy import GRB, quicksum
from utils.console_log import warn


class Generator(BaseDevice):
    def __init__(self, id, params):
        super().__init__(params)
        self.id = id

    def add_vars(self, model, num_t, num_f=None):
        self.prod = model.addVars(num_t, lb=self.minP, ub=self.maxP, name=f"{self.type}{self.id}_prod")
        if self.is_gas_fired:
            self.gas = model.addVars(num_t, name=f"{self.type}{self.id}_gas")
            self.gas_re = model.addVars(num_f, lb=-GRB.INFINITY, name=f"{self.type}{self.id}_gas_re")
            self.gas_im = model.addVars(num_f, lb=-GRB.INFINITY, name=f"{self.type}{self.id}_gas_im")
            self.fix_to_zero([self.gas_im[0]] if num_t & 1 else [self.gas_im[0], self.gas_im[num_f - 1]])
        if self.produce_heat:
            self.heat = model.addVars(num_t, name=f"{self.type}{self.id}_heat")
            self.heat_re = model.addVars(num_f, lb=-GRB.INFINITY, name=f"{self.type}{self.id}_heat_re")
            self.heat_im = model.addVars(num_f, lb=-GRB.INFINITY, name=f"{self.type}{self.id}_heat_im")
            self.fix_to_zero([self.heat_im[0]] if num_t & 1 else [self.heat_im[0], self.heat_im[num_f - 1]])

    def add_ramp_cons(self, model):
        for t in range(len(self.prod) - 1):
            model.addLConstr(self.prod[t + 1] <= self.prod[t] + self.upRamp)
            model.addLConstr(self.prod[t + 1] >= self.prod[t] - self.downRamp)

    def add_coupling_cons(self, model):
        if self.is_gas_fired:
            for t in range(len(self.prod)):
                model.addLConstr(self.prod[t] - self.gas[t] * self.ratio_p2g == 0)
        if self.produce_heat:
            for t in range(len(self.prod)):
                model.addLConstr(self.prod[t] - self.heat[t] * self.ratio_p2h == 0)

    def add_fd2td_cons(self, model):
        if self.is_gas_fired:
            gas_his = None if self.his is None else (np.asarray(self.his) / self.ratio_p2g)
            self.add_ift_cons(model, self.gas_re, self.gas_im, gas_his, self.gas)
        if self.produce_heat:
            heat_his = None if self.his is None else (np.asarray(self.his) / self.ratio_p2h)
            self.add_ift_cons(model, self.heat_re, self.heat_im, heat_his, self.heat)

    def get_cost_expr(self):
        if self.is_gas_fired:
            return 0

        cost_expr = quicksum(
            self.quad_P * power ** 2 + self.linear_P * power + self.const_P for power in self.prod.values()
        )
        if self.produce_heat:
            cost_expr += quicksum(
                self.quad_H * heat ** 2 + self.linear_H * heat + self.const_H for heat in self.heat.values()
            )
        return cost_expr

    def get_high_freq_penalty(self, rho=0):
        penalty_expr = 0
        if self.produce_heat:
            penalty_expr += quicksum(
                [(self.heat_re[i] ** 2 + self.heat_im[i] ** 2) * i ** 2 * rho for i in range(10, len(self.heat_re))]
            )
        if self.is_gas_fired:
            penalty_expr += quicksum(
                [(self.gas_re[i] ** 2 + self.gas_im[i] ** 2) * i ** 2 * rho for i in range(10, len(self.gas_re))]
            )
        return penalty_expr

    def get_optimal_production(self):
        return [prod.x for prod in self.prod.values()]

    def get_optimal_gas_consumption(self):
        if not self.is_gas_fired:
            warn("Query gas consumption for a coal-fired unit.")
            return []
        return [gas.x for gas in self.gas.values()]

    def get_optimal_heat_production(self):
        if not self.produce_heat:
            warn("Query heat production for a pure-power unit.")
            return []
        return [heat.x for heat in self.heat.values()]


class ThermalPowerUnit(Generator):
    def __init__(self, id, params, buses, nodes=None):
        # provided attributes, see `read.parse_tables.py` -> function `read_generator`
        super().__init__(id, params)

        # more attributes
        if not self.is_gas_fired:
            buses[self.bus].TPUs.append(self)
        else:
            buses[self.bus].gTPUs.append(self)
            nodes[self.node].gTPUs.append(self)


class CombinedHeatPowerUnit(Generator):
    def __init__(self, id, params, buses, pipes, nodes=None):
        # provided attributes, see `read.parse_tables.py` -> function `read_generator`
        super().__init__(id, params)

        # more attributes
        if not self.is_gas_fired:
            buses[self.bus].CHPs.append(self)
            pipes[self.pipe].CHPs.append(self)
        else:
            buses[self.bus].gCHPs.append(self)
            pipes[self.pipe].gCHPs.append(self)
            nodes[self.node].gCHPs.append(self)
