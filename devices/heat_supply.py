from devices.base_device import BaseDevice
from gurobipy import GRB, quicksum
import numpy as np


class Boiler(BaseDevice):
    def __init__(self, params):
        super().__init__(params)

    def add_vars(self, model, num_t, num_f):
        self.heat = model.addVars(num_t, lb=self.minH, ub=self.maxH, name=f"boiler{self.id}_heat")
        self.heat_re = model.addVars(num_f, lb=-GRB.INFINITY, name=f"boiler{self.id}_heat_re")
        self.heat_im = model.addVars(num_f, lb=-GRB.INFINITY, name=f"boiler{self.id}_heat_im")

    def add_ramp_cons(self, model):
        for t in range(len(self.heat) - 1):
            model.addLConstr(self.heat[t + 1] <= self.heat[t] + self.upRamp)
            model.addLConstr(self.heat[t + 1] >= self.heat[t] - self.downRamp)

    def get_optimal_heat_production(self):
        return [heat.x for heat in self.heat.values()]

    def get_high_freq_penalty(self, rho=0):
        penalty_expr = quicksum(
            [(self.heat_re[i] ** 2 + self.heat_im[i] ** 2) * i ** 2 * rho for i in range(10, len(self.heat_re))]
        )
        return penalty_expr


class HeatPump(Boiler):
    def __init__(self, id, params, buses, pipes):
        # provide attributes: bus, pipe, maxH, minH, upRamp, downRamp, ratio_p2h, his
        super().__init__(params)

        # more attributes
        self.id = id
        buses[self.bus].heat_pumps.append(self)
        pipes[self.pipe].heat_pumps.append(self)

    def add_vars(self, model, num_t, num_f):
        self.heat = model.addVars(num_t, lb=self.minH, ub=self.maxH, name=f"hPump{self.id}_heat")
        self.heat_re = model.addVars(num_f, lb=-GRB.INFINITY, name=f"hPump{self.id}_heat_re")
        self.heat_im = model.addVars(num_f, lb=-GRB.INFINITY, name=f"hPump{self.id}_heat_im")
        self.fix_to_zero([self.heat_im[0]] if num_t & 1 else [self.heat_im[0], self.heat_im[num_f - 1]])
        self.power = model.addVars(num_t, name=f"hPump{self.id}_power")

    def add_coupling_cons(self, model):
        for t in range(len(self.heat)):
            model.addLConstr(self.power[t] - self.heat[t] * self.ratio_p2h == 0)

    def add_fd2td_cons(self, model):
        self.add_ift_cons(model, self.heat_re, self.heat_im, self.his, self.heat)


class GasBoiler(Boiler):
    def __init__(self, id, params, pipes, nodes):
        # provide attributes: node, pipe, maxH, minH, upRamp, downRamp, ratio_h2g, his
        super().__init__(params)

        # more attributes
        self.id = id
        pipes[self.pipe].gas_boilers.append(self)
        nodes[self.node].gas_boilers.append(self)

    def add_vars(self, model, num_t, num_f):
        self.heat = model.addVars(num_t, lb=self.minH, ub=self.maxH, name=f"gBoiler{self.id}_heat")
        self.heat_re = model.addVars(num_f, lb=-GRB.INFINITY, name=f"gBoiler{self.id}_heat_re")
        self.heat_im = model.addVars(num_f, lb=-GRB.INFINITY, name=f"gBoiler{self.id}_heat_im")
        self.gas = model.addVars(num_t, name=f"gBoiler{self.id}_gas")
        self.gas_re = model.addVars(num_f, lb=-GRB.INFINITY, name=f"gBoiler{self.id}_gas_re")
        self.gas_im = model.addVars(num_f, lb=-GRB.INFINITY, name=f"gBoiler{self.id}_gas_im")
        self.fix_to_zero([self.heat_im[0]] if num_t & 1 else [self.heat_im[0], self.heat_im[num_f - 1]])
        self.fix_to_zero([self.gas_im[0]] if num_t & 1 else [self.gas_im[0], self.gas_im[num_f - 1]])

    def add_coupling_cons(self, model):
        for t in range((len(self.heat))):
            model.addLConstr(self.heat[t] - self.gas[t] * self.ratio_h2g == 0)

    def add_fd2td_cons(self, model):
        gas_his = None if self.his is None else (np.asarray(self.his) / self.ratio_h2g)
        self.add_ift_cons(model, self.heat_re, self.heat_im, self.his, self.heat)
        self.add_ift_cons(model, self.gas_re, self.gas_im, gas_his, self.gas)

    def get_optimal_gas_consumption(self):
        return [gas.x for gas in self.gas.values()]
