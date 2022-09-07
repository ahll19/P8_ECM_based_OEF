from devices.base_device import BaseDevice
from gurobipy import LinExpr, quicksum
import numpy as np
from utils.console_log import warn


class Bus(BaseDevice):
    def __init__(self, id, params):
        # provide attributes: wind, wind_his, load, load_his
        super().__init__(params)

        # more attributes
        self.id = id
        self.has_load = "load" in params
        self.has_wind = "wind" in params
        self.inflow = []
        self.outflow = []
        self.TPUs = []
        self.CHPs = []
        self.gTPUs = []
        self.gCHPs = []
        self.heat_pumps = []

    def add_vars(self, model, num_t):
        # fixed loads
        load = self.load if self.has_load else np.zeros(num_t)
        self.net_injection = [LinExpr(-load[t]) for t in range(num_t)]

        # variable loads
        for heat_pump in self.heat_pumps:
            for t in range(num_t):
                self.net_injection[t] -= heat_pump.power[t]

        # wind generator
        if self.has_wind:
            self.wind_prod = model.addVars(num_t, ub=dict(zip(range(num_t), self.wind)), name=f"bus{self.id}_wind")
            for t in range(num_t):
                self.net_injection[t] += self.wind_prod[t]

        # other generators
        for gen in self.TPUs + self.gTPUs + self.CHPs + self.gCHPs:
            for t in range(num_t):
                self.net_injection[t] += gen.prod[t]

    def get_optimal_wind(self):
        if not self.has_wind:
            return []
        return [wind_prod.x for wind_prod in self.wind_prod.values()]

    def get_bus_injection_curve(self):
        return [net_injection.getValue() for net_injection in self.net_injection]


class TransmissionLine(BaseDevice):
    def __init__(self, id, params, buses):
        # provide attributes: from_node, to_node, x, maxP
        super().__init__(params)

        # more attributes
        self.id = id
        self.is_monitored = "maxP" in params
        if not self.is_monitored:
            self.maxP = float("inf")
        if params.get("consider_contingency", 0) < 0.5:
            self.consider_contingency = False
        else:
            self.consider_contingency = True

        # incidence modeling
        buses[self.from_node].outflow.append(self)
        buses[self.to_node].inflow.append(self)

        self.power_flow_expr = {}
        self.added_cons_pre = set()
        self.added_cons_post = set()

    def set_ptdf(self, ptdf):
        self.ptdf = ptdf

    def set_lodf(self, lodf):
        self.lodf = lodf

    def get_power_flow(self, t, buses=None):
        if t not in self.power_flow_expr:
            ptdf = self.ptdf
            _, cols = ptdf.nonzero()
            vals = ptdf.data
            self.power_flow_expr[t] = quicksum([buses[cols[i]].net_injection[t] * vals[i] for i in range(ptdf.nnz)])

        return self.power_flow_expr[t]

    def add_pre_contingency_transmission_cons(self, model, t, buses):
        if t in self.added_cons_pre:
            warn(f"pre-cont security constraint of line{self.id} at t{t} is added for the second time.")
            return

        self.added_cons_pre.add(t)
        model.addLConstr(self.get_power_flow(t, buses) <= self.maxP)
        model.addLConstr(self.get_power_flow(t) >= -self.maxP)

    def add_post_contingency_transmission_cons(self, model, t, buses, cont_line, lodf, contingency_limit_rate=1.2):
        if (t, cont_line) in self.added_cons_post:
            warn(f"post-cont security constraint of line{self.id} at t{t} is added for the second time.")
            return

        self.added_cons_post.add((t, cont_line))
        power_flow_on_this_line = self.get_power_flow(t, buses)
        power_flow_on_cont_line = cont_line.get_power_flow(t, buses)
        model.addLConstr(
            power_flow_on_this_line + power_flow_on_cont_line * lodf <= self.maxP * contingency_limit_rate
        )
        model.addLConstr(
            -power_flow_on_this_line - power_flow_on_cont_line * lodf <= self.maxP * contingency_limit_rate
        )
