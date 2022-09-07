import numpy as np
from utils.ft import td2fd
from devices.base_device import BaseDevice
from gurobipy import LinExpr, GRB, quicksum
from utils.console_log import warn


class HeatNode(BaseDevice):
    def __init__(self, id, params):
        # provide no attributes
        super().__init__(params)

        # more attributes
        self.id = id
        self.inflow = []
        self.outflow = []


class HeatPipe(BaseDevice):
    def __init__(self, id, params, nodes):
        # provide attributes: from_node, to_node, length, mass, velocity, diameter, area, dissipation,
        #                     type, maxT, minT, load, load_his
        super().__init__(params)

        # more attributes
        self.id = id
        self.heat_capacity = 4.2e3
        self.rho = 1e3
        self.is_monitored_upper = "maxT" in params
        self.is_monitored_lower = "minT" in params
        if not self.is_monitored_upper:
            self.maxT = float("inf")
        if not self.is_monitored_lower:
            self.minT = 0
        self.has_load = "load" in params
        self.CHPs = []
        self.gCHPs = []
        self.heat_pumps = []
        self.gas_boilers = []
        if self.has_load:
            self.load_fd = td2fd(np.hstack((self.load_his, self.load)))
        self.end_temperature_expr = {}  # key: time
        self.prod_ZI = {}  # key: frequency
        self.added_ub_cons = set()
        self.added_lb_cons = set()

        # incidence modeling
        nodes[self.from_node].outflow.append(self)
        nodes[self.to_node].inflow.append(self)

        # time-domain circuit parameters
        self.R = self.dissipation / self.heat_capacity ** 2 / (abs(self.mass) + 1e-10) ** 2
        self.L = self.rho * self.area / self.heat_capacity / (abs(self.mass) + 1e-10) ** 2

    def add_vars(self, model, num_t, num_f, implicit=True):
        self.implicit = implicit
        # fixed loads
        load = self.load if self.has_load else np.zeros(num_t)
        self.net_injection = [LinExpr(-load[t]) for t in range(num_t)]
        load_fd = self.load_fd if self.has_load else np.zeros(num_f, dtype=complex)
        self.net_injection_re = [LinExpr(-load_fd[fi].real) for fi in range(num_f)]
        self.net_injection_im = [LinExpr(-load_fd[fi].imag) for fi in range(num_f)]

        # heat production
        for gen in self.CHPs + self.gCHPs + self.heat_pumps + self.gas_boilers:
            for t in range(num_t):
                self.net_injection[t] += gen.heat[t]
            for fi in range(num_f):
                self.net_injection_re[fi] += gen.heat_re[fi]
                self.net_injection_im[fi] += gen.heat_im[fi]

        # branch temperature at "to" side
        if implicit:
            self.Tt = model.addVars(num_t, lb=self.minT, ub=self.maxT, name=f"node{self.id}_Tt")
            self.Tt_re = model.addVars(num_f, lb=-GRB.INFINITY, name=f"node{self.id}_Tt_re")
            self.Tt_im = model.addVars(num_f, lb=-GRB.INFINITY, name=f"node{self.id}_Tt_im")

    def add_ift_cons(self, model, var_re, var_im, var_tx, Th):
        N = len(var_re)
        Tx = len(var_tx)
        T = Th + Tx
        for t in range(Th, T):
            model.addLConstr(
                quicksum(var_re[fi] * np.cos(2 * np.pi * fi * t / T) - var_im[fi] * np.sin(2 * np.pi * fi * t / T)
                         for fi in range(N)) == var_tx[t - Th]
            )

    def get_incidence(self):
        return (self.from_node, self.to_node) if self.mass >= 0 else (self.to_node, self.from_node)

    def get_optimal_temperature(self, hNet=None):
        # I recommend using this method only when the implicit modeling is adopted;
        # otherwise, use `HeatingNetwork.get_temperature_curves()` instead.
        if self.implicit:
            curve = [T.x for T in self.Tt.values()]
        else:
            curve = []
            for t in range(hNet.num_th, hNet.num_th + hNet.num_tx):
                curve.append(
                    self.get_pipe_end_temperature(t, hNet.IDFT[t], hNet.Zs, hNet.fd_pipe_injection).getValue()
                )

        return curve

    def get_complex_pipe_injection(self):
        injection_re = np.asarray([net_injection_re.getValue() for net_injection_re in self.net_injection_re])
        injection_im = np.asarray([net_injection_im.getValue() for net_injection_im in self.net_injection_im])
        return injection_re + complex(0, 1) * injection_im

    def add_temperature_upper_bound_cons(self, model, t, IDFT, Zs, fd_injections):
        if t in self.added_ub_cons:
            warn(f"security constraint (ub) of pipe{self.id} at t{t} is added for the second time.")
            return

        self.added_ub_cons.add(t)
        model.addLConstr(self.get_pipe_end_temperature(t, IDFT, Zs, fd_injections) <= self.maxT)

    def add_temperature_lower_bound_cons(self, model, t, IDFT, Zs, fd_injections):
        if t in self.added_lb_cons:
            warn(f"security constraint (lb) of pipe{self.id} at t{t} is added for the second time.")
            return

        self.added_lb_cons.add(t)
        model.addLConstr(self.get_pipe_end_temperature(t, IDFT, Zs, fd_injections) >= self.minT)

    def get_pipe_end_temperature(self, t, IDFT=None, Zs=None, fd_injections=None):
        if t not in self.end_temperature_expr:
            temperature_expr = LinExpr()

            for fi, Z in enumerate(Zs):
                if fi not in self.prod_ZI:
                    Zre_row = Z[self.id].real
                    Zim_row = Z[self.id].imag
                    Ire_col, Iim_col = fd_injections[fi]

                    prod_re = quicksum(
                        ((Zre_row[i] * Ire_col[i]) if abs(Zre_row[i]) > 1e-6 else 0) -
                        ((Zim_row[i] * Iim_col[i]) if abs(Zim_row[i]) > 1e-6 else 0)
                        for i in range(len(Zre_row))
                    )
                    prod_im = quicksum(
                        ((Zre_row[i] * Iim_col[i]) if abs(Zre_row[i]) > 1e-6 else 0) +
                        ((Zim_row[i] * Ire_col[i]) if abs(Zim_row[i]) > 1e-6 else 0)
                        for i in range(len(Zre_row))
                    )

                    self.prod_ZI[fi] = (prod_re, prod_im)
                else:
                    prod_re, prod_im = self.prod_ZI[fi]

                temperature_expr += IDFT[fi].real * prod_re - IDFT[fi].imag * prod_im

            self.end_temperature_expr[t] = temperature_expr

        return self.end_temperature_expr[t]
