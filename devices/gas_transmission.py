import numpy as np
from utils.ft import td2fd
from devices.base_device import BaseDevice
from gurobipy import LinExpr, GRB, quicksum
from utils.console_log import warn


class GasNode(BaseDevice):
    def __init__(self, id, params):
        # provide attributes: pMax, pMin, load, load_his
        super().__init__(params)

        # more attributes
        self.id = id
        self.is_monitored_upper = "pMax" in params
        self.is_monitored_lower = "pMin" in params
        if not self.is_monitored_upper:
            self.pMax = float("inf")
        if not self.is_monitored_lower:
            self.pMin = 0
        self.has_load = "load" in params
        self.inflow = []
        self.outflow = []
        self.gTPUs = []
        self.gCHPs = []
        self.gas_boilers = []
        self.gas_wells = []
        self.pressure_expr = {}
        self.prod_ZI = {}
        self.added_ub_cons = set()
        self.added_lb_cons = set()

        if self.has_load:
            self.load_fd = td2fd(np.hstack((self.load_his, self.load)))

    def add_vars(self, model, num_t, num_f, implicit=True):
        self.implicit = implicit
        # fixed loads
        load = self.load if self.has_load else np.zeros(num_t)
        self.net_injection = [LinExpr(-load[t]) for t in range(num_t)]
        load_fd = self.load_fd if self.has_load else np.zeros(num_f, dtype=complex)
        self.net_injection_re = [LinExpr(-load_fd[fi].real) for fi in range(num_f)]
        self.net_injection_im = [LinExpr(-load_fd[fi].imag) for fi in range(num_f)]

        # variable loads
        for gen in self.gTPUs + self.gCHPs + self.gas_boilers:
            for t in range(num_t):
                self.net_injection[t] -= gen.gas[t]
            for fi in range(num_f):
                self.net_injection_re[fi] -= gen.gas_re[fi]
                self.net_injection_im[fi] -= gen.gas_im[fi]

        # gas production
        for well in self.gas_wells:
            for t in range(num_t):
                self.net_injection[t] += well.prod[t]
            for fi in range(num_f):
                self.net_injection_re[fi] += well.prod_re[fi]
                self.net_injection_im[fi] += well.prod_im[fi]

        # node pressure
        if implicit:
            self.pressure = model.addVars(num_t, lb=self.pMin, ub=self.pMax, name=f"node{self.id}_pressure")
            self.pressure_re = model.addVars(num_f, lb=-GRB.INFINITY, name=f"node{self.id}_pressure_re")
            self.pressure_im = model.addVars(num_f, lb=-GRB.INFINITY, name=f"node{self.id}_pressure_im")

    def add_ift_cons(self, model, var_re, var_im, var_tx, Th):
        N = len(var_re)
        Tx = len(var_tx)
        T = Th + Tx
        for t in range(Th, T):
            model.addLConstr(
                quicksum(var_re[fi] * np.cos(2 * np.pi * fi * t / T) - var_im[fi] * np.sin(2 * np.pi * fi * t / T)
                         for fi in range(N)) == var_tx[t - Th]
            )

    def get_optimal_pressure(self, gNet=None):
        # I recommend using this method only when the implicit modeling is adopted;
        # otherwise, use `GasNetwork.get_pressure_curves()` instead.
        if self.implicit:
            curve = [pressure.x for pressure in self.pressure.values()]
        else:
            curve = []
            for t in range(gNet.num_th, gNet.num_th + gNet.num_tx):
                curve.append(
                    self.get_node_pressure(t, gNet.IDFT[t], gNet.Zs, gNet.fd_node_injection,
                                           gNet.pressure_reference_base).getValue()
                )

        return curve

    def get_complex_node_injection(self):
        injection_re = np.asarray([net_injection_re.getValue() for net_injection_re in self.net_injection_re])
        injection_im = np.asarray([net_injection_im.getValue() for net_injection_im in self.net_injection_im])
        return injection_re + complex(0, 1) * injection_im

    def get_complex_node_pressure(self):
        pressure_re = np.asarray([pressure_re.x for pressure_re in self.pressure_re.values()])
        pressure_im = np.asarray([pressure_im.x for pressure_im in self.pressure_im.values()])
        return pressure_re + complex(0, 1) * pressure_im

    def add_pressure_upper_bound_cons(self, model, t, IDFT, Zs, fd_injections, pressure_refer_base,
                                      cont_line_id=None, delta_p=None):
        if (t, cont_line_id) in self.added_ub_cons:
            warn(f"gas net: security constraint (ub) of node-{self.id} at t-{t} after line-{cont_line_id} " +
                 f"outage is added for the second time.")
            return

        self.added_ub_cons.add((t, cont_line_id))
        if cont_line_id is None:
            model.addLConstr(self.get_node_pressure(t, IDFT, Zs, fd_injections, pressure_refer_base) <= self.pMax)
        else:
            # Not Implemented
            pass

    def add_pressure_lower_bound_cons(self, model, t, IDFT, Zs, fd_injections, pressure_refer_base,
                                      cont_line_id=None, delta_p=None):
        if (t, cont_line_id) in self.added_lb_cons:
            warn(f"gas net: security constraint (lb) of node-{self.id} at t-{t} after line-{cont_line_id} " +
                 f"outage is added for the second time.")
            return

        self.added_lb_cons.add((t, cont_line_id))
        if cont_line_id is None:
            model.addLConstr(self.get_node_pressure(t, IDFT, Zs, fd_injections, pressure_refer_base) >= self.pMin)
        else:
            # Not Implemented
            pass

    def get_node_pressure(self, t, IDFT=None, Zs=None, fd_injections=None, pressure_refer_base=None):
        if t not in self.pressure_expr:
            pressure_expr = LinExpr()

            for fi, Z in enumerate(Zs):
                if fi in self.prod_ZI:
                    prod_re, prod_im = self.prod_ZI[fi]
                else:
                    if fi or self.id < Z.shape[0]:
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
                    else:  # this is the last node
                        prod_re = pressure_refer_base
                        prod_im = 0

                    self.prod_ZI[fi] = (prod_re, prod_im)

                pressure_expr += prod_re * IDFT[fi].real - prod_im * IDFT[fi].imag

            self.pressure_expr[t] = pressure_expr

        return self.pressure_expr[t]

    def get_node_pressure_f0(self, Z0, fd_injection0, pressure_refer_base=None):
        if self.id == Z0.shape[0]:
            return pressure_refer_base

        if 0 not in self.prod_ZI:
            Zre_row = Z0[self.id].real
            Ire_col, _ = fd_injection0

            prod_re = quicksum(
                ((Zre_row[i] * Ire_col[i]) if abs(Zre_row[i]) > 1e-6 else 0)
                for i in range(len(Zre_row))
            )

            self.prod_ZI[0] = (prod_re, 0)

        return self.prod_ZI[0][0]  # return only the real part


class GasPipe(BaseDevice):
    def __init__(self, id, params, nodes, sonic_speed=350):
        # provide attributes: from_node, to_node, length, diameter, area, friction, vBase
        super().__init__(params)
        if params.get("consider_contingency", 0) < 0.5:
            self.consider_contingency = False
        else:
            self.consider_contingency = True

        # more attributes
        self.id = id
        self.sonic_speed = sonic_speed

        # incidence modeling
        nodes[self.from_node].outflow.append(self)
        nodes[self.to_node].inflow.append(self)

        # time-domain circuit parameters
        self.L = 1 / self.area  # inductance
        self.C = self.area / self.sonic_speed ** 2  # capacitance
        self.R = self.friction * self.vBase / self.area / self.diameter  # resistance
        self.U = -self.friction * self.vBase ** 2 / 2 / self.sonic_speed ** 2 / self.diameter  # controlled source
