from utils.timer import timer
from utils.cutoff import cutoff
from utils.console_log import info
from utils.console_log import warn
from networks.base_net import BaseNet
from matrices.incidence_matrix import node_branch_incidence_matrix
from matrices.admittance_matrix import branch_admittance_matrix
from matrices.admittance_matrix import node_admittance_matrix
from matrices.distribution_factor import power_transfer_distribution_factor
from matrices.distribution_factor import line_outage_distribution_factor
from gurobipy import quicksum
import numpy as np


class ElectricityNetwork(BaseNet):
    @timer("initializing the electricity network")
    def __init__(self, nodes, branches, interval=900, cut_off: int=None):
        # common attributes
        super().__init__(nodes, branches, interval, cut_off)

        # specialized attributes
        # A matrix is reduced, in which the last bus is regarded as the reference bus
        self.A = node_branch_incidence_matrix(branches=[(branch.from_node, branch.to_node) for branch in branches],
                                              num_node=len(nodes), num_branch=len(branches), reduced=len(nodes) - 1)
        self.y = branch_admittance_matrix(yb=[1 / branch.x for branch in branches])
        # IMPORTANT: the reference bus is excluded in the following node admittance matrix (non-singularity guaranteed)
        self.Y = node_admittance_matrix(y=self.y, A=self.A)
        self.ptdf = power_transfer_distribution_factor(y=self.y, A=self.A, Y=self.Y)
        self.lodf = line_outage_distribution_factor(A=self.A, ptdf=self.ptdf)

        info(f"remove {cutoff(self.ptdf, 1e-6)}/{self.ptdf.nnz} nonzero elements in PTDF that are less than 1e-6.")
        info(f"remove {cutoff(self.lodf, 1e-6)}/{self.lodf.nnz} nonzero elements in LODF that are less than 1e-6.")
        for line_id, line in enumerate(self.branches):
            line.set_ptdf(self.ptdf.getrow(line_id))
            line.set_lodf(self.lodf.getrow(line_id))

    def add_vars(self, model, implicit=True):
        for node in self.nodes:
            node.add_vars(model, self.num_tx)

    def add_system_balance_cons(self, model):
        # real-time balance
        for t in range(self.num_tx):
            model.addLConstr(
                quicksum(bus.net_injection[t] for bus in self.nodes) == 0
            )

    @timer("electricity network modeling")
    def add_network_cons(self, model, implicit=True):
        # explicit modeling: ptdf (can be converted into lazy-constraint implementation)
        for line in self.branches:
            for t in range(self.num_tx):
                line.add_pre_contingency_transmission_cons(model, t, self.nodes)

        # n-1 contingencies
        # Not Implemented

    def get_available_wind_power(self):
        available_wind = np.zeros(self.num_tx)
        for node in self.nodes:
            if not node.has_wind:
                continue
            available_wind += node.wind
        return available_wind

    def get_accommodated_wind_power(self):
        accommodated_wind = np.zeros(self.num_tx)
        for node in self.nodes:
            if not node.has_wind:
                continue
            accommodated_wind += np.asarray(node.get_optimal_wind())
        return accommodated_wind

    def get_total_load(self):
        total_load = np.zeros(self.num_tx)
        for node in self.nodes:
            if not node.has_load:
                continue
            total_load += node.load
        return total_load

    def get_pre_contingency_line_flow(self):
        # bus injection (2d array): axis-0, bus; axis-1, time.
        bus_injection = np.asarray([bus.get_bus_injection_curve() for bus in self.nodes[:-1]])

        # line flow (2d array): axis-0, line; axis-1, time.
        pre_contingency_line_flow = self.ptdf @ bus_injection

        return pre_contingency_line_flow  # (line_id, t)

    def get_post_contingency_line_flow(self, pre_contingency_line_flow, contingency_line_id):
        raise NotImplementedError()

    def security_check(self, reserved_each_t=3, epsilon=1e-3, contingency_limit_rate=1.2):
        pre_contingency_line_flow = self.get_pre_contingency_line_flow()
        line_thermal_limits = np.asarray([line.maxP for line in self.branches])[:, np.newaxis] + epsilon

        violations = [[] for _ in range(pre_contingency_line_flow.shape[1])]
        num_normal_violations = num_cont_violations = 0

        # pre-contingency power flow check
        for line_id, t in zip(*np.where(abs(pre_contingency_line_flow) > line_thermal_limits)):
            num_normal_violations += 1
            excess = abs(pre_contingency_line_flow[line_id, t]) - line_thermal_limits[line_id, 0]
            violations[t].append((t, line_id, None, excess))

        # post-contingency power flow check
        # NotImplemented

        if num_normal_violations + num_cont_violations:
            warn(f"{num_normal_violations} pre-contingency and {num_cont_violations} post-contingency" +
                 f" overflow detected in the electricity network.")
        else:
            info(f"no overflow detected in the electricity network.")

        filtered = []
        for violations_at_t in violations:
            violations_at_t.sort(key=lambda x: x[3], reverse=True)
            filtered.extend(violations_at_t[:reserved_each_t])
        return filtered
