import numpy as np
from utils.timer import timer
from networks.base_net import BaseNet
from matrices.incidence_matrix import node_branch_incidence_matrix
from matrices.admittance_matrix import branch_admittance_matrix
from matrices.admittance_matrix import generalized_node_admittance_matrix
from gurobipy import quicksum, GRB
from scipy.sparse.linalg import inv
from utils.ft import fd2td, get_IDFT_matrix
from utils.console_log import info, warn


class GasNetwork(BaseNet):
    @timer("initializing the gas network")
    def __init__(self, nodes, branches, interval=900):
        def adaptive_inv(mat):
            if mat.shape[0] > 1000:
                return inv(mat.tocsc()).A
            else:
                return np.linalg.inv(mat.A)  # quicker for small-scale Y

        # common attributes
        super().__init__(nodes, branches, interval)

        # specialized attributes
        self.num_augmented_node = self.num_node + 1  # one more `ground node`
        self.num_augmented_branch = self.num_branch * 3  # two times more `shunt branches`
        self.A = node_branch_incidence_matrix(branches=self.get_augmented_incidence(self.branches, self.num_node),
                                              num_node=self.num_augmented_node, num_branch=self.num_augmented_branch,
                                              reduced=self.num_node)
        self.Af = self.get_Af(self.A)
        # IMPORTANT: the ground node is excluded in the following node admittance matrix (non-singularity guaranteed
        # for f != 0)
        self.Ys = self.get_Ys()
        self.Zs = []
        for fi, Y in enumerate(self.Ys):
            if not fi:
                self.Zs.append(adaptive_inv(Y[:-1, :-1]))
            else:
                self.Zs.append(adaptive_inv(Y))

        self.pressure_reference_base = None
        self.implicit_model = False
        self.explicit_model = False
        self.branch_flow = {}

    def get_posf(self):
        # sensitivity factors from (branch flow on outage pipeline) to (node pressure)
        raise NotImplementedError()

    @staticmethod
    def get_augmented_incidence(pipes, num_node):
        augmented_branches = []
        for pipe in pipes:
            augmented_branches.append((pipe.from_node, pipe.to_node))
            augmented_branches.append((pipe.from_node, num_node))
            augmented_branches.append((pipe.to_node, num_node))
        return augmented_branches

    @staticmethod
    def get_augmented_branch_freq_domain_lumped_params(pipes, num_f, fr, need_Z_and_U=False):
        # time-domain distributed parameters
        Rs = np.asarray([pipe.R for pipe in pipes])
        Ls = np.asarray([pipe.L for pipe in pipes])
        Cs = np.asarray([pipe.C for pipe in pipes])
        Us = np.asarray([pipe.U for pipe in pipes])
        Lens = np.asarray([pipe.length for pipe in pipes])

        j = complex(0, 1)
        ybs = []
        ubs = []
        Zbs = []
        Ubs = []
        for fi in range(num_f):
            f = fi * fr

            # frequency-domain distributed parameters
            Zs = Rs + j * 2 * np.pi * f * Ls
            Ys = j * 2 * np.pi * f * Cs

            # frequency-domain lumped parameters
            za = (np.cosh(np.sqrt(Us ** 2 + 4 * Zs * Ys) / 2 * Lens) - Us / np.sqrt(Us ** 2 + 4 * Zs * Ys) * np.sinh(
                np.sqrt(Us ** 2 + 4 * Zs * Ys) / 2 * Lens)) * np.exp(-Us * Lens / 2)
            zb = -2 * Zs / np.sqrt(Us ** 2 + 4 * Zs * Ys) * np.sinh(np.sqrt(Us ** 2 + 4 * Zs * Ys) / 2 * Lens) * np.exp(
                -Us * Lens / 2)
            zc = -2 * Ys / np.sqrt(Us ** 2 + 4 * Zs * Ys) * np.sinh(np.sqrt(Us ** 2 + 4 * Zs * Ys) / 2 * Lens) * np.exp(
                -Us * Lens / 2)
            zd = (np.cosh(np.sqrt(Us ** 2 + 4 * Zs * Ys) / 2 * Lens) + Us / np.sqrt(Us ** 2 + 4 * Zs * Ys) * np.sinh(
                np.sqrt(Us ** 2 + 4 * Zs * Ys) / 2 * Lens)) * np.exp(-Us * Lens / 2)
            Yb1 = (za * zd - zb * zc - za) / zb
            Yb2 = (1 - zd) / zb
            Zb = -zb
            Ub = 1 - za * zd + zb * zc

            # augment
            yb = []
            ub = []
            for i in range(len(pipes)):
                yb.extend([1 / Zb[i], Yb1[i], Yb2[i]])
                ub.extend([Ub[i], 0, 0])

            ybs.append(yb)  # list[list[complex]]
            ubs.append(ub)
            Zbs.append(Zb)  # list[1d array[complex]]
            Ubs.append(Ub)

        if not need_Z_and_U:
            return ybs, ubs
        else:
            return ybs, ubs, Zbs, Ubs

    def get_Ys(self):
        ybs, ubs = self.get_augmented_branch_freq_domain_lumped_params(self.branches, self.num_f, self.fr)
        ys = [branch_admittance_matrix(yb) for yb in ybs]
        us = [branch_admittance_matrix(ub) for ub in ubs]
        Ys = [generalized_node_admittance_matrix(ys[i], us[i], self.A, self.Af) for i in range(self.num_f)]
        for Y in Ys:
            Y *= 1e6  # thereby the pressure unit is in `MPa`
        return Ys

    def add_vars(self, model, implicit=True):
        for node in self.nodes:
            node.add_vars(model, self.num_tx, self.num_f, implicit)

    def add_system_balance_cons(self, model):
        # intraday balance
        model.addLConstr(
            quicksum(node.net_injection[t] for node in self.nodes for t in range(self.num_tx)) >= 0
        )

    @timer("gas network uec modeling")
    def add_network_cons(self, model, implicit=True):
        if not implicit:
            # explicit modeling: ZI=U
            self.pressure_reference_base = model.addVar(lb=-GRB.INFINITY)
            self.fd_node_injection = [self.get_node_injection(fi) for fi in range(self.num_f)]
            self.IDFT = get_IDFT_matrix(self.num_th + self.num_tx)
            # remark: `self.pressure_reference_base` is zero-frequency component of last node's pressure
            for node in self.nodes:
                for t in range(self.num_th, self.num_th + self.num_tx):
                    node.add_pressure_upper_bound_cons(model, t, self.IDFT[t], self.Zs, self.fd_node_injection,
                                                       self.pressure_reference_base)
                    node.add_pressure_lower_bound_cons(model, t, self.IDFT[t], self.Zs, self.fd_node_injection,
                                                       self.pressure_reference_base)
            self.explicit_model = True
            return

        # implicit modeling: YU=I
        nodes = self.nodes
        for fi in range(self.num_f):
            Y = self.Ys[fi]
            Y_re = Y.real
            Y_im = Y.imag
            for nd1 in range(self.num_node):  # exclude the ground node
                model.addLConstr(
                    quicksum(Y_re[nd1, nd2] * nodes[nd2].pressure_re[fi] - Y_im[nd1, nd2] * nodes[nd2].pressure_im[fi]
                             for nd2 in Y.getrow(nd1).nonzero()[1]) == nodes[nd1].net_injection_re[fi]
                )
                model.addLConstr(
                    quicksum(Y_re[nd1, nd2] * nodes[nd2].pressure_im[fi] + Y_im[nd1, nd2] * nodes[nd2].pressure_re[fi]
                             for nd2 in Y.getrow(nd1).nonzero()[1]) == nodes[nd1].net_injection_im[fi]
                )
        self.implicit_model = True

    def add_fd2td_cons(self, model):
        for node in self.nodes:
            node.add_ift_cons(model, node.pressure_re, node.pressure_im, node.pressure, self.num_th)

    def security_check(self, reserved_each_t=3, epsilon=1e-3):
        # pre-contingency pressure
        pre_cont_pressure = self.get_pressure_curves()
        upper_bound = np.asarray([node.pMax for node in self.nodes])[:, np.newaxis] + epsilon
        lower_bound = np.asarray([node.pMin for node in self.nodes])[:, np.newaxis] - epsilon

        num_tx = pre_cont_pressure.shape[1]
        violations_over_pressure = [[] for _ in range(num_tx)]
        num_total_over_pressure = 0
        for node_id, t in zip(*np.where(pre_cont_pressure > upper_bound)):
            num_total_over_pressure += 1
            excess = pre_cont_pressure[node_id, t] - upper_bound[node_id, 0]
            violations_over_pressure[t].append((t, node_id, None, excess))

        violations_under_pressure = [[] for _ in range(num_tx)]
        num_total_under_pressure = 0
        for node_id, t in zip(*np.where(pre_cont_pressure < lower_bound)):
            num_total_under_pressure += 1
            excess = lower_bound[node_id, 0] - pre_cont_pressure[node_id, t]
            violations_under_pressure[t].append((t, node_id, None, excess))

        # post-contingency pressure
        # NotImplemented

        if num_total_over_pressure or num_total_under_pressure:
            warn(f"{num_total_over_pressure} over pressure and {num_total_under_pressure} under pressure detected " +
                 f"in the natural gas network.")
        else:
            info("no over pressure or under pressure detected in the natural gas network.")

        filtered_over_pressure = []
        filtered_under_pressure = []
        if num_total_over_pressure <= num_tx / 2 and num_total_under_pressure <= num_tx / 2:
            reserved_each_t *= 2
        for violations_at_t in violations_over_pressure:
            filtered_over_pressure.extend(sorted(violations_at_t, key=lambda x: x[3], reverse=True)[:reserved_each_t])
        for violations_at_t in violations_under_pressure:
            filtered_under_pressure.extend(sorted(violations_at_t, key=lambda x: x[3], reverse=True)[:reserved_each_t])

        return filtered_over_pressure, filtered_under_pressure

    def get_delta_pressure_by_cont_line(self, node_id, cont_line_id):
        if cont_line_id is None:
            return 0
        else:
            return NotImplementedError()

    def get_node_injection(self, fi):
        nodes = self.nodes

        if fi == 0:
            # note injection (f=0) here is modified.
            injection_re = [nodes[nd].net_injection_re[fi] - self.Ys[0][nd, -1].real * self.pressure_reference_base
                            for nd in range(self.num_node)]
            injection_im = [0 for _ in range(self.num_node)]
        else:
            injection_re = [nodes[i].net_injection_re[fi] for i in range(self.num_node)]
            injection_im = [nodes[i].net_injection_im[fi] for i in range(self.num_node)]

        return injection_re, injection_im

    def get_pressure_curves(self):
        node_injection = np.asarray([node.get_complex_node_injection() for node in self.nodes])

        """
        # verification: passed.
        node_pressure = np.asarray([node.get_complex_node_pressure() for node in self.nodes])

        # known: node_injection, and we use node_pressure[-1, 0] as reference base
        # unknown & ground truth: node_pressure

        recover_pressure = np.zeros_like(node_pressure)
        for fi, Y in enumerate(self.Ys):
            if fi:
                recover_pressure[:, [fi]] = inv(Y.tocsc()) @ node_injection[:, [fi]]
            else:
                recover_pressure[-1, 0] = node_pressure[-1, 0]
                recover_pressure[:-1, [0]] = inv(Y[:-1, :-1].tocsc()) @ (
                        node_injection[:-1, [0]] - Y[:-1, [-1]] @ recover_pressure[[-1], 0][:, np.newaxis])
                assert abs(Y[[-1], :] @ recover_pressure[:, [0]] - node_injection[-1, 0]).max() < 1e-10

        assert abs(recover_pressure - node_pressure).max() < 1e-10
        """

        if self.pressure_reference_base is not None:
            reference_base = self.pressure_reference_base.x
        else:
            reference_base = self.nodes[-1].pressure_re[0].x

        recover_pressure = np.zeros((self.num_node, self.num_f), dtype=complex)
        for fi, Z in enumerate(self.Zs):
            if fi:
                recover_pressure[:, [fi]] = Z @ node_injection[:, [fi]]
            else:
                recover_pressure[-1, 0] = reference_base
                recover_pressure[:-1, [0]] = Z @ (node_injection[:-1, [0]] -
                                                  self.Ys[0][:-1, [-1]] @ recover_pressure[[-1], 0][:, np.newaxis])

        recover_pressure = fd2td(recover_pressure)[:, -self.num_tx:]

        return recover_pressure

    def get_pipe_flow_curves(self, return_base_value=False):
        # note: the branch flow in this method is the average value
        node_injection_fd = np.asarray([node.get_complex_node_injection() for node in self.nodes])

        if self.pressure_reference_base is not None:
            # lazy explicit
            reference_base = self.pressure_reference_base.x
        else:
            reference_base = self.nodes[-1].pressure_re[0].x
        node_pressure_fd = np.zeros((self.num_node, self.num_f), dtype=complex)
        for fi, Z in enumerate(self.Zs):
            if fi:
                node_pressure_fd[:, [fi]] = Z @ node_injection_fd[:, [fi]]
            else:
                node_pressure_fd[-1, 0] = reference_base
                node_pressure_fd[:-1, [0]] = Z @ (node_injection_fd[:-1, [0]] -
                                                  self.Ys[0][:-1, [-1]] @ node_pressure_fd[[-1], 0][:, np.newaxis])
        node_pressure_fd *= 1e6  # recover MPa to Pa (SI)

        branch_flow_fd = np.zeros((self.num_branch, self.num_f), dtype=complex)
        _, _, Zbs, Ubs = self.get_augmented_branch_freq_domain_lumped_params(self.branches, self.num_f, self.fr, True)
        for fi in range(self.num_f):
            branch_flow_fd[:, [fi]] = (self.A.T[::3] @ node_pressure_fd[:, [fi]] - Ubs[fi][:, np.newaxis] *
                                       (self.Af.T[::3] @ node_pressure_fd[:, [fi]])) / Zbs[fi][:, np.newaxis]

        if return_base_value:
            return branch_flow_fd[:, [0]].real

        return fd2td(branch_flow_fd)[:, -self.num_tx:]

    def get_total_load(self):
        total_load = np.zeros(self.num_tx)
        for nd in self.nodes:
            if nd.has_load:
                total_load += np.asarray(nd.load)
            for g_tpu in nd.gTPUs:
                total_load += np.asarray(g_tpu.get_optimal_gas_consumption())
            for g_chp in nd.gCHPs:
                total_load += np.asarray(g_chp.get_optimal_gas_consumption())
            for g_boiler in nd.gas_boilers:
                total_load += np.asarray(g_boiler.get_optimal_heat_production()) / g_boiler.ratio_h2g
        return total_load
