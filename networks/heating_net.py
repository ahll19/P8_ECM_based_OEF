import numpy as np
from scipy import sparse
from scipy.sparse.linalg import inv
from utils.timer import timer
from networks.base_net import BaseNet
from matrices.incidence_matrix import node_branch_incidence_matrix
from matrices.admittance_matrix import branch_admittance_matrix
from gurobipy import quicksum
from utils.ft import fd2td, get_IDFT_matrix
from utils.console_log import info, warn


class HeatingNetwork(BaseNet):
    @timer("initializing the heating network")
    def __init__(self, nodes, branches, interval=900):
        # common attributes
        super().__init__(nodes, branches, interval)

        # specialized attributes
        self.heat_capacity = 4.2e3
        self.rho = 1e3
        self.A = node_branch_incidence_matrix(branches=[branch.get_incidence() for branch in branches],
                                              num_node=self.num_node, num_branch=self.num_branch)
        self.Af = self.get_Af(self.A)
        self.w_At = self.get_weighted_At(self.A, [abs(branch.mass) for branch in branches])
        self.Ys, self.Ks = self.get_Ys_and_Ks()
        if self.Ys[0].shape[0] > 1000:
            self.Zs = [inv(Y.tocsc()).A for Y in self.Ys]
        else:
            self.Zs = [np.linalg.inv(Y.A) for Y in self.Ys]  # quicker for small-scale Y
        self.implicit_model = False
        self.explicit_model = False

    def get_Ys_and_Ks(self):
        Rs = np.asarray([pipe.R for pipe in self.branches])
        Ls = np.asarray([pipe.L for pipe in self.branches])
        ms = np.asarray([abs(pipe.mass) for pipe in self.branches])
        lens = np.asarray([pipe.length for pipe in self.branches])

        j = complex(0, 1)
        Ys = []
        Ks = []
        for fi in range(self.num_f):
            f = fi * self.fr
            Zs = Rs + j * 2 * np.pi * f * Ls
            K = branch_admittance_matrix(np.exp(-self.heat_capacity * ms * Zs * lens))
            Y = sparse.eye(self.num_branch) - K @ self.Af.T @ self.w_At

            Ys.append(Y)
            Ks.append(K)

        return Ys, Ks

    def add_vars(self, model, implicit=True):
        for branch in self.branches:
            branch.add_vars(model, self.num_tx, self.num_f, implicit)

    def add_system_balance_cons(self, model):
        # intraday balance
        model.addLConstr(
            quicksum(pipe.net_injection[t] for pipe in self.branches for t in range(self.num_tx)) >= 0
        )

    @timer("heating network uec modeling")
    def add_network_cons(self, model, implicit=True):
        if not implicit:
            # explicit modeling: ZI=U
            self.fd_pipe_injection = [self.get_pipe_injection(fi) for fi in range(self.num_f)]
            self.IDFT = get_IDFT_matrix(self.num_th + self.num_tx)
            for pipe in self.branches:
                for t in range(self.num_th, self.num_th + self.num_tx):
                    if pipe.is_monitored_upper:
                        pipe.add_temperature_upper_bound_cons(model, t, self.IDFT[t], self.Zs, self.fd_pipe_injection)
                    if pipe.is_monitored_lower:
                        pipe.add_temperature_lower_bound_cons(model, t, self.IDFT[t], self.Zs, self.fd_pipe_injection)
            self.explicit_model = True
            return

        # implicit modeling: YU=I
        pipes = self.branches
        for fi in range(self.num_f):
            Y = self.Ys[fi]
            Y_re = Y.real
            Y_im = Y.imag
            for pipe1 in range(self.num_branch):
                model.addLConstr(
                    quicksum(Y_re[pipe1, pipe2] * pipes[pipe2].Tt_re[fi] - Y_im[pipe1, pipe2] * pipes[pipe2].Tt_im[fi]
                             for pipe2 in Y.getrow(pipe1).nonzero()[1]) ==
                    pipes[pipe1].net_injection_re[fi] * 1e6 / self.heat_capacity / (abs(pipes[pipe1].mass) + 1e-10)
                )
                model.addLConstr(
                    quicksum(Y_re[pipe1, pipe2] * pipes[pipe2].Tt_im[fi] + Y_im[pipe1, pipe2] * pipes[pipe2].Tt_re[fi]
                             for pipe2 in Y.getrow(pipe1).nonzero()[1]) ==
                    pipes[pipe1].net_injection_im[fi] * 1e6 / self.heat_capacity / (abs(pipes[pipe1].mass) + 1e-10)
                )
        self.implicit_model = True

    def add_fd2td_cons(self, model):
        for branch in self.branches:
            branch.add_ift_cons(model, branch.Tt_re, branch.Tt_im, branch.Tt, self.num_th)

    def get_total_load(self):
        total_load = np.zeros(self.num_tx)
        for branch in self.branches:
            if not branch.has_load:
                continue
            total_load += branch.load
        return total_load

    def security_check(self, reserved_each_t=3, epsilon=1e-2):
        recover_Tt = self.get_temperature_curves()
        upper_bound = np.asarray([pipe.maxT for pipe in self.branches])[:, np.newaxis] + epsilon
        lower_bound = np.asarray([pipe.minT for pipe in self.branches])[:, np.newaxis] - epsilon

        # filtering strategy: reserve k most severe violations along all pipes each time step (i.e., argmax_{pipe})
        # in addition to this strategy, I also tried:
        # 1) reserve k most severe violations along all time steps each pipe (i.e., argmax_{time step})
        # 2) reserve k most severe violations along all pipes and time steps (i.e., argmax_{pipe, time step})
        # but they both own worse performance.
        num_tx = recover_Tt.shape[1]
        violations_over_temperature = [[] for _ in range(num_tx)]
        num_total_over_temperature = 0
        for node_id, t in zip(*np.where(recover_Tt > upper_bound)):
            num_total_over_temperature += 1
            excess = recover_Tt[node_id, t] - upper_bound[node_id, 0]
            violations_over_temperature[t].append((t, node_id, None, excess))

        violations_under_temperature = [[] for _ in range(num_tx)]
        num_total_under_temperature = 0
        for node_id, t in zip(*np.where(recover_Tt < lower_bound)):
            num_total_under_temperature += 1
            excess = lower_bound[node_id, 0] - recover_Tt[node_id, t]
            violations_under_temperature[t].append((t, node_id, None, excess))

        if num_total_over_temperature or num_total_under_temperature:
            warn(f"{num_total_over_temperature} over temperature and {num_total_under_temperature} under temperature" +
                 f" detected in the natural gas network.")
        else:
            info("no over temperature or under temperature detected in the heating network.")

        filtered_over_temperature = []
        filtered_under_temperature = []
        if num_total_over_temperature <= num_tx / 2 and num_total_under_temperature <= num_tx / 2:
            reserved_each_t *= 2
        for violations_at_t in violations_over_temperature:
            filtered_over_temperature.extend(
                sorted(violations_at_t, key=lambda x: x[3], reverse=True)[:reserved_each_t]
            )
        for violations_at_t in violations_under_temperature:
            filtered_under_temperature.extend(
                sorted(violations_at_t, key=lambda x: x[3], reverse=True)[:reserved_each_t]
            )

        return filtered_over_temperature, filtered_under_temperature

    def get_pipe_injection(self, fi):
        pipes = self.branches
        injection_re = [pipes[i].net_injection_re[fi] * 1e6 / self.heat_capacity / (abs(pipes[i].mass) + 1e-10)
                        for i in range(self.num_branch)]
        injection_im = [pipes[i].net_injection_im[fi] * 1e6 / self.heat_capacity / (abs(pipes[i].mass) + 1e-10)
                        for i in range(self.num_branch)]
        return injection_re, injection_im

    def get_temperature_curves(self):
        # pipeline end temperature
        pipe_injection = np.asarray([pipe.get_complex_pipe_injection() for pipe in self.branches])
        pipe_mass = np.asarray([abs(pipe.mass) + 1e-10 for pipe in self.branches])[:, np.newaxis]
        pipe_delta_temp = pipe_injection * 1e6 / self.heat_capacity / pipe_mass

        recover_Tt = np.zeros((self.num_branch, self.num_f), dtype=complex)
        for fi, Z in enumerate(self.Zs):
            recover_Tt[:, [fi]] = Z @ pipe_delta_temp[:, [fi]]

        recover_Tt = fd2td(recover_Tt)[:, -self.num_tx:]

        return recover_Tt

    def get_node_temperature_curves(self):
        return self.w_At @ self.get_temperature_curves()

    @timer("add security constraints of sources (pre-calculation)")
    def add_source_security_cons(self, model):
        IDFT = get_IDFT_matrix(self.num_th + self.num_tx)
        num_monitored_pipe = 0
        for pipe in self.branches:
            if not (pipe.CHPs or pipe.gCHPs or pipe.heat_pumps or pipe.gas_boilers):
                continue
            num_monitored_pipe += 1
            for t in range(self.num_th, self.num_th + self.num_tx):
                pipe.add_temperature_upper_bound_cons(model, t, IDFT[t], self.Zs, self.fd_pipe_injection)
                pipe.add_temperature_lower_bound_cons(model, t, IDFT[t], self.Zs, self.fd_pipe_injection)
        return num_monitored_pipe * 2 * self.num_tx
