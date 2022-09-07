import time
import numpy as np
from utils.timer import timer
from utils.console_log import info
from utils.ft import get_IDFT_matrix
from networks.gas_net import GasNetwork
from gurobipy import Model, GRB, quicksum
from read.read_instance import read_instance
from networks.heating_net import HeatingNetwork
from networks.electricity_net import ElectricityNetwork


class OptimalEnergyFlowUsingUEC:
    def __init__(self, instance_file):
        # read instance file
        (buses, lines, heat_nodes, heat_pipes, gas_nodes, gas_pipes,
         TPUs, gTPUs, CHPs, gCHPs, heat_pumps, gas_boilers, gas_wells) = read_instance(instance_file)

        # construct `network` instances
        e_net = ElectricityNetwork(buses, lines)
        g_net = GasNetwork(gas_nodes, gas_pipes)
        h_net = HeatingNetwork(heat_nodes, heat_pipes)

        # check
        assert e_net.num_tx == g_net.num_tx and e_net.num_tx == h_net.num_tx, "inconsistent boundary condition."
        assert e_net.num_th == g_net.num_th and e_net.num_th == h_net.num_th, "inconsistent boundary condition."
        assert e_net.interval == g_net.interval and e_net.interval == h_net.interval, "inconsistent boundary condition."

        # bonding attributes
        self.e_net = e_net
        self.g_net = g_net
        self.h_net = h_net
        self.TPUs = TPUs
        self.gTPUs = gTPUs
        self.CHPs = CHPs
        self.gCHPs = gCHPs
        self.heat_pumps = heat_pumps
        self.gas_boilers = gas_boilers
        self.gas_wells = gas_wells
        self.num_tx = g_net.num_tx
        self.num_th = g_net.num_th
        self.num_f = g_net.num_f
        self.all_devices = self.TPUs + self.gTPUs + self.CHPs + self.gCHPs
        self.all_devices += self.heat_pumps + self.gas_boilers + self.gas_wells
        self.all_nets = [self.e_net, self.g_net, self.h_net]
        self.model = None

    @timer("optimizing the implicit uec model")
    def optimize_implicit_uec_model(self, improve_numeric_condition=True):
        # implicit modeling means we introduce both excitation vars and response vars and use the following
        # constraints to connect them:
        # 1. FT cons for td and fd excitation vars (both history and future)
        # 2. "YU=I" cons for fd excitation vars and fd response vars
        # 3. FT cons for td and fd response vars (only future, since we cannot change the history)
        #
        # make it more clear:
        # excitation vars: device schedules, including gas production/consumption, heat production/consumption
        # response vars: network states, including network pressure, network temperature
        #
        # one more thing, for electricity networks, we always adopt explicit modeling.
        tick = time.time()
        model = Model()
        if improve_numeric_condition:
            model.setParam("NumericFocus", 3)
        all_devices = self.all_devices
        all_nets = self.all_nets

        # decision variables
        for device in all_devices:
            device.add_vars(model, self.num_tx, self.num_f)
        for net in all_nets:
            net.add_vars(model)

        # constraints
        # (1) device: max/min production [done when created variables]
        # (2) device: frequency-domain variables' freedom degree limit [done when created variables]
        # (3) device: max up/down ramp
        for device in all_devices:
            device.add_ramp_cons(model)
        # (4) device: coupling characteristics
        for device in all_devices:
            device.add_coupling_cons(model)
        # (5) system: supply-demand balance
        for net in all_nets:
            net.add_system_balance_cons(model)
        # (6) device & system: time domain-frequency domain conversion
        for device in all_devices:
            device.add_fd2td_cons(model)  # FT cons for td and fd excitation vars
        for net in all_nets:
            net.add_fd2td_cons(model)  # FT cons for td and fd response vars
        # (7) system: network equation
        for net in all_nets:
            net.add_network_cons(model)  # "YU=I" cons

        # objective
        model.setObjective(
            quicksum(device.get_cost_expr() for device in all_devices) +
            quicksum(device.get_high_freq_penalty(rho=5e-3) for device in all_devices)
        )

        model.optimize()
        solving_time = model.runTime
        modeling_time = time.time() - tick - solving_time
        info(f"modeling runs for {modeling_time:.2f}s, and solving runs for {solving_time:.2f}s.")
        self.model = model
        return model

    @timer("optimizing the explicit uec model")
    def optimize_explicit_uec_model(self, improve_numeric_condition=True):
        # explicit modeling means that we introduce only excitation vars and represent response vars as
        # expressions about excitation vars. Thus, constraints include
        # 1. FT cons for td and fd excitation vars (both history and future)
        # 2. linear expressions of response vars about excitation vars
        tick = time.time()
        model = Model()
        if improve_numeric_condition:
            model.setParam("NumericFocus", 3)
        all_devices = self.all_devices
        all_nets = self.all_nets

        # decision variables
        for device in all_devices:
            device.add_vars(model, self.num_tx, self.num_f)
        for net in all_nets:
            net.add_vars(model, implicit=False)

        # constraints
        # (1) device: max/min production [done when created variables]
        # (2) device: frequency-domain variables' freedom degree limit [done when created variables]
        # (3) device: max up/down ramp
        for device in all_devices:
            device.add_ramp_cons(model)                  # pure time-domain
        # (4) device: coupling characteristics
        for device in all_devices:
            device.add_coupling_cons(model)              # pure time-domain
        # (5) system: supply-demand balance
        for net in all_nets:
            net.add_system_balance_cons(model)           # pure time-domain
        # (6) device & system: time domain-frequency domain conversion
        for device in all_devices:
            device.add_fd2td_cons(model)  # excitation transformation in both history and future
        # (7) system: network equation
        for net in all_nets:
            net.add_network_cons(model, implicit=False)  # "U=ZI" cons

        # objective
        model.setObjective(
            quicksum(device.get_cost_expr() for device in all_devices) +
            quicksum(device.get_high_freq_penalty(rho=5e-3) for device in all_devices)
        )

        model.optimize()
        solving_time = model.runTime
        modeling_time = time.time() - tick - solving_time
        info(f"modeling runs for {modeling_time:.2f}s, and solving runs for {solving_time:.2f}s.")
        self.model = model
        return model

    @timer("optimizing the explicit uec model with lazy implementation")
    def optimize_lazy_explicit_uec_model(self, improve_numeric_condition=True, reserved_violations_each_t=1,
                                         lp_torlence=1e-8):
        # we first relax all security constraints of three energy networks, and then add them when violated
        # the involved `security constraints` include
        # (1) power flow limits of transmission lines in electricity networks
        # (2) node pressure limits of nodes in natural gas networks
        # (3) pipeline end temperature of pipelines in heating networks

        # statistics information initialization
        modeling_time = solving_time = security_check_time = 0
        num_security_cons_in_e_net = num_security_cons_in_g_net = num_security_cons_in_h_net = 0
        tick1 = time.time()

        model = Model()
        if improve_numeric_condition:
            model.setParam("NumericFocus", 3)
        all_devices = self.all_devices
        all_nets = self.all_nets
        e_net = self.e_net
        h_net = self.h_net
        g_net = self.g_net

        # decision variables
        for device in all_devices:
            device.add_vars(model, self.num_tx, self.num_f)
        for net in all_nets:
            net.add_vars(model, implicit=False)

        # constraints
        # (1) device: max/min production [done when created variables]
        # (2) device: frequency-domain variables' freedom degree limit [done when created variables]
        # (3) device: max up/down ramp
        for device in all_devices:
            device.add_ramp_cons(model)                  # pure time-domain
        # (4) device: coupling characteristics
        for device in all_devices:
            device.add_coupling_cons(model)              # pure time-domain
        # (5) system: supply-demand balance
        for net in all_nets:
            net.add_system_balance_cons(model)           # pure time-domain
        # (6) device & system: time domain-frequency domain conversion
        for device in all_devices:
            device.add_fd2td_cons(model)  # excitation transformation in both history and future
        # (7) system: network equation ---- **these constraints are relaxed**
        # for net in all_nets:
        #     net.add_network_cons(model, implicit=False)

        # objective
        model.setObjective(
            quicksum(device.get_cost_expr() for device in all_devices) +
            quicksum(device.get_high_freq_penalty(rho=5e-3) for device in all_devices)
        )

        iteration = 0
        IDFT = get_IDFT_matrix(g_net.num_th + g_net.num_tx)
        g_net.pressure_reference_base = model.addVar(lb=-GRB.INFINITY)
        g_net.fd_node_injection = [g_net.get_node_injection(fi) for fi in range(self.num_f)]
        h_net.fd_pipe_injection = [h_net.get_pipe_injection(fi) for fi in range(self.num_f)]
        num_security_cons_in_h_net += h_net.add_source_security_cons(model)  # pre-calculation trick
        while True:
            # solve the optimization
            iteration += 1
            model.setParam("BarConvTol", lp_torlence)
            model.optimize()
            solving_time += model.runTime

            # do security check
            tick2 = time.time()
            violations = self.security_check(reserved_violations_each_t)
            if sum(map(len, violations)) == 0:
                info(f"iteration-{iteration}: no more violations. End iterations.")
                security_check_time += time.time() - tick2
                break
            security_check_time += time.time() - tick2

            # add violated security constraints
            over_flow, over_pressure, under_pressure, over_temperature, under_temperature = violations
            num_security_cons_in_e_net += len(over_flow) * 2
            num_security_cons_in_g_net += len(over_pressure) + len(under_pressure)
            num_security_cons_in_h_net += len(over_temperature) + len(under_temperature)
            tick2 = time.time()
            # security constraints for power net
            for t, line_id, cont_line_id, excess in over_flow:
                if cont_line_id is None:
                    # violation without a contingency
                    e_net.branches[line_id].add_pre_contingency_transmission_cons(model, t, e_net.nodes)
                    print(" " * 4 + f"add pre-cont security cons of line-{line_id} at t-{t}: excess={excess:.3f}MW")
                else:
                    # violation with a contingency
                    e_net.branches[line_id].add_post_contingency_transmission_cons(model, t, e_net.nodes,
                                                                                   e_net.branches[cont_line_id],
                                                                                   e_net.lodf[line_id, cont_line_id])
                    print(" " * 4 + f"add post-cont security cons of line-{line_id} at t-{t}: excess={excess:.3f}MW")
            # security constraints for gas net
            for t, node_id, cont_line_id, excess in over_pressure:
                t += g_net.num_th
                g_net.nodes[node_id].add_pressure_upper_bound_cons(
                    model, t, IDFT[t], g_net.Zs, g_net.fd_node_injection, g_net.pressure_reference_base, cont_line_id,
                    g_net.get_delta_pressure_by_cont_line(node_id, cont_line_id)
                )
                print(" " * 4 + f"add max. pressure cons ({'pre' if cont_line_id is None else 'post'}-contingency) " +
                                f"of node-{node_id} at t-{t}: excess={excess:.3f}MPa")
            for t, node_id, cont_line_id, excess in under_pressure:
                t += g_net.num_th
                g_net.nodes[node_id].add_pressure_lower_bound_cons(
                    model, t, IDFT[t], g_net.Zs, g_net.fd_node_injection, g_net.pressure_reference_base, cont_line_id,
                    g_net.get_delta_pressure_by_cont_line(node_id, cont_line_id)
                )
                print(" " * 4 + f"add min. pressure cons ({'pre' if cont_line_id is None else 'post'}-contingency) " + 
                                f"of node-{node_id} at t-{t}: excess={excess:.3f}MPa")
            # security constraints for heat net
            for t, pipe_id, _, excess in over_temperature:
                t += h_net.num_th
                h_net.branches[pipe_id].add_temperature_upper_bound_cons(
                    model, t, IDFT[t], h_net.Zs, h_net.fd_pipe_injection
                )
                print(" " * 4 + f"add max. temperature cons of pipe-{pipe_id} at t-{t}: excess={excess:.3f}Celsius")
            for t, pipe_id, _, excess in under_temperature:
                t += h_net.num_th
                h_net.branches[pipe_id].add_temperature_lower_bound_cons(
                    model, t, IDFT[t], h_net.Zs, h_net.fd_pipe_injection
                )
                print(" " * 4 + f"add min. temperature cons of pipe-{pipe_id} at t-{t}: excess={excess:.3f}Celsius")
            info(f"iteration-{iteration}: {'+'.join(map(lambda x: str(len(x)), violations))} constraints added, " +
                 f"using {time.time() - tick2:.2f}s.")

        modeling_time += time.time() - tick1 - solving_time - security_check_time
        info(f"modeling runs for {modeling_time:.2f}s, solving runs for {solving_time:.2f}s, and security check " +
             f"runs for {security_check_time:.2f}s.")
        info(f"add {num_security_cons_in_e_net}(max.{e_net.num_branch * e_net.num_tx * 2}) security cons of " +
             f"electricity network.")
        info(f"add {num_security_cons_in_g_net}(max.{g_net.num_node * g_net.num_tx * 2}) security cons of " +
             f"natural gas network.")
        info(f"add {num_security_cons_in_h_net}(max.{h_net.num_branch * h_net.num_tx * 2}) security cons of " +
             f"heating network.")
        self.model = model
        return model

    def get_optimal_operation_cost(self):
        return quicksum(device.get_cost_expr() for device in self.all_devices).getValue()

    @timer("IES security check")
    def security_check(self, reserved_each_t=3):
        over_flow = self.e_net.security_check(reserved_each_t)  # consider n-1 contingencies: done
        over_pressure, under_pressure = self.g_net.security_check(reserved_each_t)  # consider n-1 contingencies: todo
        over_temperature, under_temperature = self.h_net.security_check(reserved_each_t)
        return over_flow, over_pressure, under_pressure, over_temperature, under_temperature

    def get_power_production(self):
        prod_wind = self.e_net.get_accommodated_wind_power()
        prod_gens = [np.zeros(self.num_tx) for _ in range(4)]
        for i, gens in enumerate([self.TPUs, self.gTPUs, self.CHPs, self.gCHPs]):
            for gen in gens:
                prod_gens[i] += np.asarray(gen.get_optimal_production())
        return prod_wind, *prod_gens

    def get_heat_production(self):
        prod_heat = [np.zeros(self.num_tx) for _ in range(4)]
        for i, gens in enumerate([self.CHPs, self.gCHPs, self.heat_pumps, self.gas_boilers]):
            for gen in gens:
                prod_heat[i] += np.asarray(gen.get_optimal_heat_production())
        return prod_heat


if __name__ == '__main__':
    # unit test
    from visualize.ies_plot import plot_ies_excitations_and_responses
    from visualize.ies_plot import plot_optimal_excitations
    from visualize.ies_plot import plot_optimal_responses

    # input
    instance_file = "../instance/small case/IES_E9H12G7-v1.xlsx"
    # instance_file = "../instance/small case/IES_E9H12G7-v2.xlsx"
    # instance_file = "../instance/large case/IES_E118H376G150.xlsx"
    # model_type = "implicit"
    # model_type = "explicit"
    model_type = "lazy_explicit"

    # parse input > model > optimize
    ies = OptimalEnergyFlowUsingUEC(instance_file)
    if model_type == "implicit":
        model = ies.optimize_implicit_uec_model()
    elif model_type == "explicit":
        model = ies.optimize_explicit_uec_model()
    elif model_type == "lazy_explicit":
        model = ies.optimize_lazy_explicit_uec_model()

    # output
    ies.security_check()
    info(f"optimal operation cost is {ies.get_optimal_operation_cost()}.")
    plot_ies_excitations_and_responses(ies)
    plot_optimal_excitations(ies)
    plot_optimal_responses(ies)

    # ies = OptimalEnergyFlowUsingUEC(instance_file)
    # m1 = ies.optimize_implicit_uec_model()
    # ies = OptimalEnergyFlowUsingUEC(instance_file)
    # m2 = ies.optimize_explicit_uec_model()
    # ies = OptimalEnergyFlowUsingUEC(instance_file)
    # m3 = ies.optimize_lazy_explicit_uec_model()
    #
    # from matplotlib import pyplot as plt
    # plot_params = {"font.size": 20, "font.family": "Times New Roman", "mathtext.fontset": "stix"}
    # plt.rcParams.update(plot_params)
    # plt.figure("implicit", tight_layout=True, dpi=80)
    # plt.spy(m1.getA(), ms=1)
    # plt.title("variables", fontsize=20)
    # plt.ylabel("constraints")
    # plt.figure("explicit", tight_layout=True, dpi=80)
    # plt.spy(m2.getA(), ms=1)
    # plt.title("variables", fontsize=20)
    # plt.ylabel("constraints")
    # plt.figure("lazy explicit", tight_layout=True, dpi=80)
    # plt.spy(m3.getA(), ms=1)
    # plt.title("variables", fontsize=20)
    # plt.ylabel("constraints")
