from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def plot_ies_excitations_and_responses(ies):
    plot_params = {"font.size": 15, "font.family": "Times New Roman", "mathtext.fontset": "stix"}

    fig = plt.figure("excitations and responses in IES", tight_layout=True, figsize=(10, 12), dpi=80)
    plt.rcParams.update(plot_params)
    gs = gridspec.GridSpec(3, 2)
    ts = np.arange(ies.num_tx)

    # sub-figure1: excitations in the electricity network
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("(a) power production and consumption")
    p_w, p_tpu, p_ngu, p_chp, p_gchp = ies.get_power_production()
    curves = []
    labels = []
    for gen, name in [(p_tpu, "TPU"), (p_ngu, "NGU"), (p_chp, "CHP"), (p_gchp, "gas fired-CHP"), (p_w, "wind")]:
        if gen.mean() < 1e-6:
            continue
        curves.append(gen)
        labels.append(name)
    ax1.stackplot(ts, *curves, labels=labels)
    ax1.plot(ies.e_net.get_total_load(), "k--", label="load")
    ax1.set_ylabel("electric power (MW)")
    ax1.set_xlim([0, ies.num_tx - 1])
    # ax1.set_ylim([0, 270])  # modify when necessary
    ax1.set_xticks(np.append(np.arange(0, ies.num_tx, ies.num_tx // 6), ies.num_tx - 1))
    ax1.set_xticklabels(["00:00", "04:00", "08:00", "12:00", "16:00", "20:00", "24:00"])
    ax1.legend(fontsize=15, loc="lower left")

    # sub-figure2: responses in the electricity network
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("(d) transmission line power")
    line_flow = ies.e_net.get_pre_contingency_line_flow()
    for line in ies.e_net.branches:
        plt.plot(line_flow[line.id], label=f"line {line.from_node + 1}-{line.to_node + 1}")
    ax2.set_ylabel("electric power (MW)")
    ax2.set_xlim([0, ies.num_tx - 1])
    # ax2.set_ylim([0, 270])  # modify when necessary
    ax2.set_xticks(np.append(np.arange(0, ies.num_tx, ies.num_tx // 6), ies.num_tx - 1))
    ax2.set_xticklabels(["00:00", "04:00", "08:00", "12:00", "16:00", "20:00", "24:00"])
    if ies.e_net.num_branch <= 12:
        ax2.legend(fontsize=12, loc="upper left")

    # sub-figure3: excitations in the natural gas network
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title("(b) gas production and consumption")
    curves = []
    labels = []
    for gas_well in ies.gas_wells:
        curves.append(gas_well.get_optimal_production())
        labels.append(f"gas well {gas_well.id + 1}")
    ax3.stackplot(ts, *curves, labels=labels)
    ax3.plot(ies.g_net.get_total_load(), "k--", label="load")
    ax3.set_ylabel("mass flow (kg/s)")
    ax3.set_xlim([0, ies.num_tx - 1])
    # ax.set_ylim([0, 270])  # modify when necessary
    ax3.set_xticks(np.append(np.arange(0, ies.num_tx, ies.num_tx // 6), ies.num_tx - 1))
    ax3.set_xticklabels(["00:00", "04:00", "08:00", "12:00", "16:00", "20:00", "24:00"])
    if len(ies.gas_wells) <= 12:
        ax3.legend(fontsize=15, loc="lower left")

    # sub-figure4: responses in the natural gas network
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_title("(e) node pressure")
    node_pressure = ies.g_net.get_pressure_curves()
    for nd_id in range(ies.g_net.num_node):
        plt.plot(node_pressure[nd_id], label=f"node {nd_id + 1}")
    ax4.set_ylabel("pressure (MPa)")
    ax4.set_xlim([0, ies.num_tx - 1])
    # ax4.set_ylim([0, 270])  # modify when necessary
    ax4.set_xticks(np.append(np.arange(0, ies.num_tx, ies.num_tx // 6), ies.num_tx - 1))
    ax4.set_xticklabels(["00:00", "04:00", "08:00", "12:00", "16:00", "20:00", "24:00"])
    if ies.e_net.num_branch <= 12:
        ax4.legend(fontsize=12, loc="lower left")

    # sub-figure5: excitations in the heating network
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.set_title("(c) heat production and consumption")
    h_CHP, h_gCHP, h_pump, h_boiler = ies.get_heat_production()
    curves = []
    labels = []
    for gen, name in [(h_CHP, "CHP"), (h_gCHP, "gas fired-CHP"), (h_pump, "heat pump"), (h_boiler, "gas boiler")]:
        if gen.mean() < 1e-6:
            continue
        curves.append(gen)
        labels.append(name)
    ax5.stackplot(ts, *curves, labels=labels)
    ax5.plot(ies.h_net.get_total_load(), "k--", label="load")
    ax5.set_ylabel("heat power (MW)")
    ax5.set_xlim([0, ies.num_tx - 1])
    # ax5.set_ylim([0, 270])  # modify when necessary
    ax5.set_xticks(np.append(np.arange(0, ies.num_tx, ies.num_tx // 6), ies.num_tx - 1))
    ax5.set_xticklabels(["00:00", "04:00", "08:00", "12:00", "16:00", "20:00", "24:00"])
    ax5.legend(fontsize=15, loc="lower left")

    # sub-figure6: responses in the heating network
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.set_title("(f) node temperature")
    node_temperature = ies.h_net.get_node_temperature_curves()
    for nd_id in range(ies.h_net.num_node):
        plt.plot(node_temperature[nd_id], label=f"node {nd_id + 1}")
    ax6.set_ylabel("temperature ($^{\circ}$C)")
    ax6.set_xlim([0, ies.num_tx - 1])
    # ax6.set_ylim([0, 270])  # modify when necessary
    ax6.set_xticks(np.append(np.arange(0, ies.num_tx, ies.num_tx // 6), ies.num_tx - 1))
    ax6.set_xticklabels(["00:00", "04:00", "08:00", "12:00", "16:00", "20:00", "24:00"])
    if ies.h_net.num_node <= 12:
        ax6.legend(fontsize=12, loc="lower left")

    plt.show()


def plot_optimal_excitations(ies):
    plt.figure("wind power accommodation", tight_layout=True)
    plt.plot(ies.e_net.get_accommodated_wind_power())
    plt.plot(ies.e_net.get_available_wind_power(), "--")

    plt.figure("generator power", tight_layout=True)
    for gen in ies.TPUs + ies.gTPUs + ies.CHPs + ies.gCHPs:
        plt.plot(gen.get_optimal_production(), label=f"{gen.type}{gen.id}")
    plt.legend()

    plt.figure("heat production", tight_layout=True)
    for h_gen in ies.CHPs + ies.gCHPs + ies.heat_pumps + ies.gas_boilers:
        plt.plot(h_gen.get_optimal_heat_production(), label=f"{h_gen.type}{h_gen.id}")
    plt.legend()

    plt.figure("heat supply and demand", tight_layout=True)
    plt.plot(ies.h_net.get_total_load(), "--")
    plt.plot(np.asarray([
        h_gen.get_optimal_heat_production() for h_gen in ies.CHPs + ies.gCHPs + ies.heat_pumps + ies.gas_boilers
    ]).sum(axis=0))

    plt.figure("gas network injection", tight_layout=True)
    for well in ies.gas_wells:
        plt.plot(well.get_optimal_production(), label=f"{well.type}{well.id}")
    for gen in ies.gTPUs + ies.gCHPs:
        plt.plot(gen.get_optimal_gas_consumption(), "--", label=f"{gen.type}{gen.id}")
    for boiler in ies.gas_boilers:
        plt.plot(boiler.get_optimal_gas_consumption(), "--", label=f"{boiler.type}{boiler.id}")
    plt.legend()

    plt.show()


def plot_optimal_responses(ies):
    plt.figure("transmission line power flow", tight_layout=True)
    plt.plot(ies.e_net.get_pre_contingency_line_flow().T)

    plt.figure("gas network pressure", tight_layout=True)
    if ies.g_net.implicit_model:
        for node in ies.g_net.nodes:
            plt.plot(node.get_optimal_pressure())
    else:
        plt.plot(ies.g_net.get_pressure_curves().T)

    plt.figure("gas pipeline flow", tight_layout=True)
    plt.plot(ies.g_net.get_pipe_flow_curves().T)

    plt.figure("heating network temperature", tight_layout=True)
    if ies.h_net.implicit_model:
        for pipe in ies.h_net.branches:
            plt.plot(pipe.get_optimal_temperature())
    else:
        plt.plot(ies.h_net.get_temperature_curves().T)

    plt.show()
