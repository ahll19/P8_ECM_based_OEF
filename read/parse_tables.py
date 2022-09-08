import bisect
import numpy as np
import pandas as pd
from utils.console_log import error


def read_bus(num_bus, wind_table, load_table):
    buses = [{} for _ in range(num_bus)]
    t0 = bisect.bisect_left(wind_table.time.values, 1)
    if wind_table.time[t0] != 1:
        error("t = 1 not in the given time series [electricity net].")
        raise ValueError("t = 1 not found [electricity net].")

    for bus in wind_table.columns[1:]:
        bus_id = int(bus[3:-4]) - 1
        buses[bus_id]["wind_his"] = wind_table[bus].values[:t0]
        buses[bus_id]["wind"] = wind_table[bus].values[t0:]

    for bus in load_table.columns[1:]:
        bus_id = int(bus[3:-4]) - 1
        buses[bus_id]["load_his"] = load_table[bus].values[:t0]
        buses[bus_id]["load"] = load_table[bus].values[t0:]

    return buses


def read_transmission_line(line_table):
    lines = [{} for _ in range(len(line_table))]

    for line_id, line in line_table.iterrows():
        lines[line_id]["from_node"] = int(line["from"]) - 1
        lines[line_id]["to_node"] = int(line["to"]) - 1
        lines[line_id]["x"] = line["x(p.u.)"]
        if not pd.isna(line["maxP(MW)"]):
            lines[line_id]["maxP"] = line["maxP(MW)"]

    return lines


def read_heat_pipe(pipe_table, load_table):
    pipes = [{} for _ in range(len(pipe_table))]

    for pipe_id, pipe in pipe_table.iterrows():
        pipes[pipe_id]["from_node"] = (int(pipe["from"]) - 1) if pipe["mass(kg/s)"] >= 0 else (int(pipe["to"]) - 1)
        pipes[pipe_id]["to_node"] = (int(pipe["to"]) - 1) if pipe["mass(kg/s)"] >= 0 else (int(pipe["from"]) - 1)
        pipes[pipe_id]["length"] = pipe["length(km)"] * 1000
        pipes[pipe_id]["mass"] = abs(pipe["mass(kg/s)"])
        pipes[pipe_id]["velocity"] = pipe["velocity(m/s)"]
        pipes[pipe_id]["diameter"] = pipe["diameter(m)"]
        pipes[pipe_id]["area"] = np.pi * pipe["diameter(m)"] ** 2 / 4
        pipes[pipe_id]["dissipation"] = pipe["dissipation(W/K/m2)"]
        pipes[pipe_id]["type"] = pipe["type"]
        if not pd.isna(pipe["maxT"]):
            pipes[pipe_id]["maxT"] = pipe["maxT"]
        if not pd.isna(pipe["minT"]):
            pipes[pipe_id]["minT"] = pipe["minT"]

    t0 = bisect.bisect_left(load_table.time.values, 1)
    if load_table.time[t0] != 1:
        error("t = 1 not in the given time series [heating net].")
        raise ValueError("t = 1 not found [heating net].")

    for pipe in load_table.columns[1:]:
        pipe_id = int(pipe[4:-4]) - 1
        pipes[pipe_id]["load_his"] = load_table[pipe].values[:t0]
        pipes[pipe_id]["load"] = load_table[pipe].values[t0:]

    return pipes


def read_gas_node(node_table, load_table):
    nodes = [{} for _ in range(len(node_table))]

    for node_id, node in node_table.iterrows():
        if not pd.isna(node["pMax(MPa)"]):
            nodes[node_id]["pMax"] = node["pMax(MPa)"]
        if not pd.isna(node["pMin(MPa)"]):
            nodes[node_id]["pMin"] = node["pMin(MPa)"]

    t0 = bisect.bisect_left(load_table.time.values, 1)
    if load_table.time[t0] != 1:
        error("t = 1 not in the given time series [gas net].")
        raise ValueError("t = 1 not found [gas net].")

    for node in load_table.columns[1:]:
        node_id = int(node[4:-6]) - 1
        nodes[node_id]["load_his"] = load_table[node].values[:t0]
        nodes[node_id]["load"] = load_table[node].values[t0:]

    return nodes


def read_gas_pipe(pipe_table):
    pipes = [{} for _ in range(len(pipe_table))]

    for pipe_id, pipe in pipe_table.iterrows():
        pipes[pipe_id]["from_node"] = int(pipe["from"]) - 1
        pipes[pipe_id]["to_node"] = int(pipe["to"]) - 1
        pipes[pipe_id]["length"] = pipe["length(km)"] * 1000
        pipes[pipe_id]["diameter"] = pipe["diameter(m)"]
        pipes[pipe_id]["area"] = np.pi * pipe["diameter(m)"] ** 2 / 4
        pipes[pipe_id]["friction"] = pipe["friction(dimensionless)"]
        pipes[pipe_id]["vBase"] = pipe["vBase(m/s)"]

    return pipes


def read_generator(gen_table):
    TPUs = []  # coal-fired, generate electric power
    CHPs = []  # coal-fired, generate both electric and heat power
    gTPUs = []  # gas-fired, generate electric power
    gCHPs = []  # gas-fired, generate both electric and heat power

    for _, gen in gen_table.iterrows():
        gen_params = dict()

        # basic parameters
        gen_params["bus"] = int(gen["eBus"]) - 1
        gen_params["maxP"] = gen["maxP(MW)"]
        gen_params["minP"] = gen["minP(MW)"]
        gen_params["upRamp"] = gen["upRamp(MW/period)"]
        gen_params["downRamp"] = gen["downRamp(MW/period)"]
        gen_params["type"] = gen["type"]
        gen_params["is_gas_fired"] = "g" in gen["type"]
        gen_params["produce_heat"] = "CHP" in gen["type"]
        if pd.isna(gen["history_schedule"]):
            gen_params["his"] = None
        else:
            gen_params["his"] = eval(gen["history_schedule"])

        # specialized parameters
        if gen["type"] == "TPU":
            gen_params["quad_P"] = gen["quad.(P)"]
            gen_params["linear_P"] = gen["linear(P)"]
            gen_params["const_P"] = gen["const.(P)"]
            TPUs.append(gen_params)
        elif gen["type"] == "CHP":
            gen_params["pipe"] = int(gen["hPipe"]) - 1
            gen_params["ratio_p2h"] = gen["p/h"]
            gen_params["quad_P"] = gen["quad.(P)"]
            gen_params["linear_P"] = gen["linear(P)"]
            gen_params["const_P"] = gen["const.(P)"]
            gen_params["quad_H"] = gen["quad.(H)"]
            gen_params["linear_H"] = gen["linear(H)"]
            gen_params["const_H"] = gen["const.(H)"]
            CHPs.append(gen_params)
        elif gen["type"] == "gTPU":
            gen_params["node"] = int(gen["gNode"]) - 1
            gen_params["ratio_p2g"] = gen["p/g(MW/kg/s)"]
            gTPUs.append(gen_params)
        elif gen["type"] == "gCHP":
            gen_params["node"] = int(gen["gNode"]) - 1
            gen_params["pipe"] = int(gen["hPipe"]) - 1
            gen_params["ratio_p2g"] = gen["p/g(MW/kg/s)"]
            gen_params["ratio_p2h"] = gen["p/h"]
            gCHPs.append(gen_params)
        else:
            error("unrecognized generator type.")
            raise ValueError("unrecognized generator type.")

    return TPUs, CHPs, gTPUs, gCHPs


def read_heat_pump(pump_table):
    heat_pumps = []
    for _, heat_pump in pump_table.iterrows():
        pump_params = dict()
        pump_params["bus"] = int(heat_pump["eBus"]) - 1
        pump_params["pipe"] = int(heat_pump["hPipe"]) - 1
        pump_params["maxH"] = heat_pump["maxH(MW)"]
        pump_params["minH"] = heat_pump["minH(MW)"]
        pump_params["upRamp"] = heat_pump["upRamp(MW/period)"]
        pump_params["downRamp"] = heat_pump["downRamp(MW/period)"]
        pump_params["ratio_p2h"] = heat_pump["p/h"]
        pump_params["type"] = "heat pump"
        if pd.isna(heat_pump["history_schedule"]):
            pump_params["his"] = None
        else:
            pump_params["his"] = eval(heat_pump["history_schedule"])
        heat_pumps.append(pump_params)

    return heat_pumps


def read_gas_boiler(boiler_table):
    gas_boilers = []
    for _, gas_boiler in boiler_table.iterrows():
        boiler_params = dict()
        boiler_params["node"] = int(gas_boiler["gNode"]) - 1
        boiler_params["pipe"] = int(gas_boiler["hPipe"]) - 1
        boiler_params["maxH"] = gas_boiler["maxH(MW)"]
        boiler_params["minH"] = gas_boiler["minH(MW)"]
        boiler_params["upRamp"] = gas_boiler["upRamp(MW/period)"]
        boiler_params["downRamp"] = gas_boiler["downRamp(MW/period)"]
        boiler_params["ratio_h2g"] = gas_boiler["h/g(MW/kg/s)"]
        boiler_params["type"] = "gas boiler"
        if pd.isna(gas_boiler["history_schedule"]):
            boiler_params["his"] = None
        else:
            boiler_params["his"] = eval(gas_boiler["history_schedule"])
        gas_boilers.append(boiler_params)

    return gas_boilers


def read_gas_well(well_table):
    gas_wells = []
    for _, gas_well in well_table.iterrows():
        well_params = dict()
        well_params["node"] = int(gas_well["gNode"]) - 1
        well_params["maxG"] = gas_well["maxG(kg/s)"]
        well_params["minG"] = gas_well["minG(kg/s)"]
        well_params["upRamp"] = gas_well["upRamp(kg/s/period)"]
        well_params["downRamp"] = gas_well["downRamp(kg/s/period)"]
        well_params["quad"] = gas_well["quad."]
        well_params["linear"] = gas_well["linear"]
        well_params["const"] = gas_well["const."]
        well_params["type"] = "gas well"
        if pd.isna(gas_well["history_schedule"]):
            well_params["his"] = None
        else:
            well_params["his"] = eval(gas_well["history_schedule"])
        gas_wells.append(well_params)

    return gas_wells
