import os
import pandas as pd
from utils.timer import timer
from utils.console_log import info, error
from devices.gas_supply import GasWell
from devices.heat_supply import HeatPump, GasBoiler
from devices.power_supply import ThermalPowerUnit, CombinedHeatPowerUnit
from devices.gas_transmission import GasNode, GasPipe
from devices.heat_transmission import HeatNode, HeatPipe
from devices.power_transmission import Bus, TransmissionLine
from read.parse_tables import read_bus, read_transmission_line, read_heat_pipe, read_gas_node, read_gas_pipe
from read.parse_tables import read_generator, read_heat_pump, read_gas_boiler, read_gas_well


@timer("reading instance")
def read_instance(case):
    if not os.path.isfile(case):
        error("Please input the correct path of instances.")
        raise ValueError("Incorrect path.")

    # excel data (occupy 99% time of this function)
    excel = pd.ExcelFile(case)
    eNet = pd.read_excel(excel, sheet_name="eNet")
    hNet = pd.read_excel(excel, sheet_name="hNet")
    gNet = pd.read_excel(excel, sheet_name="gNet")
    gNode = pd.read_excel(excel, sheet_name="gNode")
    wind = pd.read_excel(excel, sheet_name="wind")
    eLoad = pd.read_excel(excel, sheet_name="eLoad")
    hLoad = pd.read_excel(excel, sheet_name="hLoad")
    gLoad = pd.read_excel(excel, sheet_name="gLoad")
    gen = pd.read_excel(excel, sheet_name="gen")
    hPump = pd.read_excel(excel, sheet_name="hPump")
    gBoiler = pd.read_excel(excel, sheet_name="gBoiler")
    gWell = pd.read_excel(excel, sheet_name="gWell")

    # dict data
    num_bus = len(set(eNet["from"].values) | set(eNet["to"].values))
    num_heat_node = len(set(hNet["from"].values) | set(hNet["to"].values))
    buses = read_bus(num_bus, wind, eLoad)
    lines = read_transmission_line(eNet)
    heat_nodes = [{} for _ in range(num_heat_node)]
    heat_pipes = read_heat_pipe(hNet, hLoad)
    gas_nodes = read_gas_node(gNode, gLoad)
    gas_pipes = read_gas_pipe(gNet)
    TPUs, CHPs, gTPUs, gCHPs = read_generator(gen)
    heat_pumps = read_heat_pump(hPump)
    gas_boilers = read_gas_boiler(gBoiler)
    gas_wells = read_gas_well(gWell)

    # self-defined data structure
    buses = [Bus(i, bus) for i, bus in enumerate(buses)]
    lines = [TransmissionLine(i, line, buses) for i, line in enumerate(lines)]
    heat_nodes = [HeatNode(i, node) for i, node in enumerate(heat_nodes)]
    heat_pipes = [HeatPipe(i, pipe, heat_nodes) for i, pipe in enumerate(heat_pipes)]
    gas_nodes = [GasNode(i, node) for i, node in enumerate(gas_nodes)]
    gas_pipes = [GasPipe(i, pipe, gas_nodes) for i, pipe in enumerate(gas_pipes)]
    TPUs = [ThermalPowerUnit(i, TPU, buses) for i, TPU in enumerate(TPUs)]
    gTPUs = [ThermalPowerUnit(i, gTPU, buses, gas_nodes) for i, gTPU in enumerate(gTPUs)]
    CHPs = [CombinedHeatPowerUnit(i, CHP, buses, heat_pipes) for i, CHP in enumerate(CHPs)]
    gCHPs = [CombinedHeatPowerUnit(i, gCHP, buses, heat_pipes, gas_nodes) for i, gCHP in enumerate(gCHPs)]
    heat_pumps = [HeatPump(i, heat_pump, buses, heat_pipes) for i, heat_pump in enumerate(heat_pumps)]
    gas_boilers = [GasBoiler(i, gas_boiler, heat_pipes, gas_nodes) for i, gas_boiler in enumerate(gas_boilers)]
    gas_wells = [GasWell(i, gas_well, gas_nodes) for i, gas_well in enumerate(gas_wells)]

    info("Parse instance data successfully.")
    return (buses, lines, heat_nodes, heat_pipes, gas_nodes, gas_pipes,
            TPUs, gTPUs, CHPs, gCHPs, heat_pumps, gas_boilers, gas_wells)


if __name__ == '__main__':
    # unit test
    instance_file = "../instance/small case/IES_E9H12G7.xlsx"
    read_instance(instance_file)
