from pickle import dump, load
import numpy as np


class Result:
    power_network = None
    gas_network = None
    heat_network = None
    optimal_cost = None
    lp_tolerance = None
    max_iter = None

    def __init__(
            self, ies=None, path: str = None, description: str = None
    ) -> None:
        done = False
        if ies is not None and not done:
            self.__set_params(ies)
            if description is not None:
                self.description = description

            done = True
        if path is not None and not done:
            self.load(path)
            done = True

        if not done:
            raise Exception("Either ies or path should be set.")

    def save(self, path):
        if self.model is None:
            raise Exception("Model is not set.")

        save_obj = [
            self.power_network,
            self.gas_network,
            self.heat_network,
            self.optimal_cost,
            self.lp_tolerance,
            self.max_iter,
            self.description
        ]

        with open(path, 'wb') as f:
            dump(save_obj, f)

    def load(self, path):
        with open(path, 'rb') as f:
            save_obj = load(f)

        self.power_network = save_obj[0]
        self.gas_network = save_obj[1]
        self.heat_network = save_obj[2]
        self.optimal_cost = save_obj[3]
        self.lp_tolerance = save_obj[4]
        self.max_iter = save_obj[5]
        self.description = save_obj[6]

    def __set_params(self, ies):
        # electricty network (MW)
        p_w, p_tpu, p_ngu, p_chp, p_gchp = ies.get_power_production()
        load = ies.e_net.get_total_load()
        line_flow = ies.e_net.get_pre_contingency_line_flow()

        self.power_network = {
            "wind": p_w,
            "tpu": p_tpu,
            "ngu": p_ngu,
            "chp": p_chp,
            "gchp": p_gchp,
            "load": load,
            "transmission line flow": {
                f"{line.from_node + 1}-{line.to_node + 1}": line_flow[line.id]
                for line in ies.e_net.branches
            }
        }

        # gas network (kg/s)
        gas_production = {
            gas_well.id + 1: np.array(gas_well.get_optimal_production())
            for gas_well in ies.gas_wells
        }
        gas_load = ies.g_net.get_total_load()
        node_pressure = ies.g_net.get_pressure_curves()
        gas_response = {
            nd_id + 1: node_pressure[nd_id]
            for nd_id in range(ies.g_net.num_node)
        }

        self.gas_network = {
            "gas production": gas_production,
            "gas load": gas_load,
            "gas response": gas_response
        }

        # heat network (MW)
        h_CHP, h_gCHP, h_pump, h_boiler = ies.get_heat_production()
        heat_load = ies.g_net.get_total_load()
        node_temperature = ies.h_net.get_node_temperature_curves()
        node_temperatures = {
            nd_id + 1: node_temperature[nd_id]
            for nd_id in range(ies.h_net.num_node)
        }

        self.heat_network = {
            "CHP": h_CHP,
            "gCHP": h_gCHP,
            "pump": h_pump,
            "boiler": h_boiler,
            "heat load": heat_load,
            "node temperatures": node_temperatures
        }

        self.optimal_cost = ies.get_optimal_operation_cost()
        self.lp_tolerance = ies.model.params.BarConvTol
        self.max_iter = ies.model.getVarByName("maxiter").Obj

        self.model = ies
