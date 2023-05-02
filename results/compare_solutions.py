from .result import Result
from typing import List, Dict
from matplotlib.gridspec import GridSpec

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Comparer:
    results: List[Result] = None
    diff_power: dict = None
    diff_gas: dict = None
    diff_heat: dict = None
    diff_op_costs: dict = None

    diff_power_l1: dict = None
    diff_gas_l1: dict = None
    diff_heat_l1: dict = None
    
    def __init__(self, results: List[Result], main_idx: int) -> None:
        if len(results) != 2:
            raise Exception(
                "Only two results can be compared. This is because this module" + 
                "was written by a very smart guy who doesn't miss anything when coding."
            )
        self.main_result = results.pop(main_idx)
        self.results = results

    def difference_power(self,):
        differences = dict()
        main_power = self.main_result.power_network
        for i, result in enumerate(self.results):
            power = result.power_network
            diffs = dict()
            for key in main_power.keys():
                if key == "transmission line flow":
                    trans_diff = dict()
                    for _key in power[key].keys():
                        trans_diff[_key] = power[key][_key] - main_power[key][_key]
                    diffs[key] = trans_diff
                else:
                    diffs[key] = power[key] - main_power[key]
            differences[i] = diffs
        self.diff_power = differences

        for key, power in self.diff_power.items():
            self.diff_power_l1 = dict()
            for _key, _power in power.items():
                if _key == "transmission line flow":
                    self.diff_power_l1[_key] = dict()
                    for _key_, _power_ in _power.items():
                        self.diff_power_l1[_key][_key_] = np.linalg.norm(_power_, ord=1)
                else:
                    self.diff_power_l1[_key] = np.linalg.norm(_power, ord=1)

    def difference_gas(self,):
        differences = dict()
        main_gas = self.main_result.gas_network
        for i, result in enumerate(self.results):
            gas = result.gas_network
            diffs = dict()
            for key in main_gas.keys():
                if key in ["gas production", "gas response"]:
                    gas_diff = dict()
                    for _key in gas[key].keys():
                        gas_diff[_key] = gas[key][_key] - main_gas[key][_key]
                    diffs[key] = gas_diff
                else:
                    diffs[key] = gas[key] - main_gas[key]
            differences[i] = diffs
        self.diff_gas = differences

        for key, gas in self.diff_gas.items():
            self.diff_gas_l1 = dict()
            for _key, _gas in gas.items():
                if _key in ["gas production", "gas response"]:
                    self.diff_gas_l1[_key] = dict()
                    for _key_, _gas_ in _gas.items():
                        self.diff_gas_l1[_key][_key_] = np.linalg.norm(_gas_, ord=1)
                else:
                    self.diff_gas_l1[_key] = np.linalg.norm(_gas, ord=1)

    def difference_heat(self,):
        differences = dict()
        main_heat = self.main_result.heat_network
        for i, result in enumerate(self.results):
            heat = result.heat_network
            diffs = dict()
            for key in main_heat.keys():
                if key == "node temperatures":
                    heat_diff = dict()
                    for _key in heat[key].keys():
                        heat_diff[_key] = heat[key][_key] - main_heat[key][_key]
                    diffs[key] = heat_diff
                else:
                    diffs[key] = heat[key] - main_heat[key]
            differences[i] = diffs
        self.diff_heat = differences

        for key, heat in self.diff_heat.items():
            self.diff_heat_l1 = dict()
            for _key, _heat in heat.items():
                if _key == "node temperatures":
                    self.diff_heat_l1[_key] = dict()
                    for _key_, _heat_ in _heat.items():
                        self.diff_heat_l1[_key][_key_] = np.linalg.norm(_heat_, ord=1)
                else:
                    self.diff_heat_l1[_key] = np.linalg.norm(_heat, ord=1)

    def compare_all(self,):
        self.difference_power()
        self.difference_gas()
        self.difference_heat()

        self.diff_op_costs = dict()
        main_op_cost = self.main_result.optimal_cost
        for i, result in enumerate(self.results):
            self.diff_op_costs[i] = result.optimal_cost - main_op_cost
        
    def visualize_comparison(self,):
        compare_to = 0
        axes = []
        fig = plt.figure(tight_layout=True, figsize=(10, 12), dpi=80)
        gs = GridSpec(3, 2, figure=fig)
        ts = np.arange(len(self.main_result.power_network["load"]))

        # Power - production and consumption
        ax1 = fig.add_subplot(gs[0, 0])
        axes.append(ax1)
        ax1.set_title("Power - production and consumption")
        ax1.set_ylabel("Power [MW]")
        plot_names = ["wind", "tpu", "ngu", "chp", "gchp"]
        plot_values = [
            self.diff_power[compare_to][name]
            for name in plot_names
        ]
        for name, values in zip(plot_names, plot_values):
            ax1.plot(ts, values, label=name)
        ax1.legend()

        # Power - transmission line flow
        ax2 = fig.add_subplot(gs[0, 1])
        axes.append(ax2)
        ax2.set_title("Power - transmission line flow")
        ax2.set_ylabel("Power [MW]")
        for name, values in self.diff_power[compare_to]["transmission line flow"].items():
            ax2.plot(ts, values, label=name)
        ax2.legend()

        # Gas - production and consumption
        ax3 = fig.add_subplot(gs[1, 0])
        axes.append(ax3)
        ax3.set_title("Gas - production and consumption")
        ax3.set_ylabel("Gas [kg/h]")
        for i, gw in enumerate(self.diff_gas[compare_to]["gas production"].keys()):
            ax3.plot(
                ts, self.diff_gas[compare_to]["gas production"][gw], label=f"gaswell {i+1}"
            )
        ax3.legend()

        # Gas - node pressure
        ax4 = fig.add_subplot(gs[1, 1])
        axes.append(ax4)
        ax4.set_title("Gas - node pressure")
        ax4.set_ylabel("Pressure [MPa]")
        for i, node in enumerate(self.diff_gas[compare_to]["gas response"].keys()):
            ax4.plot(
                ts, self.diff_gas[compare_to]["gas response"][node], label=f"node {i+1}"
            )
        ax4.legend()

        # Heat - production and consumption
        ax5 = fig.add_subplot(gs[2, 0])
        axes.append(ax5)
        ax5.set_title("Heat - production and consumption")
        ax5.set_ylabel("Heat [MW]")
        plot_names = ["CHP", "gCHP", "pump", "boiler"]
        plot_values = [
            self.diff_heat[compare_to][name]
            for name in plot_names
        ]
        for name, values in zip(plot_names, plot_values):
            ax5.plot(ts, values, label=name)
        ax5.legend()

        # Heat - node temperature
        ax6 = fig.add_subplot(gs[2, 1])
        axes.append(ax6)
        ax6.set_title("Heat - node temperature")
        ax6.set_ylabel("Temperature [C]")
        for key, value in self.diff_heat[compare_to]["node temperatures"].items():
            ax6.plot(ts, value, label=key)
        ax6.legend()
        
        fig.show()
        print("woohoo")
    
    def summary(self,) -> List[str]:
        avg_trans_pow = np.mean([
            self.diff_power_l1["transmission line flow"][key]
            for key in self.diff_power_l1["transmission line flow"].keys()
        ])
        avg_resp_gas = np.mean([
            self.diff_gas_l1["gas response"][key]
            for key in self.diff_gas_l1["gas response"].keys()
        ])
        avg_prod_gas = np.mean([
            self.diff_gas_l1["gas production"][key]
            for key in self.diff_gas_l1["gas production"].keys()
        ])
        avg_heat_prod = np.mean([
            self.diff_heat_l1["node temperatures"][key]
            for key in self.diff_heat_l1["node temperatures"].keys()
        ])


        print_out = [
            f"L1 norms between the two compared results:",
            f"  Power:",
            f"    Wind: {self.diff_power_l1['wind']:.2e}",
            f"    TPU: {self.diff_power_l1['tpu']:.2e}",
            f"    NGU: {self.diff_power_l1['ngu']:.2e}",
            f"    CHP: {self.diff_power_l1['chp']:.2e}",
            f"    gCHP: {self.diff_power_l1['gchp']:.2e}",
            f"    Avg. transmission line flow. {avg_trans_pow:.2e}",
            f"  Gas:",
            f"    Avg. gas production: {avg_prod_gas:.2e}",
            f"    Avg. node pressure: {avg_resp_gas:.2e}",
            f"  Heat:",
            f"    CHP: {self.diff_heat_l1['CHP']:.2e}",
            f"    gCHP: {self.diff_heat_l1['gCHP']:.2e}",
            f"    Pump: {self.diff_heat_l1['pump']:.2e}",
            f"    Boiler: {self.diff_heat_l1['boiler']:.2e}",
            f"    Avg. node temperature: {avg_heat_prod:.2e}",
            f"==========================================",
            f"Difference in operational cost: {self.diff_op_costs[0]:.2e}",
            f"==========================================",
        ]
        print("\n".join(print_out))
        
        return print_out