from algorithms.uec2 import OptimalEnergyFlowUsingUEC2
from visualize.ies_plot import plot_optimal_responses
from visualize.ies_plot import plot_optimal_excitations
from visualize.ies_plot import plot_ies_excitations_and_responses
import matplotlib as mpl


if __name__ == '__main__':
    mpl.use("TkAgg")
    ies = OptimalEnergyFlowUsingUEC2("instance/small case/IES_E9H12G7-v1.xlsx", cut_off=None)
    ies.optimize_lazy_explicit_uec_model()

    # do check and output
    print(f"optimal operation cost is {ies.get_optimal_operation_cost():.2f}.")
    # plot_optimal_excitations(ies)
    # plot_optimal_responses(ies)
    plot_ies_excitations_and_responses(ies)

    print("Done")
