import argparse
from algorithms.uec import OptimalEnergyFlowUsingUEC
from visualize.ies_plot import plot_optimal_responses
from visualize.ies_plot import plot_optimal_excitations
from visualize.ies_plot import plot_ies_excitations_and_responses
import matplotlib as mpl


if __name__ == '__main__':
    mpl.use("TkAgg")

    # do arguments parse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--instance_file", "-f",
        type=str,
        help="instance file in Excel format",
        default="instance/small case/IES_E9H12G7-v1.xlsx"
    )
    parser.add_argument(
        "--cut_off", "-c",
        type=int,
        help="cutoff value for frequency",
        default=None
    )
    parser.add_argument(
        "--maxiter", "-m",
        type=int,
        help="maximum iteration number",
        default=20
    )
    args = parser.parse_args()

    # do information reading
    ies = OptimalEnergyFlowUsingUEC(args.instance_file, args.cut_off)
    ies.optimize_lazy_explicit_uec_model(maxiter=args.maxiter)

    # do check and output
    print(f"optimal operation cost is {ies.get_optimal_operation_cost():.2f}.")
    # plot_optimal_excitations(ies)
    # plot_optimal_responses(ies)
    plot_ies_excitations_and_responses(ies)

    print("Done")
