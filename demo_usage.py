import argparse
from algorithms.uec import OptimalEnergyFlowUsingUEC
from visualize.ies_plot import plot_optimal_responses
from visualize.ies_plot import plot_optimal_excitations
from visualize.ies_plot import plot_ies_excitations_and_responses


if __name__ == '__main__':
    # do arguments parse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--instance_file", "-f",
        type=str,
        help="instance file in Excel format",
        # default="instance/small case/IES_E9H12G7-v1.xlsx"
        default="instance/case n-1/IES_E9H12G7.xlsx"
        # default="instance/case n-1/IES_E118H376G150.xlsx"
    )
    parser.add_argument(
        "--model_type", "-m",
        type=str,
        help="model type: implicit, explicit, and lazy_explicit",
        default="lazy_explicit"
    )
    args = parser.parse_args()

    # do information reading
    ies = OptimalEnergyFlowUsingUEC(args.instance_file)

    # do modeling and optimization
    if args.model_type == "implicit":
        ies.optimize_implicit_uec_model()
    elif args.model_type == "explicit":
        ies.optimize_explicit_uec_model()
    elif args.model_type == "lazy_explicit":
        ies.optimize_lazy_explicit_uec_model()

    # do check and output
    print(f"optimal operation cost is {ies.get_optimal_operation_cost()}.")
    # plot_optimal_excitations(ies)
    # plot_optimal_responses(ies)
    # plot_ies_excitations_and_responses(ies)
