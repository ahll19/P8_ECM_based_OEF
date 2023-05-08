import argparse
import matplotlib as mpl

from algorithms.uec import OptimalEnergyFlowUsingUEC
from visualize.ies_plot import plot_optimal_responses
from visualize.ies_plot import plot_optimal_excitations
from visualize.ies_plot import plot_ies_excitations_and_responses
from results import Result, Comparer

if __name__ == '__main__':
    mpl.use("TkAgg")

    # do arguments parse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--instance_file", "-f",
        type=str,
        help="instance file in Excel format",
        default="instance/small case/v3.xlsx"
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
    parser.add_argument(
        "--lp_tolerance", "-t",
        type=float,
        help="tolerance for linear programming",
        default=1e-8
    )
    parser.add_argument(
        "--epsilon", "-e",
        type=float,
        help="tolerance for linear programming",
        default=1e-3
    )
    args = parser.parse_args()

    # do information reading
    ies = OptimalEnergyFlowUsingUEC(args.instance_file, args.cut_off)
    model, time, iteration_count = ies.optimize_lazy_explicit_uec_model(maxiter=args.maxiter, lp_torlence=args.lp_tolerance)

    # do check and output
    print(f"optimal operation cost is {ies.get_optimal_operation_cost():.2f}.")
    # plot_optimal_excitations(ies)
    # plot_optimal_responses(ies)
    # plot_ies_excitations_and_responses(ies)
    description_str = f"Instance: {args.instance_file}\n" \
                      f"Cut off: {args.cut_off}\n" \
                      f"Maxiter: {args.maxiter}\n" \
                      f"LP tolerance: {args.lp_tolerance}\n" \
                      f"modeling time: {time:.2f} seconds\n" \
                      f"Epsilon: {args.epsilon}\n" \
                      f"optimal operation cost: {ies.get_optimal_operation_cost():.2f}\n" \
                      f"iteration count: {iteration_count}"
    save_name = f"{args.instance_file.split('/')[-1].split('.')[0]}/{args.cut_off}_{args.maxiter}_{args.lp_tolerance}_{args.epsilon}"
    result = Result(ies, description=description_str)
    result.save(f"results/runs/{save_name}.pkl")
    print(result.description)

    print("done")
