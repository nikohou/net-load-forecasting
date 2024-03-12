# optimization-based control of bess
import os
import cvxpy as cp
import pandas as pd
import numpy as np
import plotly.express as px
from bess import BatteryFleet

np.random.seed(13)


class OptimizationProblem:
    def __init__(self, optimization_horizon, bess):
        self.optimization_horizon = optimization_horizon
        self.bess = bess
        self.problem = None

    def define_problem(self):
        """Define the optimization problem."""
        n_households = self.bess.capacities.shape[0]
        timestep_seconds = self.bess.timestep_sec

        # parameters
        p_load = cp.Parameter((self.optimization_horizon, n_households), name="p_load")
        p_pv = cp.Parameter((self.optimization_horizon, n_households), name="p_pv")
        p_price = cp.Parameter((self.optimization_horizon, 1), name="p_price")
        p_init_en = cp.Parameter((n_households), name="p_init_en")

        # variables
        v_bess_en = cp.Variable(
            (self.optimization_horizon, n_households), nonneg=True, name="v_bess_en"
        )  # battery energy (kWh)
        v_bess_c = cp.Variable(
            (self.optimization_horizon, n_households), nonneg=True, name="v_bess_c"
        )  # battery charge power (kW)
        v_bess_d = cp.Variable(
            (self.optimization_horizon, n_households), nonneg=True, name="v_bess_d"
        )  # battery discharge power (kW)
        v_bess_direction = cp.Variable(
            (self.optimization_horizon, n_households),
            boolean=True,
            name="v_bess_direction",
        )  # 1 if bess charges, 0 if bess discharges
        v_netload = cp.Variable(
            (self.optimization_horizon, n_households), name="v_netload"
        )  # net load (kW), positive when drawing from grid, negative when feeding into grid
        v_peak = cp.Variable((1, n_households), name="v_peak")  # peak load (kW)

        # constraints
        constraints = []
        constraints += [
            v_bess_c.T <= np.diag(self.bess.max_powers) @ v_bess_direction.T
        ]
        constraints += [
            v_bess_d.T <= np.diag(self.bess.max_powers) @ (1 - v_bess_direction).T
        ]
        constraints += [v_bess_en[0] == p_init_en]  # initial state of charge
        constraints += [
            v_bess_en[1:]
            == v_bess_en[:-1]
            + (v_bess_c[:-1] - v_bess_d[:-1]) * (timestep_seconds / 3600)
        ]
        constraints += [v_bess_en <= self.bess.capacities.reshape(1, -1)]

        # energy balance
        constraints += [v_netload == p_load - p_pv + v_bess_c - v_bess_d]

        # peak constraint (optional, notice how the net load changes when included in the objective function)
        constraints += [v_peak >= v_netload]
        constraints += [v_peak >= -v_netload]

        # objective
        # objective = cp.Minimize(cp.sum(p_price.T * v_netload) + 1000 * cp.sum(v_peak))

        objective = cp.Minimize(100 * cp.sum(v_peak) + cp.sum(p_price.T * 1))

        # problem
        self.problem = cp.Problem(objective, constraints)

    def set_problem(self, df_idx, p_init_en):
        """
        Set the optimization problem with the given parameters.
        """
        self.problem.param_dict["p_price"].value = df_idx["prices"].values.reshape(
            -1, 1
        )
        self.problem.param_dict["p_load"].value = df_idx.filter(like="Load").values
        self.problem.param_dict["p_pv"].value = df_idx.filter(like="PV").values
        self.problem.param_dict["p_init_en"].value = p_init_en
        self.problem.solve(verbose=True)

    def solve_problem(self):
        self.problem.solve()


def do_one_optimization(df, problem, battery, idx, optimization_horizon):

    df_idx = df.iloc[idx : idx + optimization_horizon]
    problem.set_problem(df_idx, battery.energies)
    problem.solve_problem()

    opt_results = np.stack(
        [
            problem.problem.var_dict["v_bess_c"].value[:, :]
            - problem.problem.var_dict["v_bess_d"].value[:, :],
            problem.problem.var_dict["v_netload"].value,
            problem.problem.var_dict["v_bess_en"].value,
            problem.problem.param_dict["p_load"].value,
            problem.problem.param_dict["p_pv"].value,
            problem.problem.param_dict["p_price"].value.repeat(3, axis=1),
        ],
        axis=2,
    )

    return opt_results


def unroll(df, problem, battery, optimization_horizon, error_std=0.05):
    opr_nl = []
    opr_bess_en = []

    # for idx in range(0, len(df) - optimization_horizon):
    for idx in range(0, 96):

        df_idx = df.iloc[idx : idx + optimization_horizon]
        problem.set_problem(df_idx, battery.energies)
        problem.solve_problem()

        opt_results = np.stack(
            [
                problem.problem.var_dict["v_bess_c"].value[:, :]
                - problem.problem.var_dict["v_bess_d"].value[:, :],
                problem.problem.var_dict["v_netload"].value,
                problem.problem.var_dict["v_bess_en"].value,
                problem.problem.param_dict["p_load"].value,
                problem.problem.param_dict["p_pv"].value,
                problem.problem.param_dict["p_price"].value.repeat(3, axis=1),
            ],
            axis=2,
        )

        opr_bess_en.append(battery.energies)  # before updating the battery state
        battery_actions = opt_results[0, :, 0]
        battery.update(battery_actions)  # update battery state
        net_load_idx = (
            opt_results[0, :, 3]
            - opt_results[0, :, 4]
            + battery_actions
            + np.random.normal(0, error_std, 1)
        )  # load - pv + battery + random forecast error

        opr_nl.append(net_load_idx)

    # convert to numpy arrays
    opr_nl = np.array(opr_nl)
    opr_bess_en = np.array(opr_bess_en)

    results = np.concatenate([opr_nl, opr_bess_en], axis=1)

    return results


def plot_optimization_results(results_array, idx, household, datetime_idx=None):
    """Helper Function to Plot a single optimization result for a given household and idx."""
    df_plot = pd.DataFrame(
        results_array[:, household, :],
        columns=["BESS Power", "Net Load", "BESS Energy", "Load", "PV", "Price"],
        index=datetime_idx,
    )[:200]

    fig = px.line(
        df_plot, title=f"Optimization Results for Household {household} at idx {idx}"
    )

    fig.show()


def aggregate_netloads(df_nl):

    df_sum_nl = df_nl.filter(like="NL").sum(axis=1).to_frame(name="Sum_NL")

    df_sum_nl.to_csv("../data/model_mock_data/sum_nl.csv")

    px.line(df_sum_nl, title="Aggregated Net Load").show()


def main():
    optimization_horizon = 96
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)
    csv_path = os.path.join(dir_path, "../data/model_mock_data/df_test.csv")
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df.index.freq = pd.Timedelta("15T")  # TODO: make this depend on the data
    n_households = df.shape[1] // 2

    caps = [5] * n_households
    c_rate = [1] * n_households
    socs_init = [0] * n_households

    # create battery fleet and define optimization problem
    battery = BatteryFleet(caps, c_rate, socs_init, df.index.freq.delta.seconds)
    problem = OptimizationProblem(optimization_horizon, battery)
    problem.define_problem()

    # NOTE: Running a roll-out
    results = unroll(df, problem, battery, optimization_horizon)

    df_nl = pd.DataFrame(
        results,
        index=df.index[: results.shape[0]],
        columns=[f"NL_{i}" for i in range(n_households)]
        + [f"BESS_EN_{i}" for i in range(n_households)],
    )

    fig = px.line(df_nl, title="Opr Net Load")
    fig.show()

    aggregate_netloads(df_nl)

    # NOTE: Run to test a single optimization
    # idx = 10
    # household = 0
    # opt_results = do_one_optimization(df, problem, battery, idx, optimization_horizon)
    # plot_optimization_results(
    #     opt_results, idx, household, df.index[:optimization_horizon]
    # )


if __name__ == "__main__":
    main()
