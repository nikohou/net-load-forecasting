# optimization-based control of bess
import os
import cvxpy as cp
import pandas as pd
import numpy as np
import plotly.express as px
from bess import BatteryFleet


def define_problem(optimization_horizon, bess):
    """Define the optimization problem."""

    n_households = bess.capacities.shape[0]
    timestep_seconds = bess.timestep_sec

    # parameters
    p_load = cp.Parameter((optimization_horizon, n_households), name="p_load")
    p_pv = cp.Parameter((optimization_horizon, n_households), name="p_pv")
    p_price = cp.Parameter((optimization_horizon, 1), name="p_price")
    p_init_en = cp.Parameter((n_households), name="p_init_en")

    # variables
    v_bess_en = cp.Variable(
        (optimization_horizon, n_households), nonneg=True, name="v_bess_en"
    )  # battery energy (kWh)
    v_bess_c = cp.Variable(
        (optimization_horizon, n_households), nonneg=True, name="v_bess_c"
    )  # battery charge power (kW)
    v_bess_d = cp.Variable(
        (optimization_horizon, n_households), nonneg=True, name="v_bess_d"
    )  # battery discharge power (kW)
    v_bess_direction = cp.Variable(
        (optimization_horizon, n_households), boolean=True, name="v_bess_direction"
    )  # 1 if bess charges, 0 if bess discharges
    v_netload = cp.Variable(
        (optimization_horizon, n_households), name="v_netload"
    )  # net load (kW), positive when drawing from grid, negative when feeding into grid
    v_peak = cp.Variable((1, n_households), name="v_peak")  # peak load (kW)

    # constraints
    constraints = []
    constraints += [v_bess_c.T <= np.diag(bess.max_powers) @ v_bess_direction.T]
    constraints += [v_bess_d.T <= np.diag(bess.max_powers) @ (1 - v_bess_direction).T]
    constraints += [v_bess_en[0] == p_init_en]  # initial state of charge
    constraints += [
        v_bess_en[1:]
        == v_bess_en[:-1] + (v_bess_c[:-1] - v_bess_d[:-1]) * (timestep_seconds / 3600)
    ]
    constraints += [v_bess_en <= bess.capacities.reshape(1, -1)]

    # energy balance
    constraints += [v_netload == p_load - p_pv + v_bess_c - v_bess_d]

    # peak constraint (optional, notice how the net load changes when included in the objective function)
    # constraints += [v_peak >= v_netload]
    # constraints += [v_peak >= -v_netload]

    # objective
    objective = cp.Minimize(cp.sum(p_price.T * v_netload))
    # objective = cp.Minimize(cp.sum(v_peak))

    # problem
    problem = cp.Problem(objective, constraints)

    return problem


def set_problem(opt_problem, df_idx, p_init_en):
    """
    Set the optimization problem with the given parameters.
    """

    opt_problem.param_dict["p_price"].value = df_idx["prices"].values.reshape(-1, 1)
    opt_problem.param_dict["p_load"].value = df_idx.filter(like="Load").values
    opt_problem.param_dict["p_pv"].value = df_idx.filter(like="PV").values

    opt_problem.param_dict["p_init_en"].value = p_init_en
    opt_problem.solve(verbose=True)

    return opt_problem


def unroll(df, problem, battery, optimization_horizon):

    opr_nl = []
    opr_bess_en = []
    # for idx in range(0, len(df) - optimization_horizon):

    for idx in range(0, 3):

        df_idx = df.iloc[idx : idx + optimization_horizon]

        problem = set_problem(problem, df_idx, battery.energies)

        problem.solve()

        opt_results = np.stack(
            [
                problem.var_dict["v_bess_c"].value[:, :]
                - problem.var_dict["v_bess_d"].value[:, :],
                problem.var_dict["v_netload"].value,
                problem.var_dict["v_bess_en"].value,
                problem.param_dict["p_load"].value,
                problem.param_dict["p_pv"].value,
                problem.param_dict["p_price"].value.repeat(3, axis=1),
            ],
            axis=2,
        )

        # Note: we update the real battery state with the actions taken by the optimizer, and also calculate the actual net load (loads - pvs + battery actions)
        # Since we are not using forecasts and have a perfect model of the battery, we would not have to do this.
        # We could just use the output of the optimizer directly
        # But to make it fit for the real world, we update the battery state and calculate the net load

        opr_bess_en.append(battery.energies)  # before updating the battery state

        battery_actions = opt_results[0, :, 0]
        battery.update(battery_actions)
        net_load_idx = opt_results[0, :, 3] - opt_results[0, :, 4] + battery_actions

        # check if the net load is the same as the one calculated by the optimizer
        assert np.allclose(net_load_idx, opt_results[0, :, 1]), "Net load mismatch"

        opr_nl.append(net_load_idx)

        # if idx % 20 == 0:
        # plot_optimization_results(opt_results_array, idx, 0, datetime_idx)

    opr_nl = np.array(opr_nl)
    opr_bess_en = np.array(opr_bess_en)

    results = np.concatenate([opr_nl, opr_bess_en], axis=1)

    return results


def plot_optimization_results(results_array, idx, household, datetime_idx=None):
    df_plot = pd.DataFrame(
        results_array[:, household, :],
        columns=["BESS Power", "Net Load", "BESS Energy", "Load", "PV", "Price"],
        index=datetime_idx,
    )[:200]

    fig = px.line(
        df_plot, title=f"Optimization Results for Household {household} at idx {idx}"
    )

    fig.show()


def main():

    optimization_horizon = 96
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)
    csv_path = os.path.join(dir_path, "../data/model_mock_data/df_test.csv")
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    df.index.freq = pd.Timedelta("15T")

    n_households = df.shape[1] // 2

    caps = [5] * n_households
    c_rate = [1] * n_households
    socs_init = [0] * n_households

    # create battery fleet and define optimization problem
    battery = BatteryFleet(caps, c_rate, socs_init, df.index.freq.delta.seconds)
    problem = define_problem(optimization_horizon, battery)

    # run optimization-based control for the entire dataset
    results = unroll(df, problem, battery, optimization_horizon)

    df_nl = pd.DataFrame(results, index=df.index[: results.shape[0]])

    fig = px.line(df_nl, title="Opr Net Load")
    fig.show()


if __name__ == "__main__":
    main()
