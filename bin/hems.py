import numpy as np
import cvxpy as cp

from cvxpylayers.torch import CvxpyLayer


class HomeEnergyManager:
    def __init__(self, horizon, energy_capacity, power_capacity, timestep):
        self.horizon = horizon
        self.energy_capacity = energy_capacity
        self.power_capacity = power_capacity
        self.timestep = timestep
        self.layer = self.get_hems()

    def get_hems(self) -> CvxpyLayer:
        p_load = cp.Parameter((self.horizon, 1), name="load", nonneg=True)
        p_production = cp.Parameter((self.horizon, 1), nonneg=True, name="production")
        p_tariff = cp.Parameter((self.horizon, 1), name="tariff", nonneg=True)
        p_initial_state_of_energy = cp.Parameter(
            (1,), nonneg=True, name="initial_state_of_energy"
        )

        v_battery_power = cp.Variable((self.horizon, 1), name="battery_power")
        v_grid_power = cp.Variable((self.horizon, 1), name="grid_power")
        v_state_of_energy = cp.Variable(
            (self.horizon, 1), nonneg=True, name="state_of_energy"
        )

        constraints = []
        constraints += [v_state_of_energy <= self.energy_capacity]
        constraints += [v_state_of_energy[0] == p_initial_state_of_energy]
        constraints += [
            v_state_of_energy[1:]
            == v_state_of_energy[:-1] + v_battery_power[:-1] * self.timestep
        ]
        constraints += [v_battery_power <= self.power_capacity]
        constraints += [v_battery_power >= -self.power_capacity]
        constraints += [v_grid_power == v_battery_power + p_load - p_production]
        constraints += [v_battery_power >= -p_load]

        tariff_cost = cp.sum(
            cp.multiply(p_tariff, cp.maximum(v_grid_power, 0)) * self.timestep
        )

        objective = cp.Minimize(tariff_cost)
        problem = cp.Problem(objective, constraints)

        hems = CvxpyLayer(
            problem,
            variables=[v_grid_power, v_state_of_energy],
            parameters=[p_load, p_production, p_tariff, p_initial_state_of_energy],
        )
        return hems
