# bess.py

"""Battery Energy Storage System (BESS) Model + Controller"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import cvxpy as cp


class BatteryFleet:
    """A fleet of batteries with the same specifications."""

    def __init__(self, cap, c_rate, soc_inits, timestep_sec):
        self.capacities = np.array(cap).astype(float)
        self.timestep_sec = timestep_sec
        self.c_rates = np.array(c_rate).astype(float)
        self.max_powers = self.capacities * self.c_rates
        self.energies = soc_inits * self.capacities  # initialize at 50% capacity

        print(
            f"Initialized batteries with capacities: {self.capacities}, max power: {self.max_powers}, energies: {self.energies}"
        )

    def update(self, powers):
        """
        Updates the energy level of the battery based on the power input.
        Do:
        1. if the power is within the max power
        2. update the energy level
        3. if the energy level is above the capacity, set it to the capacity, if it is below 0, set it to 0
        """

        powers = np.clip(powers, -1 * self.max_powers, self.max_powers)
        self.energies += powers * self.timestep_sec / 3600
        self.energies = np.clip(self.energies, 0, self.capacities)


class Controller(ABC):

    def __init__(self, df_loads: pd.DataFrame, df_pvs: pd.DataFrame, bess_cap, c_rate):
        """
        Initialize the controller with load and PV data, and specifications for the battery system.
        """
        self.df_loads = df_loads.values
        self.df_pvs = df_pvs.values
        self.bess = BatteryFleet(
            [bess_cap] * len(df_loads.columns),
            c_rate,
            timestep_sec=df_loads.index.freq.delta.seconds,
        )
        self.idx = 0  # To keep track of the current timestep

    @abstractmethod
    def update(self):
        """
        Update the state of the controller for a single timestep.
        This method must be implemented by subclasses.
        """
        pass

    def run(self):
        """
        Execute the control logic for the entire dataset.
        """
        grid_interactions = []
        battery_actions = []
        bess_energies = []
        loads = []
        pvs = []

        while self.idx < len(self.df_loads):
            grid_power, battery_action, bess_energy, load, pv = self.update()
            grid_interactions.append(grid_power)
            battery_actions.append(battery_action)
            bess_energies.append(bess_energy)
            loads.append(load)
            pvs.append(pv)

        # make one numpy array out of the list of arrays

        grid_interactions = np.array(grid_interactions)
        battery_actions = np.array(battery_actions)
        bess_energies = np.array(bess_energies)
        loads = np.array(loads)
        pvs = np.array(pvs)

        results_array = np.stack(
            [grid_interactions, battery_actions, bess_energies, loads, pvs], axis=2
        )

        return results_array


class RuleBasedController(Controller):

    def __init__(self, df_loads, df_pvs, bess_cap, c_rate):
        super().__init__(df_loads, df_pvs, bess_cap, c_rate)

    def update(self):
        """
        Rule-based Logic:
        1) Calculate net load
        2) Calculate potential battery charge or discharge without exceeding limits
        3) Determine battery action, respecting full or empty constraints
        4) Update batteries based on action
        5) Calculate the new net load considering the battery's contribution
        6) Grid interaction: positive for drawing from grid, negative for feeding to grid
        7) Return grid power, battery action, bess energy, load, pv
        """

        load = self.df_loads[self.idx, :]  # Load for all columns at the current index
        pv = self.df_pvs[self.idx, :]  # PV for all columns

        # Calculate net load directly with vectorized operation
        net_load = load - pv

        # Calculate potential battery charge or discharge without exceeding limits
        # Calculate remaining capacity for charging
        remaining_capacity = self.bess.capacity - self.bess.energy

        # Calculate available energy for discharging
        available_energy = self.bess.energy

        # Calculate charge or discharge potential, considering the battery's state
        discharge_potential = np.where(
            net_load > 0,
            np.minimum(net_load, np.minimum(available_energy, self.bess.max_power)),
            0,
        )
        charge_potential = np.where(
            net_load < 0,
            np.minimum(-net_load, np.minimum(remaining_capacity, self.bess.max_power)),
            0,
        )

        # Determine battery action, respecting full or empty constraints
        battery_action = np.where(net_load > 0, -discharge_potential, charge_potential)

        # Update batteries based on action
        self.bess.update(battery_action)

        # Calculate the new net load considering the battery's contribution
        net_load_after_battery = net_load + battery_action

        # Grid interaction: positive for drawing from grid, negative for feeding to grid
        grid_power = net_load_after_battery

        bess_energy = self.bess.energy

        self.idx += 1  # Increment index for next timestep

        return grid_power, battery_action, bess_energy, load, pv
