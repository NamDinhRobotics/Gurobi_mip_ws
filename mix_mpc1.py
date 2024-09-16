import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
from scipy import linalg


class VehicleMPC:
    def __init__(self):
        # System parameters
        self.Ts = 0.1  # Sampling time
        self.Np = 20  # Prediction horizon
        self.Nc = 10  # Control horizon (Nc <= Np)

        # Vehicle parameters
        self.vmax = 30  # Maximum velocity (m/s)
        self.vmin = 0  # Minimum velocity (m/s)
        self.amax = 3  # Maximum acceleration (m/s^2)
        self.amin = -3  # Minimum acceleration (m/s^2)
        self.jmax = 2  # Maximum jerk (m/s^3)
        self.jmin = -2  # Minimum jerk (m/s^3)

        # Speed bump parameters
        self.xbump_start = 50  # Start of speed bump (m)
        self.xbump_end = 52  # End of speed bump (m)
        self.vmax_bump = 5  # Maximum velocity on speed bump (m/s)

        # Big M for constraints
        self.M = 1000

        # State space model
        self.A = np.array([[1, self.Ts, 0.5 * self.Ts ** 2],
                           [0, 1, self.Ts],
                           [0, 0, 1]])
        self.B = np.array([self.Ts ** 3 / 6, self.Ts ** 2 / 2, self.Ts]).reshape(-1, 1)
        self.C = np.array([[1, 0, 0],
                           [0, 1, 0]])

        # Weight matrices
        self.Q = np.diag([10, 1, 0.1])  # State weights
        self.R = 0.1  # Input weight

        # Terminal cost (solution to discrete-time algebraic Riccati equation)
        self.P = linalg.solve_discrete_are(self.A, self.B, self.Q, self.R)

    def create_mpc_model(self, x0):
        model = gp.Model("VehicleMPC")

        # Decision variables
        x = model.addMVar((3, self.Np + 1), lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")
        u = model.addMVar((1, self.Np), lb=self.jmin, ub=self.jmax, name="u")
        delta = model.addMVar((3, self.Np + 1), vtype=GRB.BINARY, name="delta")

        # Objective function
        obj = 0
        for k in range(self.Np):
            obj += x[:, k].T @ self.Q @ x[:, k] + u[:, k].T @ self.R @ u[:, k]
        obj += x[:, self.Np].T @ self.P @ x[:, self.Np]  # Terminal cost
        model.setObjective(obj, GRB.MINIMIZE)

        # Constraints
        # Initial condition
        model.addConstr(x[:, 0] == x0)

        # System dynamics
        for k in range(self.Np):
            model.addConstr(x[:, k + 1] == self.A @ x[:, k] + self.B @ u[:, k])

        # State and input constraints
        for k in range(self.Np + 1):
            model.addConstr(self.vmin <= x[1, k])
            model.addConstr(x[1, k] <= self.vmax)
            model.addConstr(self.amin <= x[2, k])
            model.addConstr(x[2, k] <= self.amax)

            # Speed bump constraints
            model.addConstr(x[1, k] <= self.vmax_bump + self.M * (2 - delta[0, k] - delta[1, k]))
            model.addConstr(x[0, k] >= self.xbump_start - self.M * (1 - delta[0, k]))
            model.addConstr(x[0, k] <= self.xbump_end + M * (1 - delta[1, k]))

        # Terminal constraint for stability
        model.addConstr(x[:, self.Np] == 0)  # Simple terminal constraint

        return model, x, u, delta

    def simulate(self, x0, sim_time):
        n_steps = int(sim_time / self.Ts)
        x_history = np.zeros((3, n_steps + 1))
        u_history = np.zeros((1, n_steps))
        x_history[:, 0] = x0

        for step in range(n_steps):
            model, x, u, delta = self.create_mpc_model(x_history[:, step])
            model.optimize()

            if model.status == GRB.OPTIMAL:
                u_history[:, step] = u[:, 0].X
                x_history[:, step + 1] = self.A @ x_history[:, step] + self.B @ u_history[:, step]
            else:
                print(f"Optimization failed at step {step}")
                break

        return x_history, u_history

    def plot_results(self, x_history, u_history):
        time = np.arange(0, x_history.shape[1]) * self.Ts

        fig, axs = plt.subplots(4, 1, figsize=(10, 15))

        axs[0].plot(time, x_history[0, :])
        axs[0].set_ylabel('Position (m)')
        axs[0].axvline(x=self.xbump_start / self.vmax, color='r', linestyle='--', label='Speed Bump')
        axs[0].axvline(x=self.xbump_end / self.vmax, color='r', linestyle='--')
        axs[0].legend()

        axs[1].plot(time, x_history[1, :])
        axs[1].set_ylabel('Velocity (m/s)')
        axs[1].axhline(y=self.vmax_bump, color='r', linestyle='--', label='Speed Bump Limit')
        axs[1].legend()

        axs[2].plot(time, x_history[2, :])
        axs[2].set_ylabel('Acceleration (m/s^2)')

        axs[3].plot(time[:-1], u_history[0, :])
        axs[3].set_ylabel('Jerk (m/s^3)')
        axs[3].set_xlabel('Time (s)')

        plt.tight_layout()
        plt.show()

    def sensitivity_analysis(self, x0, sim_time):
        base_Np = self.Np
        base_Q = self.Q.copy()

        variations = [0.5, 1, 2]
        results = []

        for var in variations:
            # Vary Np
            self.Np = int(base_Np * var)
            x_history, _ = self.simulate(x0, sim_time)
            results.append(('Np', var, x_history[1, -1]))  # Final velocity

            # Vary Q
            self.Np = base_Np
            self.Q = base_Q * var
            x_history, _ = self.simulate(x0, sim_time)
            results.append(('Q', var, x_history[1, -1]))  # Final velocity

        self.Np = base_Np
        self.Q = base_Q
        return results


# Main simulation
mpc = VehicleMPC()
x0 = np.array([0, 20, 0])  # Initial state: [position, velocity, acceleration]
sim_time = 10  # Simulation time in seconds

x_history, u_history = mpc.simulate(x0, sim_time)
mpc.plot_results(x_history, u_history)

# Sensitivity analysis
sensitivity_results = mpc.sensitivity_analysis(x0, sim_time)
for param, var, final_vel in sensitivity_results:
    print(f"Parameter: {param}, Variation: {var}, Final Velocity: {final_vel}")