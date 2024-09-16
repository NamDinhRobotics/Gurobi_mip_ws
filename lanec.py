import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import numpy as np

# Problem parameters
N = 200  # Number of time steps
T = 20  # Total time horizon (seconds)
x_start = 0  # Initial position (m)
v_start = 10  # Initial speed (m/s)
x_target = 100  # Target position (m)
v_target = 10  # Target speed (m/s)
v_max = 20  # Max speed outside the speed bump (m/s)
v_desired = 10  # Desired speed to track (m/s)
lambda_speed_tracking = 1  # Weight for speed tracking in the objective
lambda_smooth_tracking = 1  # Weight for smooth acceleration in the objective
speed_bump_min = 30  # Speed bump start (m)
speed_bump_max = 33  # Speed bump end (m)
speed_limit_bump = 2.5  # Speed limit on the bump (m/s)
speed_limit_bump_up = 2.5 # Speed limit on the bump (m/s)
a_min, a_max = -2, 2  # Acceleration limits (m/s^2)
delta_t = T / N  # Time step size (s)
epsilon = 1e-6  # Small value for numerical stability

# Define additional weight for smooth acceleration (jerk minimization)
lambda_jerk = 10  # Weight for jerk minimization in the objective

# Create Gurobi model
model = gp.Model("smoother_vehicle_dynamics")

# Variables: position (x), speed (v), acceleration (a), binary indicators (delta)
x = model.addVars(N + 1, lb=0, name="x")  # Position (m)
v = model.addVars(N + 1, lb=0, name="v")  # Speed (m/s)
a = model.addVars(N, lb=a_min, ub=a_max, name="a")  # Acceleration (m/s^2)
delta1 = model.addVars(N + 1, vtype=GRB.BINARY, name="delta1")  # Indicator for x >= speed_bump_min
delta2 = model.addVars(N + 1, vtype=GRB.BINARY, name="delta2")  # Indicator for x <= speed_bump_max
delta3 = model.addVars(N + 1, vtype=GRB.BINARY, name="delta3")  # Indicator for v <= speed_limit_bump

# Initial conditions
model.addConstr(x[0] == x_start, "initial_position")
model.addConstr(v[0] == v_start, "initial_speed")

# Vehicle dynamics
for k in range(N):
    model.addConstr(x[k + 1] == x[k] + delta_t * v[k] + 0.5 * a[k] * delta_t**2, name=f"dynamics_x_{k}")
    model.addConstr(v[k + 1] == v[k] + delta_t * a[k], name=f"dynamics_v_{k}")

# Indicator constraints using addGenConstrIndicator (same as previous code)
for k in range(N + 1):
    model.addGenConstrIndicator(delta1[k], True, x[k] >= speed_bump_min, name=f"delta1_true_{k}")
    model.addGenConstrIndicator(delta1[k], False, x[k] <= speed_bump_min - epsilon, name=f"delta1_false_{k}")
    model.addGenConstrIndicator(delta2[k], True, x[k] <= speed_bump_max, name=f"delta2_true_{k}")
    model.addGenConstrIndicator(delta2[k], False, x[k] >= speed_bump_max + epsilon, name=f"delta2_false_{k}")
    model.addGenConstrIndicator(delta3[k], True, v[k] <= speed_limit_bump, name=f"delta3_true_{k}")
    model.addGenConstrIndicator(delta3[k], False, v[k] >= speed_limit_bump + epsilon, name=f"delta3_false_{k}")
    model.addConstr(-delta1[k] + delta3[k] <= 0, name=f"relation1_{k}")
    model.addConstr(-delta2[k] + delta3[k] <= 0, name=f"relation2_{k}")
    model.addConstr(delta1[k] + delta2[k] - delta3[k] <= 1, name=f"relation3_{k}")

# Speed limit constraints
for k in range(N + 1):
    model.addConstr(v[k] <= v_max, name=f"overall_speed_limit_{k}")

# Jerk minimization constraint: limit the difference between consecutive accelerations
jerk_limit = 1.0  # Set a reasonable limit on jerk (m/s^3)
for k in range(N - 1):
    model.addConstr(a[k + 1] - a[k] <= jerk_limit, name=f"jerk_upper_{k}")
    model.addConstr(a[k + 1] - a[k] >= -jerk_limit, name=f"jerk_lower_{k}")

# Objective: minimize control effort (acceleration), track desired speed, and smooth acceleration (jerk)
obj = lambda_smooth_tracking * gp.quicksum((a[k] * a[k]) for k in range(N)) + \
      lambda_speed_tracking * gp.quicksum((v[k] - v_desired) * (v[k] - v_desired) for k in range(N + 1)) + \
      lambda_jerk * gp.quicksum((a[k+1] - a[k]) * (a[k+1] - a[k]) for k in range(N - 1))

# Set the objective function
model.setObjective(obj, GRB.MINIMIZE)

# Solve the model
model.optimize()

# Extract solution if feasible
if model.status == GRB.OPTIMAL:
    x_sol = [x[k].X for k in range(N + 1)]
    v_sol = [v[k].X for k in range(N + 1)]
    a_sol = [a[k].X for k in range(N)]
    delta1_sol = [delta1[k].X for k in range(N + 1)]
    delta2_sol = [delta2[k].X for k in range(N + 1)]
    delta3_sol = [delta3[k].X for k in range(N + 1)]

    # Plotting (as in the previous code)
    plt.figure(figsize=(12, 10))

    # Subplot 1: Velocity vs Position
    plt.subplot(2, 1, 1)
    plt.plot(x_sol, v_sol, label='Velocity')
    plt.axvline(x=speed_bump_min, color='r', linestyle='--', label='Speed Bump Start')
    plt.axvline(x=speed_bump_max, color='r', linestyle='--', label='Speed Bump End')
    plt.axhline(y=speed_limit_bump, color='r', linestyle=':', label='Speed Bump Limit')
    plt.fill_between([speed_bump_min, speed_bump_max], 0, max(v_sol), color='r', alpha=0.2, label='Speed Bump')
    plt.axhline(y=v_max, color='g', linestyle='--', label='Max Speed')
    plt.xlabel('Position (m)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Vehicle Velocity vs Position')
    plt.legend()
    plt.grid(True)
    plt.xlim(min(x_sol), max(x_sol))

    # Subplot 2: Acceleration vs Position
    plt.subplot(2, 1, 2)
    plt.plot(x_sol[:-1], a_sol, label='Acceleration', color='b')
    plt.axvline(x=speed_bump_min, color='r', linestyle='--', label='Speed Bump Start')
    plt.axvline(x=speed_bump_max, color='r', linestyle='--', label='Speed Bump End')
    plt.fill_between([speed_bump_min, speed_bump_max], min(a_sol), max(a_sol), color='r', alpha=0.2, label='Speed Bump')
    plt.xlabel('Position (m)')
    plt.ylabel('Acceleration (m/s^2)')
    plt.title('Vehicle Acceleration vs Position')
    plt.legend()
    plt.grid(True)
    plt.xlim(min(x_sol), max(x_sol))

    plt.tight_layout()
    plt.show()

else:
    print("No feasible solution found")
