import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 15  # Prediction horizon (seconds)
dt = 1  # Time step (seconds)
N = int(T / dt)  # Number of time steps

# State and control bounds
x_min, x_max = 0, np.inf
y_min, y_max = 0, 15  # 3 lanes, each 5m wide
v_x_min, v_x_max = 0, 30
v_y_min, v_y_max = -2, 2
a_x_min, a_x_max = -1, 2
a_y_min, a_y_max = -1, 1
j_x_min, j_x_max = -0.5, 0.5
j_y_min, j_y_max = -0.5, 0.5

# Other parameters
theta_min, theta_max = -0.4, 0.4  # in radians
omega_min, omega_max = -0.26, 0.26  # in radians/s

# Initial state and reference values
x0, y0 = 0, 2.5  # Starting in the rightmost lane
v_x0, v_y0 = 15, 0
a_x0, a_y0 = 0, 0
v_r = 15.0  # Reference speed

# Lane centers
y_lanes = [2.5, 7.5, 12.5]

# Surrounding vehicles (initial x, speed, lane)
surrounding_vehicles = [
    (30, 10, 0),  # Vehicle in right lane
    (0, 10, 1),  # Vehicle in middle lane
    (50, 10, 2)  # Vehicle in left lane
]

# Cost function weights
q1, q2, q3, q4, q5 = 1, 2, 1, 2, 4
r1, r2 = 4, 4

# Create the model
model = gp.Model("LaneChange_MIQP")

# Create variables
x = model.addVars(N + 1, lb=x_min, ub=x_max, name="x")
y = model.addVars(N + 1, lb=y_min, ub=y_max, name="y")
v_x = model.addVars(N + 1, lb=v_x_min, ub=v_x_max, name="v_x")
v_y = model.addVars(N + 1, lb=v_y_min, ub=v_y_max, name="v_y")
a_x = model.addVars(N + 1, lb=a_x_min, ub=a_x_max, name="a_x")
a_y = model.addVars(N + 1, lb=a_y_min, ub=a_y_max, name="a_y")
j_x = model.addVars(N, lb=j_x_min, ub=j_x_max, name="j_x")
j_y = model.addVars(N, lb=j_y_min, ub=j_y_max, name="j_y")

# Binary variables for lane selection
delta = model.addVars(N + 1, 3, vtype=GRB.BINARY, name="delta")

# Set initial conditions
model.addConstr(x[0] == x0)
model.addConstr(y[0] == y0)
model.addConstr(v_x[0] == v_x0)
model.addConstr(v_y[0] == v_y0)
model.addConstr(a_x[0] == a_x0)
model.addConstr(a_y[0] == a_y0)

# Add dynamics constraints
for k in range(N):
    model.addConstr(x[k + 1] == x[k] + v_x[k] * dt + 0.5 * a_x[k] * dt ** 2 + (1 / 6) * j_x[k] * dt ** 3)
    model.addConstr(y[k + 1] == y[k] + v_y[k] * dt + 0.5 * a_y[k] * dt ** 2 + (1 / 6) * j_y[k] * dt ** 3)
    model.addConstr(v_x[k + 1] == v_x[k] + a_x[k] * dt + 0.5 * j_x[k] * dt ** 2)
    model.addConstr(v_y[k + 1] == v_y[k] + a_y[k] * dt + 0.5 * j_y[k] * dt ** 2)
    model.addConstr(a_x[k + 1] == a_x[k] + j_x[k] * dt)
    model.addConstr(a_y[k + 1] == a_y[k] + j_y[k] * dt)

# Add constraints for theta and omega
for k in range(N + 1):
    model.addConstr(v_y[k] >= v_x[k] * np.tan(theta_min))
    model.addConstr(v_y[k] <= v_x[k] * np.tan(theta_max))
    model.addConstr(a_y[k] >= -v_x[k] * omega_max)
    model.addConstr(a_y[k] <= v_x[k] * omega_max)

# Lane selection constraints
for k in range(N + 1):
    # Equation 15: Lane assignment
    for gamma in range(3):
        model.addGenConstrIndicator(delta[k, gamma], True, y[k] >= y_lanes[gamma] - 2.5)
        model.addGenConstrIndicator(delta[k, gamma], True, y[k] <= y_lanes[gamma] + 2.5)

    # Equation 17: Only in one lane at a time
    model.addConstr(gp.quicksum(delta[k, gamma] for gamma in range(3)) == 1)

# Equation 16: Collision avoidance
# Equation 16: Collision avoidance
for k in range(N + 1):
    for x_v0, v, lane in surrounding_vehicles:
        x_v = x_v0 + v * k * dt  # Predicted position of the surrounding vehicle

        # If ego vehicle is in the same lane as the surrounding vehicle
        model.addGenConstrIndicator(delta[k, lane], True, x[k] <= x_v - 10)
        model.addGenConstrIndicator(delta[k, lane], True, x[k] >= x_v + 10)

# Equation 18: Complete lane change
model.addConstr(gp.quicksum(delta[N, gamma] * y_lanes[gamma] for gamma in range(3)) == y[N])

# Objective function
obj = gp.QuadExpr()
for k in range(N + 1):
    obj += q1 * (v_x[k] - v_r) ** 2 + q2 * a_x[k] ** 2 + \
           q3 * (y[k] - gp.quicksum(delta[k, gamma] * y_lanes[gamma] for gamma in range(3))) ** 2 + \
           q4 * v_y[k] ** 2 + q5 * a_y[k] ** 2
for k in range(N):
    obj += r1 * j_x[k] ** 2 + r2 * j_y[k] ** 2
model.setObjective(obj, GRB.MINIMIZE)

# Optimize the model
model.optimize()

# Extract results
x_res = [x[k].X for k in range(N + 1)]
y_res = [y[k].X for k in range(N + 1)]
v_x_res = [v_x[k].X for k in range(N + 1)]
v_y_res = [v_y[k].X for k in range(N + 1)]
a_x_res = [a_x[k].X for k in range(N + 1)]
a_y_res = [a_y[k].X for k in range(N + 1)]
j_x_res = [j_x[k].X for k in range(N)]
j_y_res = [j_y[k].X for k in range(N)]

# Plot results
fig, axs = plt.subplots(3, 2, figsize=(15, 20))

# Trajectory plot
axs[0, 0].plot(x_res, y_res, 'b-', linewidth=2)
axs[0, 0].set_xlabel('x (m)')
axs[0, 0].set_ylabel('y (m)')
axs[0, 0].set_title('Vehicle Trajectory')
for y_lane in y_lanes:
    axs[0, 0].axhline(y=y_lane, color='r', linestyle='--')
axs[0, 0].set_ylim(0, 15)
axs[0, 0].grid(True)

# Speed plots
axs[0, 1].plot(x_res, v_x_res, 'b-', label='v_x')
axs[0, 1].plot(x_res, v_y_res, 'r-', label='v_y')
axs[0, 1].set_xlabel('x (m)')
axs[0, 1].set_ylabel('Speed (m/s)')
axs[0, 1].set_title('Speed Profiles')
axs[0, 1].legend()
axs[0, 1].grid(True)

# Acceleration plots
axs[1, 0].plot(x_res, a_x_res, 'b-', label='a_x')
axs[1, 0].plot(x_res, a_y_res, 'r-', label='a_y')
axs[1, 0].set_xlabel('x (m)')
axs[1, 0].set_ylabel('Acceleration (m/s²)')
axs[1, 0].set_title('Acceleration Profiles')
axs[1, 0].legend()
axs[1, 0].grid(True)

# Jerk plots
axs[1, 1].step(x_res[:-1], j_x_res, 'b-', label='j_x', where='post')
axs[1, 1].step(x_res[:-1], j_y_res, 'r-', label='j_y', where='post')
axs[1, 1].set_xlabel('x (m)')
axs[1, 1].set_ylabel('Jerk (m/s³)')
axs[1, 1].set_title('Jerk Profiles')
axs[1, 1].legend()
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()

print("Optimization status:", model.status)
print("Objective value:", model.objVal)