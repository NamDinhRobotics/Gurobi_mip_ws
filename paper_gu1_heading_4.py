import time

import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
from gurobipy import GRB

# Parameters
T = 10  # Prediction horizon (seconds)
dt = 0.1  # Time step (seconds)
N = int(T / dt)  # Number of time steps

# State and control bounds
x_min, x_max = 0, np.inf
y_min, y_max = 0, 2.
v_x_min, v_x_max = 0, 30
v_y_min, v_y_max = -1, 1
a_x_min, a_x_max = -3, 3
a_y_min, a_y_max = -0.5, 0.5
j_x_min, j_x_max = -3.5, 3.5
j_y_min, j_y_max = -1, 1

# Other parameters from Table 1
theta_min, theta_max = -0.4, 0.4  # in radians
omega_min, omega_max = -0.26, 0.26  # in radians/s

# Speed bump parameters
x_bump_start, x_bump_end = 30, 35
v_max_bump = 5

# Initial state and reference values
x0, y0 = 0, 0.75
v_x0, v_y0 = 10, 0
a_x0, a_y0 = 0, 0
v_r = 10.0  # Reference speed
y_r = y0  # Reference lateral position (center of the lane)

# wheel base
L = 2.7

# Cost function weights (from Table 1)
q1, q2, q3, q4, q5 = 1, 1, 1, 1, 1
r1, r2 = 4, 4

# Create the model
model = gp.Model("SpeedBump_MIQP")

# Create variables
x = model.addVars(N+1, lb=x_min, ub=x_max, name="x")
y = model.addVars(N+1, lb=y_min, ub=y_max, name="y")
v_x = model.addVars(N+1, lb=v_x_min, ub=v_x_max, name="v_x")
v_y = model.addVars(N+1, lb=v_y_min, ub=v_y_max, name="v_y")
a_x = model.addVars(N+1, lb=a_x_min, ub=a_x_max, name="a_x")
a_y = model.addVars(N+1, lb=a_y_min, ub=a_y_max, name="a_y")
j_x = model.addVars(N, lb=j_x_min, ub=j_x_max, name="j_x")
j_y = model.addVars(N, lb=j_y_min, ub=j_y_max, name="j_y")


epsilon = 1e-6  # Small value for numerical stability

# Binary variables for speed bump logical constraints
delta1 = model.addVars(N+1, vtype=GRB.BINARY, name="delta1")
delta2 = model.addVars(N+1, vtype=GRB.BINARY, name="delta2")
delta3 = model.addVars(N+1, vtype=GRB.BINARY, name="delta3")

# binary variables for steer to left or right
# binary variables for steer to left or right
turn_left = model.addVars(N+1, vtype=GRB.BINARY, name="delta4")
turn_right = model.addVars(N+1, vtype=GRB.BINARY, name="delta5")
is_turning = model.addVars(N+1, vtype=GRB.BINARY, name="is_turning")

# Set initial conditions
model.addConstr(x[0] == x0)
model.addConstr(y[0] == y0)
model.addConstr(v_x[0] == v_x0)
model.addConstr(v_y[0] == v_y0)
model.addConstr(a_x[0] == a_x0)
model.addConstr(a_y[0] == a_y0)

# Add dynamics constraints
for k in range(N):
    model.addConstr(x[k+1] == x[k] + v_x[k]*dt + 0.5*a_x[k]*dt**2 + (1/6)*j_x[k]*dt**3)
    model.addConstr(y[k+1] == y[k] + v_y[k]*dt + 0.5*a_y[k]*dt**2 + (1/6)*j_y[k]*dt**3)
    model.addConstr(v_x[k+1] == v_x[k] + a_x[k]*dt + 0.5*j_x[k]*dt**2)
    model.addConstr(v_y[k+1] == v_y[k] + a_y[k]*dt + 0.5*j_y[k]*dt**2)
    model.addConstr(a_x[k+1] == a_x[k] + j_x[k]*dt)
    model.addConstr(a_y[k+1] == a_y[k] + j_y[k]*dt)

# Add constraints for theta and omega (equations 5 and 6 in the paper)
for k in range(N+1):
    model.addConstr(v_y[k] >= v_x[k] * np.tan(theta_min))
    model.addConstr(v_y[k] <= v_x[k] * np.tan(theta_max))
    model.addConstr(a_y[k] >= -v_x[k] * omega_max)
    model.addConstr(a_y[k] <= v_x[k] * omega_max)

v_turn = 0.1

# Speed bump logical constraints using indicator constraints
for k in range(N+1):
    # δ1(k) = 1 ⇔ x(k) ≥ x_bump_start
    model.addGenConstrIndicator(delta1[k], True, x[k] >= x_bump_start)
    model.addGenConstrIndicator(delta1[k], False, x[k] <= x_bump_start + epsilon)

    # δ2(k) = 1 ⇔ x(k) ≤ x_bump_end
    model.addGenConstrIndicator(delta2[k], True, x[k] <= x_bump_end + epsilon)
    model.addGenConstrIndicator(delta2[k], False, x[k] >= x_bump_end)

    # δ3(k) = 1 ⇔ v_x(k) ≤ v_max_bump
    model.addGenConstrIndicator(delta3[k], True, v_x[k] <= v_max_bump)
    model.addGenConstrIndicator(delta3[k], False, v_x[k] >= v_max_bump - epsilon)

    model.addGenConstrIndicator(turn_left[k], True, v_y[k] >= v_turn)
    model.addGenConstrIndicator(turn_right[k], True, v_y[k] <= -v_turn)

    # Add the OR constraint for turning
    # Add the OR constraint for turning
    model.addGenConstrOr(is_turning[k], [turn_left[k], turn_right[k]], name=f"turn_choice_{k}")

    # Add the original condition involving delta1 and delta2
    model.addConstr(delta1[k] + delta2[k] - is_turning[k] <= 1)

    # Logical implications from equation 7
    model.addConstr(-delta1[k] + delta3[k] <= 0)
    model.addConstr(-delta2[k] + delta3[k] <= 0)
    model.addConstr(delta1[k] + delta2[k] - delta3[k] <= 1)

# Objective function
obj = gp.QuadExpr()
for k in range(N+1):
    #obj += q1 * (v_x[k] - v_r)**2 + q2 * a_x[k]**2 + q3 * (y[k] - y_r)**2 + q4 * v_y[k]**2 + q5 * a_y[k]**2
    obj += q1 * (v_x[k] - v_r)**2 + q3 * (y[k] - y_r)**2 + q4 * v_y[k]**2 + q5 * a_y[k]**2

for k in range(N):
    obj += r1 * j_x[k]**2 + r2 * j_y[k]**2

model.setObjective(obj, GRB.MINIMIZE)

# Optimize the model
time_start = time.time()
model.optimize()
time_end = time.time()
# Extract results in ms
time_solve = (time_end - time_start) * 1000
print("Time to solve the model: {:.2f} ms".format(time_solve))

# Extract results
x_res = [x[k].X for k in range(N+1)]
y_res = [y[k].X for k in range(N+1)]
v_x_res = [v_x[k].X for k in range(N+1)]
v_y_res = [v_y[k].X for k in range(N+1)]
a_x_res = [a_x[k].X for k in range(N+1)]
a_y_res = [a_y[k].X for k in range(N+1)]
j_x_res = [j_x[k].X for k in range(N)]
j_y_res = [j_y[k].X for k in range(N)]

# Compute steering angle theta = arctan(v_y / v_x)
theta_res = [np.arctan(v_y_res[k] / v_x_res[k]) for k in range(N+1)]

# compute steering angle

# compute sum of squared jerk
jerk_sum = sum(j_x_res[k]**2 + j_y_res[k]**2 for k in range(N))
print("Sum of squared jerk:", jerk_sum)

# Create the plots
fig, axs = plt.subplots(4, 2, figsize=(10, 10))

# Longitudinal speed plot
axs[0, 0].plot(x_res, v_x_res, 'b-', linewidth=2, label='v_x')
axs[0, 0].set_xlabel('x (m)')
axs[0, 0].set_ylabel('v_x (m/s)')
axs[0, 0].set_title('Longitudinal Speed Profile')
axs[0, 0].axvline(x=x_bump_start, color='r', linestyle='--', label='Speed bump start')
axs[0, 0].axvline(x=x_bump_end, color='r', linestyle='--', label='Speed bump end')
axs[0, 0].axhline(y=v_max_bump, color='g', linestyle=':', label='Max speed in bump')
axs[0, 0].fill_between([x_bump_start, x_bump_end], 0, v_x_max, alpha=0.2, color='r')
axs[0, 0].set_xlim(0, max(x_res))
axs[0, 0].set_ylim(0, v_x_max)
# axs[0, 0].legend()
#legend on left
axs[0, 0].legend(loc='upper right')
axs[0, 0].grid(True, linestyle=':', alpha=0.7)

# Lateral speed plot
axs[0, 1].plot(x_res, v_y_res, 'b-', linewidth=2, label='v_y')
axs[0, 1].set_xlabel('x (m)')
axs[0, 1].set_ylabel('v_y (m/s)')
axs[0, 1].set_title('Lateral Speed Profile')
axs[0, 1].axvline(x=x_bump_start, color='r', linestyle='--', label='Speed bump start')
axs[0, 1].axvline(x=x_bump_end, color='r', linestyle='--', label='Speed bump end')
axs[0, 1].set_xlim(0, max(x_res))
axs[0, 1].set_ylim(v_y_min, v_y_max)
axs[0, 1].legend()
axs[0, 1].grid(True, linestyle=':', alpha=0.7)

# Longitudinal acceleration plot
axs[1, 0].plot(x_res, a_x_res, 'b-', linewidth=2, label='a_x')
axs[1, 0].set_xlabel('x (m)')
axs[1, 0].set_ylabel('a_x (m/s²)')
axs[1, 0].set_title('Longitudinal Acceleration Profile')
axs[1, 0].axvline(x=x_bump_start, color='r', linestyle='--', label='Speed bump start')
axs[1, 0].axvline(x=x_bump_end, color='r', linestyle='--', label='Speed bump end')
axs[1, 0].set_xlim(0, max(x_res))
axs[1, 0].set_ylim(a_x_min, a_x_max)
axs[1, 0].legend()
axs[1, 0].grid(True, linestyle=':', alpha=0.7)

# Lateral acceleration plot
axs[1, 1].plot(x_res, a_y_res, 'b-', linewidth=2, label='a_y')
axs[1, 1].set_xlabel('x (m)')
axs[1, 1].set_ylabel('a_y (m/s²)')
axs[1, 1].set_title('Lateral Acceleration Profile')
axs[1, 1].axvline(x=x_bump_start, color='r', linestyle='--', label='Speed bump start')
axs[1, 1].axvline(x=x_bump_end, color='r', linestyle='--', label='Speed bump end')
axs[1, 1].set_xlim(0, max(x_res))
axs[1, 1].set_ylim(a_y_min, a_y_max)
axs[1, 1].legend()
axs[1, 1].grid(True, linestyle=':', alpha=0.7)

# Vehicle Trajectory plot
axs[2, 0].plot(x_res, y_res, 'b-', linewidth=2, label='Vehicle Path')
axs[2, 0].set_xlabel('x (m)')
axs[2, 0].set_ylabel('y (m)')
axs[2, 0].set_title('Vehicle Trajectory')
axs[2, 0].axvline(x=x_bump_start, color='r', linestyle='--', label='Speed bump start')
axs[2, 0].axvline(x=x_bump_end, color='r', linestyle='--', label='Speed bump end')
axs[2, 0].set_xlim(0, max(x_res))
axs[2, 0].set_ylim(y_min, y_max)
axs[2, 0].legend()
axs[2, 0].grid(True, linestyle=':', alpha=0.7)

# Steering angle plot
axs[2, 1].plot(x_res, theta_res, 'b-', linewidth=2, label='Steering Angle')
axs[2, 1].set_xlabel('x (m)')
axs[2, 1].set_ylabel('θ (rad)')
axs[2, 1].set_title('Steering Angle Profile')
axs[2, 1].axvline(x=x_bump_start, color='r', linestyle='--', label='Speed bump start')
axs[2, 1].axvline(x=x_bump_end, color='r', linestyle='--', label='Speed bump end')
axs[2, 1].set_xlim(0, max(x_res))
axs[2, 1].set_ylim(theta_min, theta_max)
axs[2, 1].legend()
axs[2, 1].grid(True, linestyle=':', alpha=0.7)
# jerk plot
axs[3, 0].step(x_res[:-1], j_x_res, 'b-', linewidth=2, where='post')
axs[3, 0].set_xlabel('x (m)')
axs[3, 0].set_ylabel('j_x (m/s³)')
axs[3, 0].set_title('Longitudinal Jerk Profile')
axs[3, 0].axvline(x=x_bump_start, color='r', linestyle='--', label='Speed bump start')
axs[3, 0].axvline(x=x_bump_end, color='r', linestyle='--', label='Speed bump end')
axs[3, 0].set_xlim(0, max(x_res))
axs[3, 0].set_ylim(j_x_min, j_x_max)
axs[3, 0].legend()
axs[3, 0].grid(True, linestyle=':', alpha=0.7)
# jerk plot y
axs[3, 1].step(x_res[:-1], j_y_res, 'b-', linewidth=2, where='post')
axs[3, 1].set_xlabel('x (m)')
axs[3, 1].set_ylabel('j_y (m/s³)')
axs[3, 1].set_title('Lateral Jerk Profile')
axs[3, 1].axvline(x=x_bump_start, color='r', linestyle='--', label='Speed bump start')
axs[3, 1].axvline(x=x_bump_end, color='r', linestyle='--', label='Speed bump end')
axs[3, 1].set_xlim(0, max(x_res))
axs[3, 1].set_ylim(j_y_min, j_y_max)
axs[3, 1].legend()
axs[3, 1].grid(True, linestyle=':', alpha=0.7)


plt.tight_layout()
plt.show()

print("Optimization status:", model.status)
print("Objective value:", model.objVal)

print("Optimization status:", model.status)
print("Objective value:", model.objVal)


print("Optimization status:", model.status)
print("Objective value:", model.objVal)