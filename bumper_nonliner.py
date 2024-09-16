import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
from gurobipy import GRB

# Parameters
T = 10  # Prediction horizon (seconds)
dt = 0.5  # Time step (seconds)
N = int(T / dt)  # Number of time steps

# Vehicle parameters
L = 2.7  # Wheelbase (m)

# State and control bounds
x_min, x_max = 0, np.inf
y_min, y_max = 0, 2.
theta_min, theta_max = -np.pi, np.pi
v_min, v_max = 0, 20
steering_min, steering_max = -0.5, 0.5  # Steering angle bounds (rad)
accel_min, accel_max = -2, 2  # Acceleration bounds (m/s^2)
jerk_min, jerk_max = -5, 5  # Jerk bounds (m/s^3)

# Speed bump parameters
x_bump_start, x_bump_end = 30, 35
v_max_bump = 5

# Initial state and reference values
x0, y0 = 0, 0.75
theta0, v0 = 0, 10
steering0, accel0 = 0, 0
v_r = 10.0  # Reference speed
y_r = y0  # Reference lateral position (center of the lane)

# Cost function weights
q1, q2, q3, q4, q5 = 1, 1, 1, 2, 10
r1, r2 = 4, 4

# Create the model
model = gp.Model("SpeedBump_MIQP")

# Create variables
x = model.addVars(N+1, lb=x_min, ub=x_max, name="x")
y = model.addVars(N+1, lb=y_min, ub=y_max, name="y")
theta = model.addVars(N+1, lb=theta_min, ub=theta_max, name="theta")
v = model.addVars(N+1, lb=v_min, ub=v_max, name="v")
steering = model.addVars(N+1, lb=steering_min, ub=steering_max, name="steering")
accel = model.addVars(N+1, lb=accel_min, ub=accel_max, name="accel")
jerk = model.addVars(N, lb=jerk_min, ub=jerk_max, name="jerk")

# Add trigonometric variables
cos_theta = model.addVars(N+1, lb=-1, ub=1, name="cos_theta")
sin_theta = model.addVars(N+1, lb=-1, ub=1, name="sin_theta")
tan_steering = model.addVars(N+1, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="tan_steering")

# Binary variables for speed bump logical constraints
delta1 = model.addVars(N+1, vtype=GRB.BINARY, name="delta1")
delta2 = model.addVars(N+1, vtype=GRB.BINARY, name="delta2")
delta3 = model.addVars(N+1, vtype=GRB.BINARY, name="delta3")

# Set initial conditions
model.addConstr(x[0] == x0)
model.addConstr(y[0] == y0)
model.addConstr(theta[0] == theta0)
model.addConstr(v[0] == v0)
model.addConstr(steering[0] == steering0)
model.addConstr(accel[0] == accel0)

# Add trigonometric constraints
for k in range(N+1):
    model.addGenConstrCos(theta[k], cos_theta[k], "cos_constr_" + str(k))
    model.addGenConstrSin(theta[k], sin_theta[k], "sin_constr_" + str(k))
    model.addGenConstrTan(steering[k], tan_steering[k], "tan_constr_" + str(k))

# Add dynamics constraints
for k in range(N):
    model.addConstr(x[k+1] == x[k] + v[k] * cos_theta[k] * dt)
    model.addConstr(y[k+1] == y[k] + v[k] * sin_theta[k] * dt)
    model.addConstr(theta[k+1] == theta[k] + v[k] / L * tan_steering[k] * dt)
    model.addConstr(v[k+1] == v[k] + accel[k] * dt)
    model.addConstr(accel[k+1] == accel[k] + jerk[k] * dt)

# Speed bump logical constraints using indicator constraints
for k in range(N+1):
    model.addGenConstrIndicator(delta1[k], True, x[k] >= x_bump_start)
    model.addGenConstrIndicator(delta1[k], False, x[k] <= x_bump_start)

    model.addGenConstrIndicator(delta2[k], True, x[k] <= x_bump_end)
    model.addGenConstrIndicator(delta2[k], False, x[k] >= x_bump_end)

    model.addGenConstrIndicator(delta3[k], True, v[k] <= v_max_bump)
    model.addGenConstrIndicator(delta3[k], False, v[k] >= v_max_bump)

    # Logical implications
    model.addConstr(-delta1[k] + delta3[k] <= 0)
    model.addConstr(-delta2[k] + delta3[k] <= 0)
    model.addConstr(delta1[k] + delta2[k] - delta3[k] <= 1)

# Objective function
obj = gp.QuadExpr()
for k in range(N+1):
    obj += q1 * (v[k] - v_r)**2 + q2 * (y[k] - y_r)**2 + q3 * theta[k]**2 + q4 * steering[k]**2

for k in range(N):
    obj += r1 * accel[k]**2 + r2 * (steering[k+1] - steering[k])**2 + q5 * jerk[k]**2  # Add jerk minimization

model.setObjective(obj, GRB.MINIMIZE)

# Optimize the model
model.optimize()

# Extract results
x_res = [x[k].X for k in range(N+1)]
y_res = [y[k].X for k in range(N+1)]
theta_res = [theta[k].X for k in range(N+1)]
v_res = [v[k].X for k in range(N+1)]
steering_res = [steering[k].X for k in range(N+1)]
accel_res = [accel[k].X for k in range(N+1)]
jerk_res = [jerk[k].X for k in range(N)]

# Create the plots
fig, axs = plt.subplots(3, 2, figsize=(15, 15))

# Longitudinal position and speed plot
axs[0, 0].plot(x_res, v_res, 'b-', linewidth=2, label='v')
axs[0, 0].set_xlabel('x (m)')
axs[0, 0].set_ylabel('v (m/s)')
axs[0, 0].set_title('Longitudinal Speed Profile')
axs[0, 0].axvline(x=x_bump_start, color='r', linestyle='--', label='Speed bump start')
axs[0, 0].axvline(x=x_bump_end, color='r', linestyle='--', label='Speed bump end')
axs[0, 0].axhline(y=v_max_bump, color='g', linestyle=':', label='Max speed in bump')
axs[0, 0].fill_between([x_bump_start, x_bump_end], 0, v_max, alpha=0.2, color='r')
axs[0, 0].set_xlim(0, max(x_res))
axs[0, 0].set_ylim(0, v_max)
axs[0, 0].legend()
axs[0, 0].grid(True, linestyle=':', alpha=0.7)

# Lateral position plot
axs[0, 1].plot(x_res, y_res, 'b-', linewidth=2, label='y')
axs[0, 1].set_xlabel('x (m)')
axs[0, 1].set_ylabel('y (m)')
axs[0, 1].set_title('Lateral Position Profile')
axs[0, 1].axvline(x=x_bump_start, color='r', linestyle='--', label='Speed bump start')
axs[0, 1].axvline(x=x_bump_end, color='r', linestyle='--', label='Speed bump end')
axs[0, 1].set_xlim(0, max(x_res))
axs[0, 1].set_ylim(y_min, y_max)
axs[0, 1].legend()
axs[0, 1].grid(True, linestyle=':', alpha=0.7)

# Heading angle plot
axs[1, 0].plot(x_res, theta_res, 'b-', linewidth=2, label='theta')
axs[1, 0].set_xlabel('x (m)')
axs[1, 0].set_ylabel('theta (rad)')
axs[1, 0].set_title('Heading Angle Profile')
axs[1, 0].axvline(x=x_bump_start, color='r', linestyle='--', label='Speed bump start')
axs[1, 0].axvline(x=x_bump_end, color='r', linestyle='--', label='Speed bump end')
axs[1, 0].set_xlim(0, max(x_res))
axs[1, 0].set_ylim(theta_min, theta_max)
axs[1, 0].legend()
axs[1, 0].grid(True, linestyle=':', alpha=0.7)

# Steering angle plot
axs[1, 1].plot(x_res, steering_res, 'b-', linewidth=2, label='steering')
axs[1, 1].set_xlabel('x (m)')
axs[1, 1].set_ylabel('steering (rad)')
axs[1, 1].set_title('Steering Angle Profile')
axs[1, 1].axvline(x=x_bump_start, color='r', linestyle='--', label='Speed bump start')
axs[1, 1].axvline(x=x_bump_end, color='r', linestyle='--', label='Speed bump end')
axs[1, 1].set_xlim(0, max(x_res))
axs[1, 1].set_ylim(steering_min, steering_max)
axs[1, 1].legend()
axs[1, 1].grid(True, linestyle=':', alpha=0.7)

# Acceleration plot
axs[2, 0].plot(x_res, accel_res, 'b-', linewidth=2, label='accel')
axs[2, 0].set_xlabel('x (m)')
axs[2, 0].set_ylabel('accel (m/s^2)')
axs[2, 0].set_title('Acceleration Profile')
axs[2, 0].axvline(x=x_bump_start, color='r', linestyle='--', label='Speed bump start')
axs[2, 0].axvline(x=x_bump_end, color='r', linestyle='--', label='Speed bump end')
axs[2, 0].set_xlim(0, max(x_res))
axs[2, 0].set_ylim(accel_min, accel_max)
axs[2, 0].legend()
axs[2, 0].grid(True, linestyle=':', alpha=0.7)

# Jerk plot
axs[2, 1].plot(x_res[:-1], jerk_res, 'b-', linewidth=2, label='jerk')
axs[2, 1].set_xlabel('x (m)')
axs[2, 1].set_ylabel('jerk (m/s^3)')
axs[2, 1].set_title('Jerk Profile')
axs[2, 1].axvline(x=x_bump_start, color='r', linestyle='--', label='Speed bump start')
axs[2, 1].axvline(x=x_bump_end, color='r', linestyle='--', label='Speed bump end')
axs[2, 1].set_xlim(0, max(x_res))
axs[2, 1].set_ylim(jerk_min, jerk_max)
axs[2, 1].legend()
axs[2, 1].grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.show()

print("Optimization status:", model.status)
print("Objective value:", model.objVal)