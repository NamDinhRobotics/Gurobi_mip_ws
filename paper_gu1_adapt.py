import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
from gurobipy import GRB

# Parameters
T = 20  # Prediction horizon (seconds)
dt = 0.1  # Time step (seconds)
N = int(T / dt)  # Number of time steps

# State and control bounds
x_min, x_max = 0, np.inf
y_min, y_max = 0, 2.
v_x_min, v_x_max = 0, 20
v_y_min, v_y_max = -1, 1
a_x_min, a_x_max = -2, 2
a_y_min, a_y_max = -0.5, 0.5
j_x_min, j_x_max = -2.5, 2.5
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
q1, q2, q3, q4, q5 = 1, 1, 1, 2, 4
r1, r2 = 40, 40

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

# steering angle
steer_max = np.pi/6

delta = model.addVars(N+1, lb=-steer_max, ub=steer_max, name="delta")

epsilon = 1e-6  # Small value for numerical stability

# Binary variables for speed bump logical constraints
delta1 = model.addVars(N+1, vtype=GRB.BINARY, name="delta1")
delta2 = model.addVars(N+1, vtype=GRB.BINARY, name="delta2")
delta3 = model.addVars(N+1, vtype=GRB.BINARY, name="delta3")

# binary variables for steer to left or right
delta4 = model.addVars(N+1, vtype=GRB.BINARY, name="delta4")

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

    # steering angle constraints steer = tan^-1(L * omega / v_x)--> steer = L vy/vx --> L * vy = vx * steer
    model.addConstr(v_y[k] * L == v_x[k] * delta[k])

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

    # δ4(k) = 1 ⇔ a_y(k) >= a_steer or a_y(k) <= -a_steer
    # a_steer_y = 0.1
    #model.addGenConstrIndicator(delta4[k], True, a_y[k] >= a_steer_y)
    bump_steer = np.pi/90  # 5 degrees steering during bump
    model.addGenConstrIndicator(delta4[k], True, delta[k] >= bump_steer)


    # Logical implications from equation 7
    model.addConstr(-delta1[k] + delta3[k] <= 0)
    model.addConstr(-delta2[k] + delta3[k] <= 0)
    model.addConstr(delta1[k] + delta2[k] - delta3[k] <= 1)

    # Logical implications from steering when in bump
    model.addConstr(delta1[k] + delta2[k] - delta4[k] <= 1)

# Objective function
obj = gp.QuadExpr()
for k in range(N+1):
    #obj += q1 * (v_x[k] - v_r)**2 + q2 * a_x[k]**2 + q3 * (y[k] - y_r)**2 + q4 * v_y[k]**2 + q5 * a_y[k]**2
    obj += q1 * (v_x[k] - v_r)**2 + q3 * (y[k] - y_r)**2 + q4 * v_y[k]**2 + q5 * a_y[k]**2

for k in range(N):
    obj += r1 * j_x[k]**2 + r2 * j_y[k]**2

model.setObjective(obj, GRB.MINIMIZE)

# Optimize the model
model.optimize()

# Extract results
x_res = [x[k].X for k in range(N+1)]
y_res = [y[k].X for k in range(N+1)]
v_x_res = [v_x[k].X for k in range(N+1)]
v_y_res = [v_y[k].X for k in range(N+1)]
a_x_res = [a_x[k].X for k in range(N+1)]
a_y_res = [a_y[k].X for k in range(N+1)]
j_x_res = [j_x[k].X for k in range(N)]
j_y_res = [j_y[k].X for k in range(N)]

# Extract steering angles
delta_res = [delta[k].X for k in range(N+1)]

# compute sum of squared jerk
jerk_sum = sum(j_x_res[k]**2 + j_y_res[k]**2 for k in range(N))
print("Sum of squared jerk:", jerk_sum)

# Create the plots
fig, axs = plt.subplots(3, 2, figsize=(15, 10))

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
axs[0, 0].legend()
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

# Vehicle Trajectory plot with heading angles
axs[2, 0].plot(x_res, y_res, 'b-', linewidth=2, label='Vehicle Path')
axs[2, 0].set_xlabel('x (m)')
axs[2, 0].set_ylabel('y (m)')
axs[2, 0].set_title('Vehicle Trajectory with Heading Angles')
axs[2, 0].axvline(x=x_bump_start, color='r', linestyle='--', label='Speed bump start')
axs[2, 0].axvline(x=x_bump_end, color='r', linestyle='--', label='Speed bump end')
axs[2, 0].set_xlim(0, max(x_res))
axs[2, 0].set_ylim(y_min, y_max)

# Plot arrows to indicate heading angle
arrow_spacing = max(1, N // 20)  # Adjust this value to change arrow density
for i in range(0, N+1, arrow_spacing):
    axs[2, 0].arrow(x_res[i], y_res[i],
                    np.cos(delta_res[i]), np.sin(delta_res[i]),
                    head_width=0.1, head_length=0.5, fc='r', ec='r')

axs[2, 0].legend()
axs[2, 0].grid(True, linestyle=':', alpha=0.7)

# Steering angle plot
axs[2, 1].plot(x_res, delta_res, 'b-', linewidth=2, label='Steering Angle')
axs[2, 1].set_xlabel('x (m)')
axs[2, 1].set_ylabel('delta (rad)')
axs[2, 1].set_title('Steering Angle Profile')
axs[2, 1].axvline(x=x_bump_start, color='r', linestyle='--', label='Speed bump start')
axs[2, 1].axvline(x=x_bump_end, color='r', linestyle='--', label='Speed bump end')
axs[2, 1].set_xlim(0, max(x_res))
axs[2, 1].set_ylim(-steer_max, steer_max)
axs[2, 1].legend()
axs[2, 1].grid(True, linestyle=':', alpha=0.7)

# Adjust layout for better visibility
plt.tight_layout()
plt.show()

print("Optimization status:", model.status)
print("Objective value:", model.objVal)


print("Optimization status:", model.status)
print("Objective value:", model.objVal)