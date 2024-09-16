import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
from gurobipy import GRB

# Parameters
T = 30  # Prediction horizon (seconds)
dt = 0.2  # Time step (seconds)
N = int(T / dt)  # Number of time steps

# Vehicle state bounds
x_min, x_max = 0, np.inf
y_min, y_max = 0, 5  # Road width is 5 meters
v_x_min, v_x_max = 0, 20
v_y_min, v_y_max = -1, 1
a_x_min, a_x_max = -2, 2
a_y_min, a_y_max = -0.5, 0.5
j_x_min, j_x_max = -2.5, 2.5
j_y_min, j_y_max = -1, 1

# Initial state and reference values
x0, y0 = 0, 2.5  # Starting in the middle of the right lane
v_x0, v_y0 = 10, 0  # Initial speed is 15 m/s
a_x0, a_y0 = 0, 0
v_r = 10.0  # Reference speed
y_r = y0  # Reference lateral position (stay in lane)

# Obstacle parameters (bounding rectangles)
obstacles = [
    {'x_center': 80, 'y_center': 1.5, 'L': 10, 'W': 2},
    {'x_center': 200, 'y_center': 3.5, 'L': 10, 'W': 3}
]

# Cost function weights
q1, q2, q3, q4, q5 = 4, 1, 1, 1, 1
r1, r2 = 40, 40

# Create the model
model = gp.Model("Obstacle_Avoidance_MIQP")

# Create variables
x = model.addVars(N + 1, lb=x_min, ub=x_max, name="x")
y = model.addVars(N + 1, lb=y_min, ub=y_max, name="y")
v_x = model.addVars(N + 1, lb=v_x_min, ub=v_x_max, name="v_x")
v_y = model.addVars(N + 1, lb=v_y_min, ub=v_y_max, name="v_y")
a_x = model.addVars(N + 1, lb=a_x_min, ub=a_x_max, name="a_x")
a_y = model.addVars(N + 1, lb=a_y_min, ub=a_y_max, name="a_y")
j_x = model.addVars(N, lb=j_x_min, ub=j_x_max, name="j_x")
j_y = model.addVars(N, lb=j_y_min, ub=j_y_max, name="j_y")

# Binary variables for obstacle avoidance
delta_obs = {}
for obs_id, obs in enumerate(obstacles):
    for i in range(4):  # 4 ràng buộc cho mỗi chướng ngại vật
        delta_obs[obs_id, i] = model.addVars(N + 1, vtype=GRB.BINARY, name=f"delta_obs_{obs_id}_{i}")

# Set initial conditions
model.addConstr(x[0] == x0)
model.addConstr(y[0] == y0)
model.addConstr(v_x[0] == v_x0)
model.addConstr(v_y[0] == v_y0)
model.addConstr(a_x[0] == a_x0)
model.addConstr(a_y[0] == a_y0)

# Add vehicle dynamics constraints
for k in range(N):
    model.addConstr(x[k + 1] == x[k] + v_x[k] * dt + 0.5 * a_x[k] * dt ** 2 + (1 / 6) * j_x[k] * dt ** 3)
    model.addConstr(y[k + 1] == y[k] + v_y[k] * dt + 0.5 * a_y[k] * dt ** 2 + (1 / 6) * j_y[k] * dt ** 3)
    model.addConstr(v_x[k + 1] == v_x[k] + a_x[k] * dt + 0.5 * j_x[k] * dt ** 2)
    model.addConstr(v_y[k + 1] == v_y[k] + a_y[k] * dt + 0.5 * j_y[k] * dt ** 2)
    model.addConstr(a_x[k + 1] == a_x[k] + j_x[k] * dt)
    model.addConstr(a_y[k + 1] == a_y[k] + j_y[k] * dt)

# Obstacle avoidance constraints (Equation 11 from the paper)
# Thay đổi ràng buộc tránh va chạm chướng ngại vật (Phương trình 11 từ bài báo)
for obs_id, obs in enumerate(obstacles):
    x_obs, y_obs = obs['x_center'], obs['y_center']
    L, W = obs['L'], obs['W']

    for k in range(N + 1):
        # Phương trình 11a
        model.addGenConstrIndicator(delta_obs[obs_id, 0][k], True, x[k] <= x_obs - L)
        # Phương trình 11b
        model.addGenConstrIndicator(delta_obs[obs_id, 1][k], True, x[k] >= x_obs + L)
        # Phương trình 11c
        model.addGenConstrIndicator(delta_obs[obs_id, 2][k], True, y[k] <= y_obs - W)
        # Phương trình 11d
        model.addGenConstrIndicator(delta_obs[obs_id, 3][k], True, y[k] >= y_obs + W)

        # Phương trình 11e
        model.addConstr(delta_obs[obs_id, 0][k] + delta_obs[obs_id, 1][k] + delta_obs[obs_id, 2][k] + delta_obs[obs_id, 3][k] == 1)

# Objective function: Minimize cost of deviation from reference trajectory and control efforts
obj = gp.QuadExpr()
for k in range(N + 1):
    obj += q1 * (v_x[k] - v_r) ** 2 + q2 * a_x[k] ** 2 + q3 * (y[k] - y_r) ** 2 + q4 * v_y[k] ** 2 + q5 * a_y[k] ** 2
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
# Sau khi trích xuất kết quả, thêm đoạn mã sau để vẽ các biểu đồ bổ sung

# Tạo lưới biểu đồ
fig, axs = plt.subplots(3, 2, figsize=(15, 20))

# 1. Quỹ đạo xe và chướng ngại vật
axs[0, 0].plot(x_res, y_res, 'b-', linewidth=2, label='Vehicle Trajectory')
axs[0, 0].set_xlabel('x (m)')
axs[0, 0].set_ylabel('y (m)')
axs[0, 0].set_title('Vehicle Trajectory with Obstacles')
# Trong hàm vẽ quỹ đạo xe (axs[0, 0])
for obs in obstacles:
    rect = plt.Rectangle((obs['x_center'] - obs['L'], obs['y_center'] - obs['W']),
                         2 * obs['L'], 2 * obs['W'], fill=True, facecolor='red', alpha=0.5)
    axs[0, 0].add_patch(rect)

axs[0, 0].set_xlim(0, max(x_res) + 10)
axs[0, 0].set_ylim(0, 5)
axs[0, 0].legend()
axs[0, 0].grid(True)

# Thêm chú thích cho chướng ngại vật
axs[0, 0].text(80, 1.5, 'Obstacle 1', ha='center', va='center')
axs[0, 0].text(160, 3.5, 'Obstacle 2', ha='center', va='center')

# 2. Biểu đồ vận tốc
axs[0, 1].plot(x_res, v_x_res, 'b-', label='v_x')
axs[0, 1].plot(x_res, v_y_res, 'r-', label='v_y')
axs[0, 1].set_xlabel('x (m)')
axs[0, 1].set_ylabel('Speed (m/s)')
axs[0, 1].set_title('Speed Profiles')
axs[0, 1].legend()
axs[0, 1].grid(True)

# 3. Biểu đồ gia tốc
axs[1, 0].plot(x_res, a_x_res, 'b-', label='a_x')
axs[1, 0].plot(x_res, a_y_res, 'r-', label='a_y')
axs[1, 0].set_xlabel('x (m)')
axs[1, 0].set_ylabel('Acceleration (m/s²)')
axs[1, 0].set_title('Acceleration Profiles')
axs[1, 0].legend()
axs[1, 0].grid(True)

# 4. Biểu đồ jerk
axs[1, 1].step(x_res[:-1], j_x_res, 'b-', label='j_x', where='post')
axs[1, 1].step(x_res[:-1], j_y_res, 'r-', label='j_y', where='post')
axs[1, 1].set_xlabel('x (m)')
axs[1, 1].set_ylabel('Jerk (m/s³)')
axs[1, 1].set_title('Jerk Profiles')
axs[1, 1].legend()
axs[1, 1].grid(True)

# 5. Góc lái (được tính từ v_y và v_x)
steering_angle = [np.arctan2(v_y_res[i], v_x_res[i]) for i in range(len(v_x_res))]
axs[2, 0].plot(x_res, steering_angle, 'g-')
axs[2, 0].set_xlabel('x (m)')
axs[2, 0].set_ylabel('Steering Angle (rad)')
axs[2, 0].set_title('Steering Angle Profile')
axs[2, 0].grid(True)

# 6. Tốc độ góc (được tính từ a_y và v_x)
yaw_rate = [a_y_res[i] / v_x_res[i] if v_x_res[i] != 0 else 0 for i in range(len(v_x_res))]
axs[2, 1].plot(x_res, yaw_rate, 'm-')
axs[2, 1].set_xlabel('x (m)')
axs[2, 1].set_ylabel('Yaw Rate (rad/s)')
axs[2, 1].set_title('Yaw Rate Profile')
axs[2, 1].grid(True)

plt.tight_layout()
plt.show()

print("Optimization status:", model.status)
print("Objective value:", model.objVal)
