import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
from gurobipy import GRB
from matplotlib.lines import Line2D

# Parameters
T = 25  # Prediction horizon (seconds)
dt = 0.3  # Time step (seconds)
N = int(T / dt)  # Number of time steps

# Vehicle state bounds
x_min, x_max = 0, np.inf
y_min, y_max = 0.2, 6  # Road width is 10 meters (2 lanes)
v_x_min, v_x_max = 0, 10
v_y_min, v_y_max = -1, 1
a_x_min, a_x_max = -3, 3
a_y_min, a_y_max = -0.5, 0.5
j_x_min, j_x_max = -2.5, 2.5
j_y_min, j_y_max = -0.5, 0.5

# Initial state and reference values
x0, y0 = 5, 3.5  # Starting in the middle of the right lane
v_x0, v_y0 = 5, 0  # Initial speed is 15 m/s
a_x0, a_y0 = 0, 0
v_r = 5.0  # Reference speed
y_r = y0  # Reference lateral position (stay in lane)

# Surrounding vehicles (x_initial, y, speed)
same_lane_vehicles = [
    {'x': 20, 'y': 3.5, 'speed': 5}
]
oncoming_vehicles = [
    {'x': 122, 'y': 2.5, 'speed': -5},
    {'x': 90, 'y': 3.5, 'speed': -8},
    {'x': 111, 'y': 4.5, 'speed': -8}
]

# Safety distances
L_v = 15  # Reduced from 20
W_v = 1.5  # Reduced from 2

# Cost function weights
q1, q2, q3, q4, q5 = 2, 1, 10, 1, 2  # Increased speed priority, decreased lane deviation penalty
r1, r2 = 10, 10

# Create the model
model = gp.Model("Overtaking_MIQP")

# Create variables
x = model.addVars(N + 1, lb=x_min, ub=x_max, name="x")
y = model.addVars(N + 1, lb=y_min, ub=y_max, name="y")
v_x = model.addVars(N + 1, lb=v_x_min, ub=v_x_max, name="v_x")
v_y = model.addVars(N + 1, lb=v_y_min, ub=v_y_max, name="v_y")
a_x = model.addVars(N + 1, lb=a_x_min, ub=a_x_max, name="a_x")
a_y = model.addVars(N + 1, lb=a_y_min, ub=a_y_max, name="a_y")
j_x = model.addVars(N, lb=j_x_min, ub=j_x_max, name="j_x")
j_y = model.addVars(N, lb=j_y_min, ub=j_y_max, name="j_y")

# Binary variables for vehicle avoidance
delta_v = {}
for v_id in range(len(same_lane_vehicles) + len(oncoming_vehicles)):
    delta_v[v_id] = model.addVars(N + 1, vtype=GRB.BINARY, name=f"delta_v_{v_id}")

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

# Vehicle avoidance constraints using ramp barrier method
for k in range(N + 1):
    # Same lane vehicles
    for v_id, vehicle in enumerate(same_lane_vehicles):
        x_v = vehicle['x'] + vehicle['speed'] * k * dt
        y_v = vehicle['y']
        model.addGenConstrIndicator(delta_v[v_id][k], False,
                                    -((x[k] - x_v) / L_v) + ((y[k] - y_v) / W_v) >= 1)
        model.addGenConstrIndicator(delta_v[v_id][k], True,
                                    ((x[k] - x_v) / L_v) + ((y[k] - y_v) / W_v) >= 1)

    # Oncoming vehicles
    for v_id, vehicle in enumerate(oncoming_vehicles, start=len(same_lane_vehicles)):
        x_v = vehicle['x'] + vehicle['speed'] * k * dt
        y_v = vehicle['y']
        model.addGenConstrIndicator(delta_v[v_id][k], False,
                                    ((x[k] - x_v) / L_v) + ((y[k] - y_v) / W_v) <= -1)
        model.addGenConstrIndicator(delta_v[v_id][k], True,
                                    -((x[k] - x_v) / L_v) + ((y[k] - y_v) / W_v) <= -1)

# Ensure the ego vehicle returns to the original lane
model.addConstr(y[N] == y_r)

# Objective function
obj = gp.QuadExpr()
for k in range(N + 1):
    obj += (q1 * (v_x[k] - v_r) ** 2 + q2 * a_x[k] ** 2 + q3 * (y[k] - y_r) ** 2 +
            + q4 * v_y[k] ** 2 + q5 * a_y[k] ** 2)
for k in range(N):
    obj += r1 * j_x[k] ** 2 + r2 * j_y[k] ** 2

# punishment for


# Add time-dependent cost
#time_weight = 0.1
#for k in range(N):
#    obj += time_weight * k  # Penalize later time steps more

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

####################################################################################################
import pygame
import sys
import math

# Initialize Pygame
pygame.init()

# Screen setup
width, height = 800, 400
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Overtaking Scenario")

# Colors
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
GRAY = (200, 200, 200)
LIGHT_BLUE = (173, 216, 230)
LIGHT_RED = (255, 204, 203)
LIGHT_GREEN = (144, 238, 144)

# Scale factors
scale_x = width / (max(x_res) + 10)
scale_y = height / 10

# Car dimensions
car_length = 40
car_width = 20


def draw_car(surface, x, y, angle, color):
    car = pygame.Surface((car_length, car_width), pygame.SRCALPHA)
    pygame.draw.rect(car, color, (0, 0, car_length, car_width))
    pygame.draw.polygon(car, color, [(car_length, car_width / 2), (car_length - 10, 0), (car_length - 10, car_width)])
    rotated_car = pygame.transform.rotate(car, -angle)
    rect = rotated_car.get_rect(center=(x, y))
    surface.blit(rotated_car, rect)


def calculate_heading(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    d = math.degrees(math.atan2(dy, dx))
    # norm -pi to pi
    if d < -180:
        d += 360
    if d > 180:
        d -= 360
    return d


# Initialize trail lists
ego_trail = []
same_lane_trails = [[] for _ in same_lane_vehicles]
oncoming_trails = [[] for _ in oncoming_vehicles]

# Main simulation loop
clock = pygame.time.Clock()
frame = 0

while frame < len(x_res) - 1:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    screen.fill(WHITE)

    # Draw road
    pygame.draw.line(screen, GRAY, (0, height / 2), (width, height / 2), 2)

    # Draw trails
    for pos in ego_trail:
        pygame.draw.circle(screen, LIGHT_BLUE, pos, 2)

    for trail in same_lane_trails:
        for pos in trail:
            pygame.draw.circle(screen, LIGHT_RED, pos, 2)

    for trail in oncoming_trails:
        for pos in trail:
            pygame.draw.circle(screen, LIGHT_GREEN, pos, 2)

    # Draw ego vehicle
    ego_x = int(x_res[frame] * scale_x)
    ego_y = int(height - y_res[frame] * scale_y)
    next_ego_x = int(x_res[frame + 1] * scale_x)
    next_ego_y = int(height - y_res[frame + 1] * scale_y)
    ego_angle = calculate_heading(ego_x, ego_y, next_ego_x, next_ego_y)
    draw_car(screen, ego_x, ego_y, ego_angle, BLUE)
    ego_trail.append((ego_x, ego_y))

    # Draw same lane vehicles
    for i, vehicle in enumerate(same_lane_vehicles):
        x = int((vehicle['x'] + vehicle['speed'] * frame * dt) * scale_x)
        y = int(height - vehicle['y'] * scale_y)
        next_x = int((vehicle['x'] + vehicle['speed'] * (frame + 1) * dt) * scale_x)
        angle = calculate_heading(x, y, next_x, y)
        draw_car(screen, x, y, angle, RED)
        same_lane_trails[i].append((x, y))

    # Draw oncoming vehicles
    for i, vehicle in enumerate(oncoming_vehicles):
        x = int((vehicle['x'] + vehicle['speed'] * frame * dt) * scale_x)
        y = int(height - vehicle['y'] * scale_y)
        next_x = int((vehicle['x'] + vehicle['speed'] * (frame + 1) * dt) * scale_x)
        angle = calculate_heading(x, y, next_x, y)
        draw_car(screen, x, y, angle, GREEN)
        oncoming_trails[i].append((x, y))

    # Display frame number
    font = pygame.font.Font(None, 36)
    text = font.render(f"Frame: {frame}", True, (0, 0, 0))
    screen.blit(text, (10, 10))

    pygame.display.flip()
    clock.tick(10)  # Adjust for desired frame rate
    frame += 1

pygame.quit()

####################################################################################################
fig, axs = plt.subplots(3, 2, figsize=(15, 20))

# 1. Vehicle trajectory and other vehicles
axs[0, 0].set_xlabel('x (m)')
axs[0, 0].set_ylabel('y (m)')
axs[0, 0].set_title('Vehicle Trajectory with Other Vehicles')

# Tạo color maps cho mỗi loại xe
ego_cmap = plt.get_cmap('Blues')
same_lane_cmap = plt.get_cmap('Reds')
oncoming_cmap = plt.get_cmap('Oranges')

# Plot ego vehicle trajectory
for k in range(N + 1):
    color = ego_cmap(1 - k / N)
    axs[0, 0].plot(x_res[k], y_res[k], 's', color=color, markersize=8)
    axs[0, 0].annotate(str(k), (x_res[k], y_res[k]), xytext=(3, 3),
                       textcoords='offset points', fontsize=8, color='blue')

# Plot other vehicles
for vehicle_id, vehicle in enumerate(same_lane_vehicles):
    x_v = [vehicle['x'] + vehicle['speed'] * k * dt for k in range(N + 1)]
    y_v = [vehicle['y']] * (N + 1)
    for k in range(N + 1):
        color = same_lane_cmap(1 - k / N)
        axs[0, 0].plot(x_v[k], y_v[k], '>', color=color, markersize=6)
        axs[0, 0].annotate(str(k), (x_v[k], y_v[k]), xytext=(3, 3),
                           textcoords='offset points', fontsize=8, color='red')

for vehicle_id, vehicle in enumerate(oncoming_vehicles):
    x_v = [vehicle['x'] + vehicle['speed'] * k * dt for k in range(N + 1)]
    y_v = [vehicle['y']] * (N + 1)
    for k in range(N + 1):
        color = oncoming_cmap(1 - k / N)
        axs[0, 0].plot(x_v[k], y_v[k], '<', color=color, markersize=6)
        axs[0, 0].annotate(str(k), (x_v[k], y_v[k]), xytext=(3, 3),
                           textcoords='offset points', fontsize=8, color='orange')

axs[0, 0].set_xlim(0, max(x_res) + 10)
axs[0, 0].set_ylim(0, y_max)

# Tạo legend với gradient color
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Ego Vehicle',
           markerfacecolor=ego_cmap(0.5), markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Same Lane Vehicles',
           markerfacecolor=same_lane_cmap(0.5), markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Oncoming Vehicles',
           markerfacecolor=oncoming_cmap(0.5), markersize=10)
]
axs[0, 0].legend(handles=legend_elements)

axs[0, 0].grid(True)

# Add lane markings
axs[0, 0].axhline(y=5, color='k', linestyle='--')
axs[0, 0].axhline(y=0, color='k', linestyle='-')
axs[0, 0].axhline(y=10, color='k', linestyle='-')

# 2. Speed profiles
axs[0, 1].plot(x_res, v_x_res, 'b-', label='v_x')
axs[0, 1].plot(x_res, v_y_res, 'r-', label='v_y')
axs[0, 1].set_xlabel('x (m)')
axs[0, 1].set_ylabel('Speed (m/s)')
axs[0, 1].set_title('Speed Profiles')
axs[0, 1].legend()
axs[0, 1].grid(True)

# 3. Acceleration profiles
axs[1, 0].plot(x_res, a_x_res, 'b-', label='a_x')
axs[1, 0].plot(x_res, a_y_res, 'r-', label='a_y')
axs[1, 0].set_xlabel('x (m)')
axs[1, 0].set_ylabel('Acceleration (m/s²)')
axs[1, 0].set_title('Acceleration Profiles')
axs[1, 0].legend()
axs[1, 0].grid(True)

# 4. Jerk profiles
axs[1, 1].step(x_res[:-1], j_x_res, 'b-', label='j_x', where='post')
axs[1, 1].step(x_res[:-1], j_y_res, 'r-', label='j_y', where='post')
axs[1, 1].set_xlabel('x (m)')
axs[1, 1].set_ylabel('Jerk (m/s³)')
axs[1, 1].set_title('Jerk Profiles')
axs[1, 1].legend()
axs[1, 1].grid(True)

# 5. Steering angle
steering_angle = [np.arctan2(v_y_res[i], v_x_res[i]) for i in range(len(v_x_res))]
axs[2, 0].plot(x_res, steering_angle, 'g-')
axs[2, 0].set_xlabel('x (m)')
axs[2, 0].set_ylabel('Steering Angle (rad)')
axs[2, 0].set_title('Steering Angle Profile')
axs[2, 0].grid(True)

# 6. Yaw rate
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
