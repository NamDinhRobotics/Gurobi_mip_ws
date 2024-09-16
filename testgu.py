import gurobipy as gp
from gurobipy import GRB
import math
import time
# Plot the optimized trajectory
import matplotlib.pyplot as plt

try:
    # Create a Gurobi environment and set parameters
    #env = gp.Env(empty=True)
    #env.setParam('MIPFocus', 0)
    #env.start()

    # Prediction horizon and other constants
    N = 10
    T = 0.1
    L = 1.5

    # Weights for the cost function
    Q = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0.1, 0], [0, 0, 0, 0.1]]
    R = [[0.1, 0], [0, 0.1]]
    Q_f = [[10, 0, 0, 0], [0, 10, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    # Initial and target state
    x_init = [0, 0, math.pi / 4, 0]
    x_target = [10, 10, math.pi / 4, 0]

    # Control limits
    steer_max = math.pi / 4
    steer_min = -math.pi / 4
    a_max = 3.0
    a_min = -3.0

    # Create the model
    # model = gp.Model("PathPlanning", env=env)
    model = gp.Model("PathPlanning")
    # Define state and control variables
    x_vars = model.addVars(N + 1, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")
    y_vars = model.addVars(N + 1, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="y")
    theta_vars = model.addVars(N + 1, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="theta")
    v_vars = model.addVars(N + 1, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="v")

    steer_vars = model.addVars(N, lb=steer_min, ub=steer_max, name="steer")
    a_vars = model.addVars(N, lb=a_min, ub=a_max, name="a")

    cos_theta_vars = model.addVars(N, lb=-1, ub=1, name="cos_theta")
    sin_theta_vars = model.addVars(N, lb=-1, ub=1, name="sin_theta")
    tan_steer_vars = model.addVars(N, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="tan_steer")

    model.update()

    # Set initial state constraints
    model.addConstr(x_vars[0] == x_init[0])
    model.addConstr(y_vars[0] == x_init[1])
    model.addConstr(theta_vars[0] == x_init[2])
    model.addConstr(v_vars[0] == x_init[3])

    # Dynamics constraints, trigonometric constraints, and cost function
    obj = gp.QuadExpr()
    for k in range(N):
        model.addGenConstrCos(theta_vars[k], cos_theta_vars[k], "cos_theta_" + str(k))
        model.addGenConstrSin(theta_vars[k], sin_theta_vars[k], "sin_theta_" + str(k))
        model.addGenConstrTan(steer_vars[k], tan_steer_vars[k], "tan_steer_" + str(k))

        # Dynamics constraints using sin, cos, and tan variables
        model.addConstr(x_vars[k + 1] == x_vars[k] + T * v_vars[k] * cos_theta_vars[k])
        model.addConstr(y_vars[k + 1] == y_vars[k] + T * v_vars[k] * sin_theta_vars[k])
        model.addConstr(theta_vars[k + 1] == theta_vars[k] + T * (v_vars[k] / L) * tan_steer_vars[k])
        model.addConstr(v_vars[k + 1] == v_vars[k] + T * a_vars[k])

        # Quadratic cost function for states and controls
        obj += (x_vars[k] ** 2 * Q[0][0] + y_vars[k] ** 2 * Q[1][1] +
                theta_vars[k] ** 2 * Q[2][2] + v_vars[k] ** 2 * Q[3][3] +
                steer_vars[k] ** 2 * R[0][0] + a_vars[k] ** 2 * R[1][1])

    # Terminal cost
    obj += ((x_vars[N] - x_target[0]) ** 2 * Q_f[0][0] +
            (y_vars[N] - x_target[1]) ** 2 * Q_f[1][1] +
            (theta_vars[N] - x_target[2]) ** 2 * Q_f[2][2] +
            (v_vars[N] - x_target[3]) ** 2 * Q_f[3][3])

    # Allow nonconvex problems optimization not a MIP
    # Set parameters for nonlinear optimization
    model.setParam('NonConvex', 2)  # Allow nonconvex problems



    # Set the objective
    model.setObjective(obj, GRB.MINIMIZE)

    # Optimize the model
    start_time = time.time()
    model.optimize()
    end_time = time.time()

    print(f"Optimization time: {end_time - start_time} seconds")

    # Output the results
    if model.status == GRB.OPTIMAL:
        print("Optimal solution found!")
        for k in range(N + 1):
            print(f"State at step {k}: ({x_vars[k].X}, {y_vars[k].X}, {theta_vars[k].X}, {v_vars[k].X})")
    else:
        print("No optimal solution found.")

except gp.GurobiError as e:
    print(f"Error code = {e.errno}")
    print(e.message)
except Exception as e:
    print("Exception during optimization.")
    print(e)

# Plot the optimized trajectory
x = [x_vars[k].X for k in range(N + 1)]
y = [y_vars[k].X for k in range(N + 1)]
plt.plot(x, y, 'ro-')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Optimized Trajectory')
plt.show()
