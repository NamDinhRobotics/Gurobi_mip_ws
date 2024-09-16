from gurobipy import Model, GRB, QuadExpr
import numpy as np

# Initialize the model
model = Model("nonlinear_optimization")

# Add continuous variables
x = model.addVar(name="x")
y = model.addVar(name="y")
# sinx
# sinx = model.addVar(name="sinx", lb=-1, ub=1)

# Define a nonlinear objective function
#model.addGenConstrSin(x, sinx, "sinx")
obj = QuadExpr()

# add sin(x)
# obj += sinx
# add x^2
obj += x * x + y * y + 2*x*y

model.setObjective(obj, GRB.MAXIMIZE)

# Add nonlinear constraints
constraint = 2*x * x + y * y <= 10
model.addConstr(constraint, name="constraint1")
# add x+y = 5
constraint2 = x + y >= 3
model.addConstr(constraint2, name="constraint2")

# Set parameters for nonlinear optimization
model.setParam('NonConvex', 2)  # Allow nonconvex problems

# Optimize
model.optimize()

# Retrieve and print results
if model.status == GRB.OPTIMAL:
    print(f"x: {x.X}")
    print(f"y: {y.X}")
