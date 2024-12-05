from src.datagen import Equations
from jax import random
import jax.numpy as jnp
import numpy as np
import jax.nn as nn
import matplotlib.pyplot as plt
import scienceplots
from scipy.integrate import RK45
from tqdm import tqdm
import os

# Define the theta parameters for the three specified non-linearities
theta_1 = jnp.array([10, -10, 1.0, 1.0, 0.0, 0.1])  # parameters for the saturation function
theta_2 = jnp.array([1.0, -1.0, 1.0, -1.0, 0.0, 0.0])  # parameters for the identity function
theta_3 = jnp.array([1.0, 2.0, -1.0, -5.0, -2.0, -1.0])  # parameters for the inverse function
theta_list = [theta_1, theta_2, theta_3]

# Define list of symbolic expressions of non-linearities
non_lin_syms = []

# Definition for the saturation function parametrized with ReLUs
def non_linearity_1(x):
    alpha1, alpha2, gamma1, gamma2, beta1, beta2 = theta_1
    return (alpha1 * nn.relu(gamma1 * x - beta1) + alpha2 * nn.relu(gamma2 * x - beta2))

#non_lin1_sym = f"{theta_1[0]} \\cdot \\text{{ReLU}}({theta_1[2]} x - {theta_1[4]}) + {theta_1[1]} \\cdot \\text{{ReLU}}({theta_1[3]} x - {theta_1[5]})"
non_lin1_sym = "\\text{{Sat}}"
non_lin_syms.append(non_lin1_sym)

# Definition for the identity function paramterized with ReLUs
def non_linearity_2(x):
    alpha1, alpha2, gamma1, gamma2, beta1, beta2 = theta_2
    return (alpha1 * nn.relu(gamma1 * x - beta1) + alpha2 * nn.relu(gamma2 * x - beta2))

#non_lin2_sym = f"{theta_2[0]} \\cdot \\text{{ReLU}}({theta_2[2]} x - {theta_2[4]}) + {theta_2[1]} \\cdot \\text{{ReLU}}({theta_2[3]} x - {theta_2[5]})"
non_lin2_sym = "\\text{{ID}}"
non_lin_syms.append(non_lin2_sym)

# Definition for the inverse function parametrized with ReLUs
def non_linearity_3(x):
    alpha1, alpha2, gamma1, gamma2, beta1, beta2 = theta_3
    return (alpha1 * nn.relu(gamma1 * x - beta1) + alpha2 * nn.relu(gamma2 * x - beta2))

#non_lin3_sym = f"{theta_3[0]} \\cdot \\text{{ReLU}}({theta_3[2]} x - {theta_3[4]}) + {theta_3[1]} \\cdot \\text{{ReLU}}({theta_3[3]} x - {theta_3[5]})"
non_lin3_sym = "\\text{{Inv}}"
non_lin_syms.append(non_lin3_sym)

# Define hyperameters for the Equations class
config = {
    "n_vars": 10,
    "n_eqs": 10,
    "max_sum_terms": 3,
    "max_mult_terms": 3,
    "non_lins": [non_linearity_1, non_linearity_2, non_linearity_3],
    "sym_non_lins": non_lin_syms,
    "seed": 42
}

# Create the system of equations
system = Equations(**config)

# Show the created equations
save_dir = "Data/"
os.makedirs(save_dir, exist_ok=True)
system.show_equations(save=True, filename=os.path.join(save_dir,"expression.pdf"))

# Initial random conditions
seed = config["seed"]
key = random.key(seed)
y0 = random.uniform(key, shape=(config["n_vars"],), minval=-1, maxval=2)  # Initial values of ys
step = 0.1
t_final = 10
t0 = 0

# Initialize the solver
solver = RK45(system, t0=t0, y0=y0, t_bound=t_final, max_step=step)

t_values = []
y_values = []
n_steps = int((t_final - t0) / step)

# Solve the system
with tqdm(total=n_steps, desc="Integrating System") as pbar:
    while solver.status == 'running':
        solver.step()
        t_values.append(solver.t)
        y_values.append(solver.y)
        pbar.update(1)

y_values = np.array(y_values)

# Plot the solutions as a function of time
plt.figure(figsize=(16, 8))
for i in range(config["n_eqs"]):
    plt.plot(t_values, y_values[:, i])
plt.xlabel("Time t")
plt.ylabel("y(t)")
plt.title("Solution of System of ODEs using RK45")
#plt.legend()
plt.grid(True)
plt.show()
plt.clf()
plt.close()