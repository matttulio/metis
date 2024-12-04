from src.datagen import DifferentialEquations
from jax import random
import jax.numpy as jnp
import numpy as np
import jax.nn as nn
import matplotlib.pyplot as plt
import scienceplots

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

system = DifferentialEquations(n_vars=5, n_eqs=3, max_sum_terms=3, max_mult_terms=3, non_lins=[non_linearity_1, non_linearity_2, non_linearity_3], sym_non_lins=non_lin_syms)
print(system.equations, system.sym_expr)

system.show_equations(save=True, filename="Data/expression.pdf")