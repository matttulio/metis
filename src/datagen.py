from jax import random
import jax.numpy as jnp
import numpy as np
import jax.nn as nn


class DifferentialEquations:

    def __init__(self, n_vars, n_eqs, max_sum_terms, max_mult_terms, non_lins, sym_non_lins = None, seed = 42):
        """
            Initialize differential equations generator.

            Parameters:
            n_vars: number of variables of the system;
            n_eqs: number of equations to generate;
            max_sum_terms: max number of addends in each equation;
            max_mult_terms: max number of multiplicands in each addend;
            non_lins: list of all the possible non linearities;
            sym_non_lins: list of the symbolic expression of the non linearities.
        """

        # Initialize variables
        self.n_vars = n_vars
        self.n_eqs = n_eqs
        self.max_sum_terms = max_sum_terms
        self.max_mult_terms = max_mult_terms
        self.non_lins = non_lins
        self.sym_non_lins = sym_non_lins

        # Initialize lists for storing the equations
        # and their symbolic expression
        self.equations = []
        self.sym_expr = []

        self.key = random.key(seed)

        # Random generation of the number of addends to have in each equation
        n_sum_terms = random.randint(self.key, shape=(self.n_eqs,), minval=1, maxval=self.max_sum_terms+1)
        n_mult_terms = []

        for i in range(self.n_eqs):

            equation = []

            # Loop on every addend in the i-th equation
            for _ in range(n_sum_terms[i]):

                self.key, subkey = random.split(self.key)
                n_mult_terms = random.randint(subkey, shape=(1,), minval=1, maxval=self.max_mult_terms+1)  # Draw how many multiplicands for the i-th addend
                non_lins_idxs = random.randint(subkey, shape=(n_mult_terms[0],), minval=0, maxval=len(self.non_lins))  # Draw a number of n_mult_terms of nls with replacement
                var_to_be_applied_idxs = random.randint(subkey, shape=(n_mult_terms[0],), minval=0, maxval=self.n_eqs)  # Draw to which variable apply the non-linearities

                # Create an array of the length equal to the number of variables
                # such that each position represent the variable to which apply
                # the non-linearity/ies
                addend = np.ones(self.n_vars).tolist()

                for j in list(set(var_to_be_applied_idxs.tolist())):
                    idxs = non_lins_idxs[np.where(var_to_be_applied_idxs == j)]
                    temp = []

                    for k in idxs:
                        temp.append(self.non_lins[k])

                    addend[j] = temp
                
                equation.append(addend)

            self.equations.append(equation)

    
    def non_linearity_symbol(self, nl):
        """
            Define the LaTeX symbolic expression for each non-linearity.

            Parameters:
            nl: proposed non linearity.

            Return:
            LaTeX symbolic expression of a non-linearity
        """

        # If the symbols are not defined a priori, use generic names
        if(self.sym_non_lins == None):

            index = self.non_lins.index(nl) if nl in self.non_lins else None  # Check if the non linearity candidate is in the list
            return f"\\text{{nl}}_{index}" if index is not None else "\\text{unknown\\_non\\_linearity}"
            
        else:

            index = self.non_lins.index(nl) if nl in self.non_lins else None
            return self.sym_expr[index] if index is not None else "\\text{unknown\\_non\\_linearity}"
        

    def __getitem__(self):
        return self.equations


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

non_lin1_sym = f"{theta_1[0]} \\cdot \\text{{ReLU}}({theta_1[2]} x - {theta_1[4]}) + {theta_1[1]} \\cdot \\text{{ReLU}}({theta_1[3]} x - {theta_1[5]})"
non_lin_syms.append(non_lin1_sym)

# Definition for the identity function paramterized with ReLUs
def non_linearity_2(x):
    alpha1, alpha2, gamma1, gamma2, beta1, beta2 = theta_2
    return (alpha1 * nn.relu(gamma1 * x - beta1) + alpha2 * nn.relu(gamma2 * x - beta2))

non_lin2_sym = f"{theta_2[0]} \\cdot \\text{{ReLU}}({theta_2[2]} x - {theta_2[4]}) + {theta_2[1]} \\cdot \\text{{ReLU}}({theta_2[3]} x - {theta_2[5]})"
non_lin_syms.append(non_lin2_sym)

# Definition for the inverse function parametrized with ReLUs
def non_linearity_3(x):
    alpha1, alpha2, gamma1, gamma2, beta1, beta2 = theta_3
    return (alpha1 * nn.relu(gamma1 * x - beta1) + alpha2 * nn.relu(gamma2 * x - beta2))

non_lin3_sym = f"{theta_3[0]} \\cdot \\text{{ReLU}}({theta_3[2]} x - {theta_3[4]}) + {theta_3[1]} \\cdot \\text{{ReLU}}({theta_3[3]} x - {theta_3[5]})"
non_lin_syms.append(non_lin3_sym)


system = DifferentialEquations(5, 5, 10, 10, [non_linearity_1, non_linearity_2, non_linearity_3])
print(system.equations)