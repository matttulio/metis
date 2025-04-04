from jax import random, jit
from flax import linen as nns
import jax.numpy as jnp
import re


def generate_alphabeta(dim, num, seed):
    key = random.key(seed)
    A = random.normal(
        key, (max(num, dim), num)
    )  # Generate (max(num, dim), num)) random matrix
    Q, _, _ = jnp.linalg.svd(A, full_matrices=True)  # Get orthogonal matrix (num, num)
    return Q[:dim, :num] * 5


def generate_callable_functions(dim, num, seed):
    bases = generate_alphabeta(dim, num, seed)  # Get num basis vectors
    bases = bases.T
    alphas = bases[:, : dim // 2]  # First half of Q are alphas
    gammas = bases[:, dim // 2 :]

    def make_function(alpha, gamma):
        @jit
        def func(x):
            return (
                alpha[0] * nn.relu(x + gamma[0])
                + alpha[1] * nn.relu(x + gamma[1])
                + alpha[2] * nn.relu(x + gamma[2])
            )

        return func

    return tuple([make_function(alphas[i], gammas[i]) for i in range(num)]), bases


def get_true_params(n_nls, n_vars, sym_sys):
    n_eqs = len(sym_sys)

    max_num_terms = n_nls * n_vars  # here: 1*2*2 = 4

    # Create the result matrix (n_eqs x max_num_terms)
    matrix = jnp.zeros((n_eqs, max_num_terms), dtype=int)

    # Regular expression to find terms of the form nl_{i}(y_{j})
    pattern = r"nl_{(\d+)}\(y_{(\d+)}\)"

    # Process each equation
    for i, eq in enumerate(sym_sys):
        # Split equation on '+' to get addends (each representing a product term)
        addends = eq.split("+")
        for addend in addends:
            # Find all occurrences in the current addend, even if they are adjacent
            matches = re.findall(pattern, addend)
            # Count each occurrence: multiple occurrences in a product add up.
            for nl, var in matches:
                # Convert to zero-indexed integers
                nl_idx = int(nl) - 1
                var_idx = int(var) - 1
                # Compute the column index according to the ordering: for each y, n_nls columns.
                col_idx = nl_idx + var_idx * n_nls
                matrix = matrix.at[i, col_idx].set(matrix[i, col_idx] + 1)

    return matrix
