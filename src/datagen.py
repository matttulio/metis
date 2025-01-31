import jax
import jax.numpy as jnp
from functools import partial
from scipy.integrate import solve_ivp
import os


class Equations:
    def __init__(
        self,
        n_vars,
        n_eqs,
        max_addends,
        max_multiplicands,
        non_lins,
        sym_non_lins=None,
        distribution="uniform",
        p=None,
        seed=42,
    ):
        """
        Initialize equations generator.

        Parameters:
        n_vars: number of variables of the system;
        n_eqs: number of equations to generate;
        max_addends: max number of addends in each equation;
        max_multiplicands: max number of multiplicands in each addend;
        non_lins: list of all the possible non-linearities;
        sym_non_lins: list of the symbolic expressions of the non-linearities;
        distribution: type of distribution to use for variable probabilities ('uniform', 'beta', 'lognormal', 'custom');
        p: custom probabilities for the variables (used only if distribution is 'custom');
        seed: seed for reproducibility.
        """

        # Initialize parameters
        self.n_vars = n_vars
        self.n_eqs = n_eqs
        self.non_lins = non_lins
        self.n_nls = len(non_lins)

        if sym_non_lins is None:
            self.sym_non_lins = [f"nl_{i+1}" for i in range(self.n_nls)]
            print(self.sym_non_lins)
        else:
            self.sym_non_lins = sym_non_lins

        self.seed = seed

        # Generate a key for reproducibility
        key = jax.random.key(seed)

        # Function that creates random numbers
        @jax.jit
        def generate_random_number(subkey, minval, maxval):
            return jax.random.randint(subkey, shape=(1,), minval=minval, maxval=maxval)

        # Number of addends for each equation
        self.n_addends = jax.random.randint(
            key, shape=(n_eqs,), minval=1, maxval=max_addends + 1
        )
        total_addends = jnp.sum(self.n_addends)

        # Number of multiplicands for each addend
        minval = jnp.ones(total_addends)
        maxval = jnp.ones(total_addends) * max_multiplicands + 1
        subkey = jax.random.split(key, total_addends)
        self.n_multiplicands = jax.vmap(generate_random_number, out_axes=1)(
            subkey, minval, maxval
        )
        self.total_multiplicands = jnp.sum(self.n_multiplicands)

        # Non-linearity index for each multiplicand
        minval = jnp.zeros(self.total_multiplicands)
        maxval = jnp.ones(self.total_multiplicands) * self.n_nls
        subkey = jax.random.split(subkey[0], self.total_multiplicands)
        self.nls_indices = jax.vmap(generate_random_number, out_axes=1)(
            subkey, minval, maxval
        )
        self.static_nls_indices = tuple(
            self.nls_indices.tolist()[0]
        )  # static versio used in jit compiled functions

        # Variable index for each non-linearity
        maxval = jnp.ones(self.total_multiplicands) * n_vars
        subkey = jax.random.split(subkey[0], self.total_multiplicands)
        p_key, subkey = jax.random.split(subkey[0])

        if distribution == "uniform":
            p = jax.random.uniform(p_key, shape=(n_vars,))
            p = p / jnp.sum(p)
        elif distribution == "beta":
            p = jax.random.beta(p_key, a=0.5, b=0.5, shape=(n_vars,))
            p = p / jnp.sum(p)
        elif distribution == "lognormal":
            p = jax.random.lognormal(p_key, shape=(n_vars,), sigma=2)
            p = p / jnp.sum(p)
        elif distribution == "custom":
            if p is None:
                raise ValueError(
                    "The probabilities p must be provided when using custom distribution."
                )

            if not jnp.isclose(jnp.sum(p), 1.0):
                raise ValueError("The sum of the probabilities p must be equal to 1.")
        else:
            raise ValueError("Unsupported distribution type.")

        # Generate variables_indices based on the probabilities p
        self.variables_indices = jax.random.choice(
            subkey, n_vars, shape=(self.total_multiplicands,), p=p
        )

        # Convert the lists of lists into a list
        self.nls_indices = self.nls_indices[0]

        # Get the sets of unique nls
        unique_nls_idx = jnp.unique(self.nls_indices)

        # Get the idxs of the variables to which apply the nls
        self.target_var_idxs = [
            jnp.unique(self.variables_indices[self.nls_indices == idx])
            for idx in unique_nls_idx
        ]

        number_ops = sum(
            len(sublist) for sublist in self.target_var_idxs
        )  # total number of operations
        lengths = jnp.array(
            [len(sublist) for sublist in self.target_var_idxs]
        )  # length of the sublists inside target_var_idxs

        # Create the indices for slicing the array of the results
        start_idxs = jnp.concatenate([jnp.array([0]), jnp.cumsum(lengths)[:-1]])
        self.end_idxs = start_idxs + lengths
        self.start_idxs = tuple(start.item() for start in start_idxs)
        self.end_idxs = tuple(end.item() for end in self.end_idxs)
        self.results = jnp.ones(number_ops)

        # Create a mask for each non-linearity
        mask_nls = jnp.zeros((self.n_nls, self.total_multiplicands), dtype=bool)

        for i in range(self.n_nls):
            mask_nls = mask_nls.at[i].set(self.nls_indices == i)

        # Create a mask for each variable
        mask_vars = jnp.zeros((number_ops, self.total_multiplicands), dtype=bool)
        k = 0
        for i in range(self.n_nls):
            for _, val in enumerate(self.target_var_idxs[i]):
                mask_vars = mask_vars.at[k].set(self.variables_indices == val)
                k += 1

        # Create the logical and mask between the previous masks
        self.nls_and_vars = jnp.zeros(
            (number_ops, self.total_multiplicands), dtype=bool
        )

        k = 0
        for i in range(self.n_nls):
            for _, val in enumerate(self.target_var_idxs[i]):
                self.nls_and_vars = self.nls_and_vars.at[k].set(
                    jnp.logical_and(mask_nls[i], mask_vars[k])
                )
                k += 1

        # Initialize the intermediate output
        self.output = jnp.zeros(self.total_multiplicands, dtype=jnp.float32)

        # Build the indices that will locate
        # the variables in the equations tensor
        eqs_idxs = []
        addend_idxs = []
        mult_idxs = []

        multiplicand_index = 0

        for eq in range(n_eqs):
            for addend in range(self.n_addends[eq]):
                num_multiplicands = self.n_multiplicands[0, multiplicand_index]
                for mult in range(num_multiplicands):
                    eqs_idxs.append(eq)
                    addend_idxs.append(addend)
                    mult_idxs.append(mult)
                multiplicand_index += 1

        self.eqs_idxs = tuple(eqs_idxs)
        self.addend_idxs = tuple(addend_idxs)
        self.mult_idxs = tuple(mult_idxs)

        # Initialize the equations tensor, and its mask for future updates
        self.equations = jnp.ones(
            (n_eqs, max_addends, max_multiplicands, self.n_nls), dtype=float
        )
        eqs_mask = jnp.zeros(
            (n_eqs, max_addends, max_multiplicands, self.n_nls), dtype=bool
        )
        eqs_mask = eqs_mask.at[
            eqs_idxs, addend_idxs, mult_idxs, self.static_nls_indices
        ].set(True)

        # Set to zero blocks that are all false
        false_blocks = ~jnp.any(eqs_mask, axis=(2, 3))
        false_blocks_expanded = false_blocks[:, :, None, None]
        self.equations = jnp.where(false_blocks_expanded, 0, self.equations)
        self.equations = update_values(
            self.equations,
            self.eqs_idxs,
            self.addend_idxs,
            self.mult_idxs,
            self.static_nls_indices,
            self.variables_indices,
        )

    # Function that will be called by the integrator
    def __call__(self, t, y):
        target_values = get_target_values(y, self.target_var_idxs)
        short_res = compute(
            self.non_lins, target_values, self.results, self.start_idxs, self.end_idxs
        )
        long_res = map_results(self.output, short_res, self.nls_and_vars)
        self.equations = update_values(
            self.equations,
            self.eqs_idxs,
            self.addend_idxs,
            self.mult_idxs,
            self.static_nls_indices,
            long_res,
        )

        return collapse(self.equations)

    def __getitem__(self, idx):
        return self.equations[idx]

    # Create the PDF with the symbolic equations
    def save_symb_expr(self, filename="equations.pdf", max_eq_per_page=35):
        if os.path.exists(os.path.join(filename)):
            print("PDF already exists")
            return

        from matplotlib.backends.backend_pdf import PdfPages
        import scienceplots
        import matplotlib.pyplot as plt

        plt.style.use("science")
        plt.rcParams["text.usetex"] = True

        sym_sys = []
        k = 0
        for eq in range(self.n_eqs):
            sym_expr = rf"f_{{{eq+1}}} = "
            for _ in range(self.n_addends[eq]):
                for _ in range(self.n_multiplicands[0, k]):
                    sym_expr += rf"{self.sym_non_lins[self.nls_indices[k]]}(y_{{{self.variables_indices[k]+1}}})"
                    k += 1
                sym_expr += " + "
            sym_expr = sym_expr[:-3]
            sym_sys.append(sym_expr)

        with PdfPages(filename) as pdf:
            for i in range(0, len(sym_sys), max_eq_per_page):
                fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 size in inches
                ax.axis("off")
                text = "\n".join([f"${eq}$" for eq in sym_sys[i : i + max_eq_per_page]])
                ax.text(
                    0.1,
                    0.9,
                    text,
                    fontsize=10,
                    ha="left",
                    va="top",
                    transform=ax.transAxes,
                )
                pdf.savefig(fig)
                plt.close(fig)
        print(f"PDF saved as {filename}")
        plt.clf()
        plt.rcdefaults()


# Function that gets the sublists of values in the specified order
@jax.jit
def get_target_values(y, idxs):
    return [y[indices] for indices in idxs]


# Function that computes the results
@partial(jax.jit, static_argnums=(0, 3, 4))
def compute(funs, values, results, start, end):
    for f, v, s, e in zip(funs, values, start, end):
        results = results.at[s:e].set(f(v))
    return results


# Function that updates the values in the intermediate results array
@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def update_values(array, eqx_idxs, addend_idxs, mult_idxs, static_nls_indices, values):
    return array.at[eqx_idxs, addend_idxs, mult_idxs, static_nls_indices].set(values)


# Function that maps the results to the correct positions in the full tensor
@jax.jit
def map_results(output, results, a_and_b):
    def update_output(mask, res):
        return jnp.where(mask, res, output)

    return jnp.sum(jax.vmap(update_output)(a_and_b, results), axis=0)


# Function that collapses the tensor to get the final results
@jax.jit
def collapse(matrix):
    product_along_axis3 = jnp.prod(matrix, axis=3)
    result_per_equation = jnp.prod(product_along_axis3, axis=2)
    return jnp.sum(result_per_equation, axis=1)
