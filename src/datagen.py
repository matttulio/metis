import jax
import jax.numpy as jnp
from functools import partial
import os
import hashlib
import json
import cloudpickle
from matplotlib.backends.backend_pdf import PdfPages
import scienceplots
import matplotlib.pyplot as plt
import networkx as nx


class Equations:
    def __init__(
        self,
        config: dict,
    ):
        """
        Initialize equations generator.

        Parameters:
        n_vars: number of variables of the system;
        n_eqs: number of equations to generate;
        bounds_addends: bounds on the number of addends in each equation;
        bounds_multiplicands: bounds on the number of multiplicands in each addend;
        non_lins: list of all the possible non-linearities;
        save_dir: directory where to save the system;
        sym_non_lins: list of the symbolic expressions of the non-linearities;
        distribution: type of distribution to use for variable probabilities ('uniform', 'beta', 'lognormal', 'custom');
        p: custom probabilities for the variables (used only if distribution is 'custom');
        seed: seed for reproducibility.
        """

        # Define required keys
        required_keys = {
            "n_vars",
            "n_eqs",
            "bounds_addends",
            "bounds_multiplicands",
            "non_lins",
        }

        # Raise error if any required key is missing
        missing_keys = required_keys - config.keys()

        if missing_keys:
            raise ValueError(f"Missing required keys: {', '.join(missing_keys)}")

        # Default configuration
        defaults = {
            "save_dir": "Data",
            "sym_non_lins": None,
            "distribution": "uniform",
            "a": None,
            "b": None,
            "sigma": None,
            "p": None,
            "seed": 42,
        }

        self.config = {**defaults, **config}

        self.save_dir = self.config["save_dir"]
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print(f"Directory {self.save_dir} created.")

        self.filename = self.__get_unique_name_from_config()

        if os.path.exists(os.path.join(self.save_dir, self.filename + ".pkl")):
            self.__load_system()
        else:
            self.system = self.__generate_system()
            self.__save_system()

    def __save_system(self):
        try:
            # Save the system and the entire state (self) to a file
            with open(os.path.join(self.save_dir, self.filename + ".pkl"), "wb") as f:
                # Include 'self' in the saved state (can store all relevant instance variables)
                cloudpickle.dump(self, f)
            print(
                f"System and state saved as {self.filename+'.pkl'} in directory {self.save_dir}"
            )
        except Exception as e:
            print(f"Error saving the system: {e}")

    def __load_system(self):
        # Load the system and state from a file (restore the entire object)
        if os.path.exists(os.path.join(self.save_dir, self.filename + ".pkl")):
            try:
                with open(
                    os.path.join(self.save_dir, self.filename + ".pkl"), "rb"
                ) as f:
                    loaded_obj = cloudpickle.load(f)
                    # Restore the instance's state
                    self.__dict__ = (
                        loaded_obj.__dict__
                    )  # This copies all instance variables
                print(f"System and state loaded from {self.filename+'.pkl'}")
            except Exception as e:
                print(f"Error loading the system: {e}")
                self.system = None  # Set to None if loading fails
        else:
            print(f"No saved system found. Generating a new one.")

    def __generate_system(self):
        # Initialize parameters
        for key, value in {**self.config}.items():
            setattr(self, key, value)

        self.n_nls = len(self.non_lins)

        if self.sym_non_lins is None:
            self.sym_non_lins = [f"nl_{{{i+1}}}" for i in range(self.n_nls)]
        else:
            self.sym_non_lins = self.sym_non_lins

        # Generate a key for reproducibility
        key = jax.random.key(self.seed)

        # Function that creates random numbers
        @jax.jit
        def generate_random_number(subkey, minval, maxval):
            return jax.random.randint(subkey, shape=(1,), minval=minval, maxval=maxval)

        # Number of addends for each equation
        self.n_addends = jax.random.randint(
            key,
            shape=(self.n_eqs,),
            minval=self.bounds_addends[0],
            maxval=self.bounds_addends[1] + 1,
        )
        total_addends = jnp.sum(self.n_addends)

        # Number of multiplicands for each addend
        minval = jnp.ones(total_addends) * self.bounds_multiplicands[0]
        maxval = jnp.ones(total_addends) * self.bounds_multiplicands[1] + 1
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
        maxval = jnp.ones(self.total_multiplicands) * self.n_vars
        subkey = jax.random.split(subkey[0], self.total_multiplicands)
        p_key, subkey = jax.random.split(subkey[0])

        if self.distribution == "uniform":
            p = jax.random.uniform(p_key, shape=(self.n_vars,))
            p = p / jnp.sum(p)
        elif self.distribution == "beta":
            if self.a is None or self.b is None:
                raise ValueError(
                    "Parameters 'a' and 'b' must be provided for beta distribution."
                )
            p = jax.random.beta(p_key, a=self.a, b=self.b, shape=(self.n_vars,))
            p = p / jnp.sum(p)
        elif self.distribution == "lognormal":
            if self.sigma is None:
                raise ValueError(
                    "Parameter 'sigma' must be provided for lognormal distribution."
                )
            p = jax.random.lognormal(p_key, shape=(self.n_vars,), sigma=self.sigma)
            p = p / jnp.sum(p)
        elif self.distribution == "custom":
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
            subkey, self.n_vars, shape=(self.total_multiplicands,), p=p
        )

        # Convert the lists of lists into a list
        self.nls_indices = self.nls_indices[0]

        # Get the sets of unique nls
        unique_nls_idx = jnp.unique(self.nls_indices)

        # Get the idxs of the variables to which apply the nls
        self.target_var_idxs = [
            jnp.unique(self.variables_indices[self.nls_indices == idx])
            for idx in range(self.n_nls)
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

        for eq in range(self.n_eqs):
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
            (
                self.n_eqs,
                self.bounds_addends[1],
                self.bounds_multiplicands[1],
                self.n_nls,
            ),
            dtype=float,
        )
        eqs_mask = jnp.zeros(
            (
                self.n_eqs,
                self.bounds_addends[1],
                self.bounds_multiplicands[1],
                self.n_nls,
            ),
            dtype=bool,
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

        # Create the symbolic equations
        self.sym_sys = []
        k = 0
        for eq in range(self.n_eqs):
            sym_expr = rf"f_{{{eq+1}}} = "
            for _ in range(self.n_addends[eq]):
                for _ in range(self.n_multiplicands[0, k]):
                    sym_expr += rf"{self.sym_non_lins[self.nls_indices[k]]}(y_{{{self.variables_indices[k]+1}}})"
                    k += 1
                sym_expr += " + "
            sym_expr = sym_expr[:-3]
            self.sym_sys.append(sym_expr)

    # Function that will be called by the integrator
    def __call__(self, t=None, y=None):
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

    def __filter_config(self):
        filtered_config = {}
        for key, value in self.config.items():
            if isinstance(value, tuple):  # Convert tuples of functions to names
                filtered_config[key] = tuple(self.__get_callable_name(v) for v in value)
            elif callable(value):  # Replace single functions with their name
                filtered_config[key] = self.__get_callable_name(value)
            elif isinstance(value, dict):  # Recursively process nested dictionaries
                filtered_config[key] = self.__filter_config(value)
            else:
                filtered_config[key] = value
        return filtered_config

    def __get_callable_name(self, value):
        if callable(value):
            # Try to get the name of the callable, otherwise use a fallback
            try:
                return value.__name__
            except AttributeError:
                return str(value)  # If no __name__, return the string representation
        return value  # Not callable, return the original value

    def __get_unique_name_from_config(self):
        # Filter out non-deterministic or non-serializable elements
        filtered_config = self.__filter_config()

        # Convert the filtered dictionary to a JSON string with sorted keys
        config_string = json.dumps(filtered_config, sort_keys=True)

        # Generate a hash from the string representation
        config_hash = hashlib.md5(config_string.encode()).hexdigest()

        # Use the hash to create a unique filename
        filename = f"equations_{config_hash}"
        return filename

    # Create the PDF with the symbolic equations
    def save_symb_expr(self, filename=None, max_eq_per_page=35):
        if filename is None:
            filename = self.filename + "_symbolic.pdf"

        if os.path.exists(os.path.join(self.save_dir, filename)):
            print("PDF already exists")
            return

        plt.style.use("science")
        plt.rcParams["text.usetex"] = True

        with PdfPages(os.path.join(self.save_dir, filename)) as pdf:
            for i in range(0, len(self.sym_sys), max_eq_per_page):
                fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 size in inches
                ax.axis("off")
                text = "\n".join(
                    [f"${eq}$" for eq in self.sym_sys[i : i + max_eq_per_page]]
                )
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
        plt.close()

    def show_graph(self):
        plt.style.use("science")

        new_wave = self.sym_sys.copy()

        for n, eq in zip(range(self.n_eqs), self.sym_sys):
            new_wave[n] = eq.removeprefix(f"f_{{{n+1}}} = ")

        dependencies = []

        for s, f in zip(range(self.n_eqs), new_wave):
            temp = f"f_{{{s+1}}} "
            for n, eq in zip(range(self.n_eqs), new_wave):
                if eq in f and f != eq:
                    temp += f"f_{{{n+1}}} "
                    new_wave[s] = new_wave[s].replace(eq, f"f_{{{n+1}}}")

                if f"y_{{{n+1}}}" in f:
                    temp += f"y_{{{n+1}}} "

            dependencies.append(temp)

        # Create an undirected graph
        G = nx.Graph()

        # Parse the dependencies
        for item in dependencies:
            tokens = item.split()
            if not tokens:
                continue
            target_node = tokens[0]
            G.add_node(target_node)

            for source_node in tokens[1:]:
                G.add_node(source_node)
                G.add_edge(source_node, target_node)

        # Categorize nodes into layers
        y_nodes = {node for node in G.nodes if node.startswith("y")}
        f_nodes = {node for node in G.nodes if node.startswith("f")}

        # f_nodes that have at least one y_node as a neighbor but NO f_node as a neighbor
        f_dependent_on_y = {
            f
            for f in f_nodes
            if any(neigh in y_nodes for neigh in G.neighbors(f))
            and not any(neigh in f_nodes for neigh in G.neighbors(f))
        }

        # f_nodes that have at least one f_node as a neighbor (can also depend on y_nodes)
        f_dependent_on_f = f_nodes - f_dependent_on_y

        # Assign positions in concentric circles
        def assign_positions(nodes, radius, center=(0, 0)):
            if not nodes:  # Check if the list of nodes is empty
                return {}
            angle_step = 2 * jnp.pi / len(nodes)
            positions = {}
            for i, node in enumerate(nodes):
                angle = i * angle_step
                x = center[0] + radius * jnp.cos(angle)
                y = center[1] + radius * jnp.sin(angle)
                positions[node] = (x, y)
            return positions

        # Bottom layer (y variables)
        pos = assign_positions(
            sorted(y_nodes, key=lambda v: int(v.split("_")[1].strip("{}"))), radius=1
        )
        y_color = "#d9ed92"

        # Middle layer (first-level functions)
        pos.update(
            assign_positions(
                sorted(
                    f_dependent_on_y, key=lambda v: int(v.split("_")[1].strip("{}"))
                ),
                radius=2,
            )
        )
        f_y_color = "#76c893"

        # Top layer (higher-level functions)
        pos.update(
            assign_positions(
                sorted(
                    f_dependent_on_f, key=lambda v: int(v.split("_")[1].strip("{}"))
                ),
                radius=3,
            )
        )
        f_f_color = "#168aad"

        plt.figure(figsize=(14, 8))

        # Draw nodes with different colors
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=y_nodes,
            node_size=300,
            node_color=y_color,
            edgecolors="black",
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=f_dependent_on_y,
            node_size=300,
            node_color=f_y_color,
            edgecolors="black",
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=f_dependent_on_f,
            node_size=300,
            node_color=f_f_color,
            edgecolors="black",
        )

        # Draw edges
        nx.draw_networkx_edges(
            G, pos, edgelist=G.edges, edge_color="#00171f", alpha=0.7
        )

        plt.title("Visualisation as Graph", fontsize=14)
        plt.axis("off")
        plt.show()
        plt.clf()
        plt.close()


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
    return jnp.sum(jnp.prod(jnp.prod(matrix, axis=3), axis=2), axis=1)
