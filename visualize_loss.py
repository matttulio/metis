import jax
from jax import random, vmap, jit
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import flax.linen as nn
from flax.training import train_state
from src.datagen import Equations
from functools import partial
from optax import adam


jax.config.update("jax_enable_x64", True)


class TrainState(train_state.TrainState):
    pass


class CustomActivation_old(nn.Module):
    input_dim: int
    L: int  # Number of parameter groups
    nls_init: jnp.ndarray | None = None
    trainable: bool = False

    def setup(self):
        if self.nls_init is None:
            self.alpha = self.param("alpha", nn.initializers.normal(), (3, self.L))
            self.gamma = self.param("gamma", nn.initializers.normal(), (3, self.L))
        else:
            assert self.nls_init.shape == (self.L, 6), "nls_init must have shape (L, 6)"
            alpha_init = self.nls_init[:, :3].T  # (3, L)
            gamma_init = self.nls_init[:, 3:6].T  # (3, L)
            self.alpha = self.param("alpha", lambda *_: alpha_init)
            self.gamma = self.param("gamma", lambda *_: gamma_init)

        assert self.input_dim % self.L == 0, "input_dim must be divisible by L"
        self.group_indices = jnp.arange(self.input_dim) % self.L

        if not self.trainable:
            self.alpha = jax.lax.stop_gradient(self.alpha)
            self.gamma = jax.lax.stop_gradient(self.gamma)

    def __call__(self, x):
        # Gather parameters for all input dimensions (3, input_dim)
        alpha = self.alpha[:, self.group_indices]  # (3, D)
        gamma = self.gamma[:, self.group_indices]  # (3, D)

        activated_terms = []
        for i in range(3):
            alpha_i = alpha[i, :]  # (D,)
            gamma_i = gamma[i, :]  # (D,)
            # Compute each activation term: (B, D)
            term = alpha_i[None, :] * nn.relu(x + gamma_i[None, :])
            activated_terms.append(term)

        # Stack along the last dimension and sum
        activated = jnp.stack(activated_terms, axis=-1)  # (B, D, 3)
        return jnp.sum(activated, axis=-1)  # (B, D)


class ZeroLayersNN(nn.Module):
    N: int
    L: int
    output_dim: int
    y_mean: jnp.ndarray | None = None
    y_std: jnp.ndarray | None = None
    nls_init: jnp.ndarray | None = None
    train_activations: bool = False

    def setup(self):
        # Set default values if not provided
        if self.y_mean is None:
            self.y_mean = jnp.zeros((self.output_dim,))
        if self.y_std is None:
            self.y_std = jnp.ones((self.output_dim,))

    @nn.compact
    def __call__(self, x):
        x = CustomActivation_old(self.N, self.L, self.nls_init, self.train_activations)(
            x
        )
        x = nn.Dense(self.output_dim, use_bias=False)(x)
        return (x - self.y_mean) / self.y_std


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


def my_eval(v):
    return system(y=v)


def create_batches(inputs, outputs, batch_size, seed=42):
    key = random.key(seed)
    num_samples = inputs.shape[0]
    indices = jnp.arange(num_samples)
    shuffled_indices = random.permutation(key, indices)

    num_batches = num_samples // batch_size
    batches_x = [
        jnp.array(inputs[shuffled_indices[i * batch_size : (i + 1) * batch_size]])
        for i in range(num_batches)
    ]

    batches_x = jnp.array(batches_x)

    batches_y = [
        jnp.array(outputs[shuffled_indices[i * batch_size : (i + 1) * batch_size]])
        for i in range(num_batches)
    ]

    batches_y = jnp.array(batches_y)

    return tuple([batches_x, batches_y])


def count_params(params):
    total_params = 0
    for _, layer_params in params.items():
        for _, param_value in layer_params.items():
            # Add the number of elements in the array to the total
            total_params += param_value.size
    return total_params


@partial(jax.jit, static_argnums=(1))
def loss_fn(params, apply_fn, batch_x, batch_y):
    predictions = apply_fn(params, batch_x)
    mse_loss = jnp.mean((predictions - batch_y) ** 2)

    return mse_loss


def visualize_loss_landscape(model, params, batch_x, batch_y, steps=200, scale=1.0):
    # Get random directions for parameter perturbation
    key = jax.random.PRNGKey(0)
    directions = []
    for param in jax.tree_util.tree_leaves(params):
        key, subkey = jax.random.split(key)
        directions.append(jax.random.normal(subkey, param.shape))

    # Create two perturbation directions
    dir1 = jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(params), directions
    )
    dir2 = jax.tree_util.tree_map(
        lambda x: jax.random.normal(jax.random.PRNGKey(1), x.shape), params
    )

    # Normalize directions
    def normalize(d):
        norm = jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree_util.tree_leaves(d)))
        return jax.tree_util.tree_map(lambda x: x / norm, d)

    dir1 = normalize(dir1)
    dir2 = normalize(dir2)

    # Create grid of alpha and beta values
    alphas = np.linspace(-scale, scale, steps)
    betas = np.linspace(-scale, scale, steps)

    # Compute loss values
    loss_values = np.zeros((len(alphas), len(betas)))

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            # Perturb parameters
            perturbed_params = jax.tree_util.tree_map(
                lambda p, d1, d2: p + alpha * d1 + beta * d2, params, dir1, dir2
            )

            # Compute loss
            loss = loss_fn(perturbed_params, model.apply, batch_x, batch_y)
            loss_values[i, j] = loss

    # Create meshgrid for plotting
    Alpha, Beta = np.meshgrid(alphas, betas)

    # Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(Alpha, Beta, loss_values, cmap="viridis")

    ax.set_xlabel("Direction 1 (alpha)")
    ax.set_ylabel("Direction 2 (beta)")
    ax.set_zlabel("Loss")
    ax.set_title("Loss Landscape Visualization")
    ax.set_zscale("log")  # Set z-axis to log scale

    plt.show()
    return loss_values


state_variables = 30
expected_number_of_nls = 20
input_dim = state_variables * expected_number_of_nls
output_dim = state_variables

seed = 42
train_batch_size = 500
test_batch_size = 128


dims = 6
n_nls = expected_number_of_nls

non_lins, vec_nls = generate_callable_functions(dims, n_nls, seed)

config = {
    "n_vars": state_variables,
    "n_eqs": state_variables,
    "bounds_addends": (1, 3),
    "bounds_multiplicands": (1, 1),
    "non_lins": non_lins,
    "sym_non_lins": None,
    "distribution": "uniform",
    "a": None,
    "b": None,
    "sigma": None,
    "p": None,
    "seed": seed,
}


system = Equations(config)


train_size = int(1e4)
test_ratio = 0.2
test_size = int(test_ratio * train_size)
total_size = train_size + test_size
bound = 10
train_values = jnp.tile(
    jnp.linspace(-bound, bound, total_size).reshape(-1, 1), (1, config["n_eqs"])
)

# Randomly select `test_size` elements for inbound_test
key = random.key(seed)
key, subkey = random.split(key)  # Ensure reproducibility
indices = random.choice(
    subkey, total_size, shape=(test_size,), replace=False
)  # Unique indices
inbound_test = train_values[indices]  # Extract test points
mask = jnp.ones(total_size, dtype=bool)
mask = mask.at[indices].set(False)  # Mask out test points
train_values = train_values[mask]  # Remove test points from training set

# Create out-of-bound test values
range1 = jnp.linspace(-bound - 5, -bound, test_size // 2)
range2 = jnp.linspace(bound, bound + 5, test_size // 2)
outofbound_test = jnp.concatenate((range1, range2), axis=0).reshape(-1, 1)
outofbound_test = jnp.tile(outofbound_test, (1, config["n_eqs"]))

# Stack both test sets
stacked_test = jnp.vstack((inbound_test, outofbound_test))

# Concatenate train and test values
values = jnp.vstack((train_values, stacked_test))

# Standardize values
values_mean = jnp.mean(values, axis=0)
values_std = jnp.std(values, axis=0)


# Evaluate
evaluated_values = vmap(my_eval)(values)
# values = (values - values_mean) / values_std


# Standardize evaluated_values
evaluated_values_mean = jnp.mean(evaluated_values, axis=0)
evaluated_values_std = jnp.std(evaluated_values, axis=0)
evaluated_values = (evaluated_values - evaluated_values_mean) / evaluated_values_std

print(evaluated_values_mean)
print(evaluated_values_std)

expanded_values = jnp.repeat(values, expected_number_of_nls, axis=1)
train_values = expanded_values[:train_size]
inbound_test = expanded_values[train_size : test_size + train_size]
outofbound_test = expanded_values[test_size + train_size :]

train_outputs, inbound_test_outputs = (
    evaluated_values[:train_size],
    evaluated_values[train_size : test_size + train_size],
)
outofbound_test_outputs = evaluated_values[test_size + train_size :]

train_batches = create_batches(train_values, train_outputs, train_batch_size)
inbound_test_batches = create_batches(
    inbound_test, inbound_test_outputs, test_batch_size
)
outofbound_test_batches = create_batches(
    outofbound_test, outofbound_test_outputs, test_batch_size
)

# train_batches = jnp.array(train_batches)
# inbound_test_batches = jnp.array(inbound_test_batches)
# outofbound_test_batches = jnp.array(outofbound_test_batches)

# # Move the entire batch dataset to the device
# train_batches = jax.device_put(train_batches)
# inbound_test_batches = jax.device_put(inbound_test_batches)
# outofbound_test_batches = jax.device_put(outofbound_test_batches)

num_epochs = 101

model = ZeroLayersNN(
    N=input_dim,
    L=expected_number_of_nls,
    output_dim=state_variables,
    y_mean=evaluated_values_mean,
    y_std=evaluated_values_std,
    nls_init=None,
    train_activations=True,
)

training_steps = (
    (train_size + train_batch_size - 1) // train_batch_size
) * num_epochs  # Ceiling division

dummy_input = jnp.zeros((train_batch_size, input_dim))
key, subkey = random.split(key)
params = model.init(subkey, dummy_input)
optimizer = adam(learning_rate=0.001)
state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

apply_fn = model.apply
batch_x, batch_y = train_batches
# Visualize the landscape
loss_values = visualize_loss_landscape(
    model, params, batch_x[0], batch_y[0], steps=50, scale=10.0
)
