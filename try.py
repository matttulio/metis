from jax import random, vmap, jit, lax
from optax import adam, warmup_cosine_decay_schedule
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
from functools import partial
from typing import Sequence, Callable, List
from src.datagen import Equations
from model import train_step, evaluate
import time

print(jax.devices())

x = jnp.ones(1000)  # Create array on GPU
print(x.device)  # Should show `CudaDevice(id=0)`


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


class LearnableActivationsFixedParams(nn.Module):
    input_features: int
    activations: List[Callable]
    n_params: int = 2

    def setup(self):
        self.num_activations = len(self.activations)
        if self.input_features % self.num_activations != 0:
            raise ValueError("Input features must be divisible by num_activations")

        # Parameters stored as (num_activations, n_params)
        self.params = self.param(
            "activation_params",
            nn.initializers.normal(),
            (self.num_activations, self.n_params),
        )

    def __call__(self, x):
        # Split input once (B, D) -> list of (B, D//num_activations)
        splits = jnp.split(x, self.num_activations, axis=-1)

        # Process all activations in parallel using list comprehension
        activated_parts = [
            activation(split, params)
            for activation, split, params in zip(self.activations, splits, self.params)
        ]

        # Single concatenation at the end
        return jnp.concatenate(activated_parts, axis=-1)


class SampleLearnModel(nn.Module):
    input_dim: int  # Neurons per activation group
    n_funcs: int  # Number of activation functions
    output_dim: int
    activations: List[Callable]
    y_mean: jnp.ndarray | None = None
    y_std: jnp.ndarray | None = None
    max_num_params: int = 2

    def setup(self):
        self.custom_activation = LearnableActivationsFixedParams(
            self.input_dim * self.n_funcs, self.activations, self.max_num_params
        )
        self.output_layer = nn.Dense(self.output_dim, use_bias=False)

    def __call__(self, x):
        x = self.custom_activation(x)

        x = self.output_layer(x)
        # print(x)
        return (x - self.y_mean) / self.y_std


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
    return Q[:dim, :num]


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


def create_batches(inputs, outputs, batch_size):
    num_batches = inputs.shape[0] // batch_size
    return jax.tree.map(
        lambda x: x[: num_batches * batch_size].reshape(num_batches, batch_size, -1),
        (inputs, outputs),
    )


def count_params(params):
    total_params = 0
    for _, layer_params in params.items():
        for _, param_value in layer_params.items():
            # Add the number of elements in the array to the total
            total_params += param_value.size
    return total_params


################################################################################
################################################################################
################################################################################

state_variables = 4
expected_number_of_nls = 4
input_dim = state_variables * expected_number_of_nls
output_dim = state_variables

seed = 42
train_batch_size = 32
test_batch_size = 64


dims = 6
n_nls = expected_number_of_nls

non_lins, vec_nls = generate_callable_functions(dims, n_nls, seed)

config = {
    "n_vars": state_variables,
    "n_eqs": state_variables,
    "bounds_addends": (1, 1),
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
schedule = warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=1e-2,
    warmup_steps=training_steps // 10,
    decay_steps=(training_steps - training_steps // 10),
    end_value=1e-6,
)
optimizer = adam(learning_rate=schedule)
state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)


@partial(jax.jit, static_argnums=(1))
def loss_fn(params, apply_fn, batch_x, batch_y):
    predictions = apply_fn(params, batch_x)
    return jnp.mean((predictions - batch_y) ** 2)


@jit
def train_epoch(carry, epoch):
    state, _ = carry  # Unpack state and dummy loss accumulator
    epoch_loss = jnp.array(0.0)

    # Loop over batches
    def batch_step(carry, batch):
        state, _ = carry
        batch_x, batch_y = batch
        state, train_loss = train_step(state, batch_x, batch_y, loss_fn)
        inbound_test_loss = evaluate(state, inbound_test_batches, loss_fn)
        outofbound_test_loss = evaluate(state, outofbound_test_batches, loss_fn)

        return (state, train_loss), (
            train_loss,
            inbound_test_loss,
            outofbound_test_loss,
        )

    (state, _), (train_losses, inbound_test_losses, outofbound_test_losses) = lax.scan(
        batch_step, (state, 0.0), train_batches
    )

    # Print every n epochs
    print_condition = epoch % 25 == 0
    jax.lax.cond(
        print_condition,
        # True branch: print metrics
        lambda: jax.debug.print(
            "Epoch: {epoch}, Train Loss: {train_loss:.4e}, Inbound-Test Loss: {inbound_test_loss:.4e}, Outofbound-Test Loss: {outofbound_test_loss:.4e} ",
            epoch=epoch,
            train_loss=jnp.mean(train_losses),
            inbound_test_loss=jnp.mean(inbound_test_losses),
            outofbound_test_loss=jnp.mean(outofbound_test_losses),
        ),
        # False branch: do nothing
        lambda: None,
    )
    return (state, epoch_loss), (
        train_losses,
        inbound_test_losses,
        outofbound_test_losses,
    )


start_time = time.time()
(state, loss), (train_losses, inbound_test_losses, outofbound_test_losses) = lax.scan(
    train_epoch, (state, 0.0), jnp.arange(num_epochs)
)
jax.block_until_ready(state)
print("Training complete!")
end_time = time.time()
print(f"Total training time: {end_time - start_time:.4f} seconds")


####################################################################################
####################################################################################
####################################################################################


expanded_values = jnp.tile(values, (1, expected_number_of_nls))
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


@jit
def activ_func(x, args):
    return (
        args[0] * nn.relu(x + args[1])
        + args[2] * nn.relu(x + args[3])
        + args[4] * nn.relu(x + args[5])
    )


activ_funcs = [activ_func] * expected_number_of_nls


model = SampleLearnModel(
    input_dim=input_dim,
    n_funcs=expected_number_of_nls,
    output_dim=state_variables,
    y_mean=evaluated_values_mean,
    y_std=evaluated_values_std,
    activations=activ_funcs,
    max_num_params=6,
)

training_steps = (
    (train_size + train_batch_size - 1) // train_batch_size
) * num_epochs  # Ceiling division

dummy_input = jnp.zeros((train_batch_size, input_dim))
key, subkey = random.split(key)
params = model.init(subkey, dummy_input)
schedule = warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=1e-2,
    warmup_steps=training_steps // 10,
    decay_steps=(training_steps - training_steps // 10),
    end_value=1e-6,
)
optimizer = adam(learning_rate=schedule)
state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)


# @partial(jax.jit, static_argnums=(1))
# def loss_fn(params, apply_fn, batch_x, batch_y):
#     predictions = apply_fn(params, batch_x)
#     return jnp.mean((predictions - batch_y) ** 2)


# @jit
# def train_epoch(carry, epoch):
#     state, _ = carry  # Unpack state and dummy loss accumulator
#     epoch_loss = jnp.array(0.0)

#     # Loop over batches
#     def batch_step(carry, batch):
#         state, _ = carry
#         batch_x, batch_y = batch
#         state, train_loss = train_step(state, batch_x, batch_y, loss_fn)
#         inbound_test_loss = evaluate(state, inbound_test_batches, loss_fn)
#         outofbound_test_loss = evaluate(state, outofbound_test_batches, loss_fn)

#         return (state, train_loss), (
#             train_loss,
#             inbound_test_loss,
#             outofbound_test_loss,
#         )

#     (state, _), (train_losses, inbound_test_losses, outofbound_test_losses) = lax.scan(
#         batch_step, (state, 0.0), train_batches
#     )

#     # Print every n epochs
#     print_condition = epoch % 25 == 0
#     jax.lax.cond(
#         print_condition,
#         # True branch: print metrics
#         lambda: jax.debug.print(
#             "Epoch: {epoch}, Train Loss: {train_loss:.4e}, Inbound-Test Loss: {inbound_test_loss:.4e}, Outofbound-Test Loss: {outofbound_test_loss:.4e} ",
#             epoch=epoch,
#             train_loss=jnp.mean(train_losses),
#             inbound_test_loss=jnp.mean(inbound_test_losses),
#             outofbound_test_loss=jnp.mean(outofbound_test_losses),
#         ),
#         # False branch: do nothing
#         lambda: None,
#     )
#     return (state, epoch_loss), (
#         train_losses,
#         inbound_test_losses,
#         outofbound_test_losses,
#     )


start_time = time.time()
(state, loss), (train_losses, inbound_test_losses, outofbound_test_losses) = lax.scan(
    train_epoch, (state, 0.0), jnp.arange(num_epochs)
)

jax.block_until_ready(state)
print("Training complete!")
end_time = time.time()
print(f"Total training time: {end_time - start_time:.4f} seconds")
