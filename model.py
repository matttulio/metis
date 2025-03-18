from jax import random, value_and_grad, jit, lax
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
from functools import partial
from typing import Sequence, Callable


class CustomActivation(nn.Module):
    input_dim: int
    L: int  # Number of parameter groups

    def setup(self):
        self.alpha = self.param("alpha", nn.initializers.normal(), (3, self.L))
        self.gamma = self.param("gamma", nn.initializers.normal(), (3, self.L))

        # Precompute group indices
        self.group_indices = jnp.arange(self.input_dim) % self.L

    def __call__(self, x):
        # Gather parameters for all input dimensions (3, input_dim)
        alpha = self.alpha[:, self.group_indices]
        gamma = self.gamma[:, self.group_indices]

        # Reshape for broadcasting: (3, D) -> (1, D, 3)
        alpha = jnp.moveaxis(alpha, 0, -1)[None, ...]  # Add batch dim
        gamma = jnp.moveaxis(gamma, 0, -1)[None, ...]  # Add batch dim

        # Expand x for broadcasting: (B, D) -> (B, D, 1)
        x_expanded = jnp.expand_dims(x, axis=-1)

        # Vectorized computation (B, D, 1) + (1, D, 3) -> (B, D, 3)
        activated = alpha * nn.relu(x_expanded + gamma)

        # Sum over the 3 terms (B, D, 3) -> (B, D)
        return jnp.sum(activated, axis=-1)


# Custom initializer for binary weights (0 or 1)
def binary_init(key, shape, dtype=jnp.float32):
    # Initialize with 0s and 1s (50% probability each)
    return random.bernoulli(key, p=0.5, shape=shape).astype(dtype) * 1.0


# Custom Dense layer with binarized weights using STE
class BinarizedDense(nn.Module):
    features: int
    use_bias: bool = False  # Bias can remain non-binary

    @nn.compact
    def __call__(self, inputs):
        # Initialize real-valued weights (latent variable)
        kernel = self.param(
            "kernel",
            binary_init,  # Use custom binary initializer
            (inputs.shape[-1], self.features),
        )

        # Binarize weights to 0 or 1 using STE
        binary_kernel = jnp.where(kernel > 0.5, 1, 0)  # Threshold at 0.5
        binary_kernel = kernel + lax.stop_gradient(binary_kernel - kernel)

        # Compute outputs with binarized weights
        outputs = jnp.dot(inputs, binary_kernel)

        if self.use_bias:
            bias = self.param("bias", nn.initializers.zeros, (self.features,))
            outputs += bias

        return outputs


# Modified CustomNN using Dense layers
class CustomNN(nn.Module):
    N: int
    L: int
    output_dim: int

    def setup(self):
        # Replace nn.Dense with Dense
        self.hidden_layer = nn.Dense(self.N * self.L)
        self.custom_activation = CustomActivation(self.L)
        self.output_layer = nn.Dense(self.output_dim)

    def __call__(self, x):
        x = self.hidden_layer(x)
        x = self.custom_activation(x)
        x = self.output_layer(x)
        return x


class ZeroLayersNN(nn.Module):
    N: int
    L: int
    output_dim: int

    @nn.compact
    def __call__(self, x):
        # Vectorized custom activation
        x = CustomActivation(self.N, self.L)(x)
        # Final dense layer
        return nn.Dense(self.output_dim)(x)


class DiscreteNN(nn.Module):
    N: int
    L: int
    output_dim: int

    def setup(self):
        # Replace nn.Dense with BinarizedDense
        self.hidden_layer = BinarizedDense(self.N * self.L)
        self.custom_activation = CustomActivation(self.L)
        self.output_layer = BinarizedDense(self.output_dim)

    def __call__(self, x):
        x = self.hidden_layer(x)
        x = self.custom_activation(x)
        x = self.output_layer(x)
        return x


class TrainState(train_state.TrainState):
    pass


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


@partial(jit, static_argnums=(1))
def loss_fn(params, apply_fn, batch_x, batch_y):
    predictions = apply_fn(params, batch_x)
    return jnp.mean((predictions - batch_y) ** 2)


@jit
def evaluate(carry):
    state, test_batches = carry
    total_loss = 0.0

    def batch_step(carry, batch):
        state, total_loss = carry
        batch_x, batch_y = batch

        loss = eval_step(state, batch_x, batch_y)

        return (
            state,
            total_loss + (loss / len(batch_x)),
        ), None  # Carry updated state, discard output

    (state, total_loss), _ = lax.scan(batch_step, (state, 0.0), test_batches)

    return total_loss


@jit
def eval_step(state, test_batch_x, test_batch_y):
    total_loss = loss_fn(state.params, state.apply_fn, test_batch_x, test_batch_y)
    return total_loss


@jit
def train_step(state, batch_x, batch_y):
    loss, grads = value_and_grad(loss_fn)(
        state.params, state.apply_fn, batch_x, batch_y
    )
    state = state.apply_gradients(grads=grads)
    return state, loss / len(batch_x)


@partial(jit, static_argnums=(1))
def loss_fn_mixed(params, apply_fn, batch_x, batch_y, learned_vecs, true_vecs):
    predictions = apply_fn(params, batch_x)
    loss_preds = jnp.mean((predictions - batch_y) ** 2)
    loss_vecs = jnp.abs(learned_vecs @ true_vecs)
    return loss_preds + loss_vecs


@jit
def evaluate_mixed(state, test_batches, learned_vecs, true_vecs):
    total_loss = 0
    for batch_x, batch_y in test_batches:
        total_loss += loss_fn_mixed(
            state.params, state.apply_fn, batch_x, batch_y, learned_vecs, true_vecs
        )
    return total_loss / len(test_batches)


@jit
def train_step_mixed(state, batch_x, batch_y, learned_vecs, true_vecs):
    loss, grads = value_and_grad(loss_fn_mixed)(
        state.params, state.apply_fn, batch_x, batch_y, learned_vecs, true_vecs
    )
    state = state.apply_gradients(grads=grads)
    return state, loss


def count_params(params):
    total_params = 0
    for _, layer_params in params.items():
        for _, param_value in layer_params.items():
            # Add the number of elements in the array to the total
            total_params += param_value.size
    return total_params
