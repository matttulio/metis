import jax
import jax.numpy as jnp

# Run a simple computation on the GPU
x = jnp.ones((1000, 1000))
y = jax.device_put(x)  # Send data to the GPU
z = y * 2

# Print the device
print(f"Computation done on: {z.device}")
