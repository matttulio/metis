![Metis Logo](metis_logo.jpg)

# metis

## Description
Metis is named after the Greek Oceanid symbol of wisdom and deep thought. In mythology, Metis was known for her intelligence and cunning, and she played a key role as the mother of Athena, the goddess of wisdom. Daughter of Oceanus and Tethys, Metis was revered not only for her intellect but also for her transformative nature, symbolizing the ever-changing and adaptive spirit of the sea. This repository draws inspiration from her legacy, aiming to bring thoughtful and innovative contributions to the problem of data-driven symbolic regression, and also of data-driven resolution of such equations.

## Table of Contents
- [Installation](#installation)
- [Folder Structure](#folder-structure)
- [Generating the equations](#generating-the-equations)
- [Saving the symbolic expressions](#saving-the-symbolic-expressions)

## Installation
The conda environment related to this work can be installed using:
```bash
  conda env create -f metis_env.yaml
```
or, if one uses the mamba package manager:
```bash
  mamba env create -f metis_env.yaml
```

## Folder Structure
The repository is organized as follows:

- `src/`: Contains the source code for the project.
  - `datagen.py`: Contains the `Equations` class for generating systems of random differential equations and their symbolic forms.
- `data/`: Directory for storing generated data and results.
- `Test_Datagen.ipynb`: Notebook to present the basic functionalities of Equations class.
- `metis_env.yaml`: Conda environment file for setting up the project dependencies.
- `README.md`: Project documentation.

## Generating the equations

To use the `Equations` class from `datagen.py`, you need to provide the following parameters:

### Required Parameters

1. `n_vars` (int): 
   - Number of variables in the system of equations
   - Example: `n_vars = 3` for a system with three variables

2. `n_eqs` (int):
   - Number of equations to generate
   - Example: `n_eqs = 3` to generate a system of three equations

3. `max_addends` (int):
   - Maximum number of addends in each equation
   - Example: `max_addends = 3` means each equation can have up to 3 terms added together

4. `max_multiplicands` (int):
   - Maximum number of multiplicands in each addend
   - Example: `max_multiplicands = 2` means each term can have up to 2 factors multiplied together

5. `non_lins` (tuple of functions):
   - List of all possible non-linear functions to use
   - Example: `non_lins = (jnp.sin, jnp.cos)` to use sine and cosine functions

### Optional Parameters

6. `sym_non_lins` (list of strings, optional):
   - Symbolic expressions of the non-linearities for equation visualization
   - Example: `sym_non_lins = [r"\sin", r"\cos"]` for LaTeX representation

7. `distribution` (str, optional):
   - Type of distribution to use for variable probabilities ('uniform', 'beta', 'lognormal', 'custom')
   - Example: `distribution = "uniform"`

8. `a` (float, optional):
   - Parameter 'a' for beta distribution (required if `distribution` is 'beta')
   - Example: `a = 2.0`

9. `b` (float, optional):
   - Parameter 'b' for beta distribution (required if `distribution` is 'beta')
   - Example: `b = 5.0`

10. `sigma` (float, optional):
    - Parameter 'sigma' for lognormal distribution (required if `distribution` is 'lognormal')
    - Example: `sigma = 1.0`

11. `p` (array-like, optional):
    - Custom probabilities for the variables (required if `distribution` is 'custom')
    - Example: `p = jnp.array([0.2, 0.3, 0.5])`

12. `seed` (int, optional):
    - Seed for reproducibility (default: 42)
    - Example: `seed = 42`

### Example Usage

```python
from src.datagen import Equations
import jax.numpy as jnp

# Define the non-linearities
non_lins = (jnp.sin, jnp.cos)
sym_non_lins = [r"\sin", r"\cos"]

# Configure the system
config = {
    "n_vars": 3,
    "n_eqs": 3,
    "max_addends": 3,
    "max_multiplicands": 2,
    "non_lins": non_lins,
    "sym_non_lins": sym_non_lins,
    "distribution": "uniform",
    "a": None,
    "b": None,
    "sigma": None,
    "p": None,
    "seed": 42
}

# Create the system of equations
system = Equations(**config)
```

## Saving the symbolic expressions

The `save_symb_expr` function is used to generate and save the symbolic expressions of the equations in the system. To use this function, you need to provide the following parameters:

### Required Parameters

1. `filename` (str): 
   - The name of the file where the symbolic expressions will be saved.
   - Example: `filename = "equations_symbolic.pdf"`

### Optional Parameters

2. `max_eq_per_page` (int, optional):
   - The maximum number of equations in each page of the PDF (default: 35).
   - Example: `max_eq_per_page = 30`

### Example Usage

```python
system.save_symb_expr(filename="equations_symbolic.pdf", max_eq_per_page=30)
```
