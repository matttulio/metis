from jax import random
import jax.numpy as jnp
from collections import Counter


class DifferentialEquation:

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

    
    def non_linearity_symbol(self, nl):
        """
            Define the LaTeX symbolic expression for each non linearity.

            Parameters:
            nl: proposed non linearity.
        """

        # If the symbols are not defined a priori, use generic names
        if(self.sym_non_lins == None):

            index = self.non_lins.index(nl) if nl in self.non_lins else None  # Check if the non linearity candidate is in the list
            return f"\\text{{nl}}_{index}" if index is not None else "\\text{unknown\\_non\\_linearity}"
            
        else:

            index = self.non_lins.index(nl) if nl in self.non_lins else None
            return self.sym_expr[index] if index is not None else "\\text{unknown\\_non\\_linearity}"
        


    def generate_equations(self):
        """
            Generate random differential equations.
        """
        # Random generation of the number of addends, and multiplicands
        n_sum_terms = random.randint(self.key, shape=(self.n_eqs,), minval=1, maxval=self.max_sum_terms+1)
        n_mult_terms = []
        k = 0
        while(len(self.equations) < self.n_eqs):

            equation = []
            sym_eq = f"\\frac{{dC_{{{i+1}}}}}{{dt}} = "
            for i in range(len(n_sum_terms)):
                n_mult_terms = random.randint(self.key, shape=(n_sum_terms[i],), minval=1, maxval=self.max_mult_terms+1)
                non_lins_idxs = random.randint(self.key, shape=(n_mult_terms,), minval=1, maxval=len(self.non_lins))
                term = jnp.prod(self.non_lins[non_lins_idxs])
                sym_term = ''.join(self.sym_non_lins[non_lins_idxs])
                sym_eq += sym_term 
                equation.append(term)

            self.equations.append(equation)

        self.equations = list(set(self.equations))
