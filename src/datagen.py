


class DifferentialEquation:

    def __init__(self, n_eqs, n_terms, n_mult, non_lins, sym_non_lins = None):
        """
            Initialize differential equations generator.

            Parameters:
            n_eqs: number of equations to generate;
            n_terms: max number of addends in each equation;
            n_mult: max number of multiplicands in each addend;
            non_lins: list of all the possible non linearities;
            sym_non_lins: list of the symbolic expression of the non linearities.
        """

        # Initialize variables
        self.n_eqs = n_eqs
        self.n_terms = n_terms
        self.n_mult = n_mult
        self.non_lins = non_lins
        self.sym_non_lins = sym_non_lins

        # Initialize lists for storing the equations
        # and their symbolic expression
        self.equations = []
        self.sym_expr = []

    
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
        