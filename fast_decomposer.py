"""
CoNAN Symbolic Decomposition (fast) Module
===================================

These functions implement the symbolic decomposition layer of CoNAN.

The module:
    - extracts symbolic block structures,
    - computes commutants,
    - searches for projectors,
    - performs recursive decomposition,
    - and constructs decomposition trees.

The implementation is intentionally lightweight and 
can caputer many of the existing homomorphisms quickly.
"""

import math
from collections import deque

import sympy as sp




class SymbolicBlocks:
    """
    Utilities for symbolic block extraction and rewriting.
    """

    @staticmethod
    def extract_blocks(M, k):

        n = M.shape[0]

        assert n % k == 0, "k must divide matrix size"

        b = n // k

        blocks = {}

        for i in range(k):
            for j in range(k):

                blocks[(i, j)] = M[
                    i*b:(i+1)*b,
                    j*b:(j+1)*b
                ]

        return blocks, b



    @staticmethod
    def rewrite_blocks_general(
        M,
        k,
        prefix="H"
    ):

        blocks, b = SymbolicBlocks.extract_blocks(M, k)

        keys = sorted(blocks.keys())

        base_blocks = []

        block_symbols = {}

        for key in keys:

            B = blocks[key]

            assigned = False

            for idx, base in enumerate(base_blocks):

                label = f"{prefix}{idx}"

                if B.equals(base):

                    block_symbols[key] = label
                    assigned = True

                    break

                if B.equals(-base):

                    block_symbols[key] = "-" + label
                    assigned = True

                    break

            if not assigned:

                base_blocks.append(B)

                idx = len(base_blocks) - 1

                block_symbols[key] = f"{prefix}{idx}"

        unique = len(base_blocks)

        next_symbol = f"{prefix}{unique}"

        return block_symbols, unique, next_symbol



    @staticmethod
    def block_symbol_matrix(symbols, k):

        M_sym = []

        for i in range(k):

            row = []

            for j in range(k):

                entry = symbols[(i, j)]

                if entry.startswith("-"):

                    base = entry[1:]

                    row.append(-sp.Symbol(base))

                else:

                    row.append(sp.Symbol(entry))

            M_sym.append(row)

        return sp.Matrix(M_sym)



    @staticmethod
    def get_symbolic_matrices(M):

        n = M.shape[0]

        k = []

        for i in range(2, 10):

            if n % i == 0 and n != i:
                k.append(i)

        symbolic_blocks = []

        for ki in k:

            element_dict, num, nextsym = (
                SymbolicBlocks.rewrite_blocks_general(
                    M,
                    ki
                )
            )

            element_mat = (
                SymbolicBlocks.block_symbol_matrix(
                    element_dict,
                    ki
                )
            )

            symbolic_blocks.append(element_mat)

        return symbolic_blocks



class GeneratorExtractor:
    """
    Generator extraction utilities.
    """

    @staticmethod
    def extract_generators(M):

        symbols = sorted(
            M.free_symbols,
            key=lambda x: x.name
        )

        gens = []

        for s in symbols:

            gens.append(
                M.applyfunc(
                    lambda x: sp.diff(x, s)
                )
            )

        return gens



# Commutant Computation for the symbolic form

class CommutantAnalyzer:
    """
    Compute commutants and projectors.
    """

 
    # Scale helper
    @staticmethod
    def scale_up(v):

        a = sp.lcm([sp.denom(term) for term in v])

        v = sp.expand(a * v)

        a = sp.gcd([sp.numer(term) for term in v])

        if a != 0:
            v = sp.expand(v / a)

        return v


    # Compute commutant basis [here commutant does not need to be
    # computed as per proposition 3 ]
    @staticmethod
    def commutant_basis(gens):

        k = len(gens)

        n = gens[0].rows

        M = sp.zeros(k*n*n, n*n)

        for t in range(k):

            for i in range(n):

                for j in range(n):

                    for r in range(n):

                        M[
                            t*n*n+n*i+j,
                            n*r+j
                        ] += gens[t][i, r]

                        M[
                            t*n*n+n*i+j,
                            n*i+r
                        ] -= gens[t][r, j]

        B = [

            CommutantAnalyzer.scale_up(b).reshape(n, n)

            for b in M.nullspace()
        ]

        return B


    # Find projector

    @staticmethod
    def find_projector(comm):

        n = comm[0].rows

        I_mat = sp.eye(n)

        x = sp.Symbol('x')

        candidates = list(comm)

        # Pairwise combinations

        for i in range(len(comm)):

            for j in range(i + 1, len(comm)):

                candidates.append(comm[i] + comm[j])

                candidates.append(comm[i] - comm[j])

        # Rational decomposition

        for C in candidates:

            if sp.simplify(
                C - C[0, 0]*I_mat
            ) == sp.zeros(n):

                continue

            poly = C.charpoly(x).as_poly()

            factors = sp.factor_list(poly)[1]

            if len(factors) >= 2:

                return (
                    CommutantAnalyzer
                    .build_projector_from_factors(
                        factors,
                        x,
                        C,
                        n,
                        I_mat
                    )
                )

        # Gaussian rational fallback

        for C in comm:

            if sp.simplify(
                C - C[0,0]*I_mat
            ) == sp.zeros(n):

                continue

            poly = C.charpoly(x).as_poly()

            factors = sp.factor_list(
                poly,
                extension=[sp.I]
            )[1]

            if len(factors) >= 2:

                return (
                    CommutantAnalyzer
                    .build_projector_from_factors(
                        factors,
                        x,
                        C,
                        n,
                        I_mat
                    )
                )

        return None

    # Build projector

    @staticmethod
    def build_projector_from_factors(
        factors,
        x,
        C,
        n,
        I_mat
    ):

        p1_expr = factors[0][0]**factors[0][1]

        p2_expr = 1

        for f, mult in factors[1:]:

            p2_expr *= (f**mult)

        p1 = sp.Poly(p1_expr, x)

        p2 = sp.Poly(p2_expr, x)

        s, t, h = sp.gcdex(p1, p2)

        P_poly = sp.Poly((t * p2) / h, x)

        P = sp.zeros(n)

        C_pow = I_mat

        coeffs = P_poly.all_coeffs()

        coeffs.reverse()

        for c in coeffs:

            if c != 0:

                P += c * C_pow

            C_pow = sp.simplify(C_pow * C)

        return sp.simplify(P)





class RestrictionTools:
    """
    Matrix restriction and block decomposition tools.
    """

    @staticmethod
    def restrict_matrix(M, P):

        n = M.rows

        Q = sp.eye(n) - P

        
        # Basis for Im(P)
        

        colsP = P.columnspace()

        if len(colsP) == 0:
            return None

        B1 = sp.Matrix.hstack(*colsP)

        B1 = sp.Matrix.hstack(*B1.columnspace())

      
        # Basis for Im(I-P)
        

        colsQ = Q.columnspace()

        if len(colsQ) == 0:
            return None

        B2 = sp.Matrix.hstack(*colsQ)

        B2 = sp.Matrix.hstack(*B2.columnspace())

        
        # Change of basis

        T = B1.row_join(B2)

        Tinv = T.inv()

        MT = sp.simplify(Tinv * M * T)

        k = B1.shape[1]

        return (
            MT[:k, :k],
            MT[k:, k:],
            (T, Tinv)
        )



# Decomposition Tree

class DecompositionTree:
    """
    Recursive decomposition tree construction.
    """

    # Variance helper
    
    @staticmethod
    def variance_of_entry(entry, base_var):
        expr = sp.expand(entry)
        
        terms = expr.as_ordered_terms()

        var = 0
        for term in terms:
            coeff = term.as_coeff_Mul()[0]
            var += (coeff**2) * base_var

        return float(var)
    
    

    @staticmethod
    def rms_variance_first_row(M, base_var):

        first_row = M[0, :]

        variances = [

            DecompositionTree.variance_of_entry(entry, base_var)

            for entry in first_row
        ]

        return math.sqrt(
            sum(v**2 for v in variances)
            / len(variances)
        )


    # print tree

    @staticmethod
    def print_tree(node, indent=0):

        prefix = "    " * indent

        print(
            prefix
            + f"Node (size {node['matrix'].shape}):"
        )

        if node["matrix"].shape == (1,1):

            print(
                prefix
                + "  Hom:"
            )

            sp.pprint(
                sp.simplify(node["matrix"][0])
            )

        else:

            sp.pprint(node["matrix"])

        for child in node.get("children", []):

            print(prefix + "  ↓")

            DecompositionTree.print_tree(
                child,
                indent+1
            )


    # BFS decomposition

    @staticmethod
    def decompose_bfs(
        M,
        base_var,
        cut=6,
        blockdim=1
    ):

        def get_dim(M):
            return M.shape[0] * blockdim

        root_dim = get_dim(M)

        root = {
            "matrix": M,
            "children": [],
            "dim": root_dim,
            "root_dim": root_dim,
            "variance":
                DecompositionTree
                .rms_variance_first_row(
                    M,
                    base_var
                )
        }

        queue = deque([root])

        processed = 0

        while queue and processed < cut:

            node = queue.popleft()

            M = node["matrix"]

            parent_dim = node["dim"]

            root_dim = node["root_dim"]

            # Extract generators

            gens = (
                GeneratorExtractor
                .extract_generators(M)
            )

            if gens == []:
                continue


            # Compute commutant

            comm = (
                CommutantAnalyzer
                .commutant_basis(gens)
            )


            # Find projector

            P = (
                CommutantAnalyzer
                .find_projector(comm)
            )

            if P is None:
                continue


            # Restrict matrices

            A, B, T = (
                RestrictionTools
                .restrict_matrix(M, P)
            )


            # Dimensions

            kA = get_dim(A)

            kB = get_dim(B)


            # Variances

            varA = (
                DecompositionTree
                .rms_variance_first_row(
                    A,
                    base_var
                )
            )

            varB = (
                DecompositionTree
                .rms_variance_first_row(
                    B,
                    base_var
                )
            )


            # Child nodes

            left = {
                "matrix": A,
                "children": [],
                "dim": kA,
                "root_dim": root_dim,
                "variance": varA
            }

            right = {
                "matrix": B,
                "children": [],
                "dim": kB,
                "root_dim": root_dim,
                "variance": varB
            }

            node["transform"] = T

            node["children"] = [left, right]

            queue.append(left)

            queue.append(right)

            processed += 2

        return root



# Main Symbolic Decomposer (Fast Decomposer)
class SymbolicDecomposer:
    """
    Main symbolic decomposition interface.
    """

    def __init__(
        self,
        symbolic_matrix,
        n,
        q=None,
        base_var=2/3.
    ):

        self.M = symbolic_matrix

        self.n = n

        self.q = q

        self.base_var = base_var


    # Get symbolic decompositions

    def get_symbolic_matrices(self):

        return (
            SymbolicBlocks
            .get_symbolic_matrices(self.M)
        )


    # Build decomposition trees

    def get_full_trees(
        self,
        verbose=True
    ):
#         print("inside the function")
        symbolic_blocks = (
            self.get_symbolic_matrices()
        )

        full_tree_list = []

        for M in symbolic_blocks:
            print(M)

            blockdim = int(
                self.n / M.shape[0]
            )

            full_tree = (
                DecompositionTree
                .decompose_bfs(
                    M,
                    self.base_var,
                    cut=6,
                    blockdim=blockdim
                )
            )

            if verbose:

                print("Checking:")

                DecompositionTree.print_tree(
                    full_tree
                )

                print(
                    "=" * 60
                )

            full_tree_list.append(
                full_tree
            )

        return full_tree_list
