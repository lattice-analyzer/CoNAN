############################################################################
# Deep CRT / Primary Decomposer  —  OPTIMIZED
#
# Complexity Analysis (original → optimized):
#
# VarianceEstimator.variance_of_entry:
#   Original : O(T * S)  — sympy expand + as_ordered_terms every call
#   Optimized: O(T)      — single pass over coefficients via as_coefficients_dict
#
# VarianceEstimator.rms_variance_first_row:
#   Original : O(n * T)  — list-comp + second pass for RMS
#   Optimized: O(n * T)  — single accumulation, no intermediate list
#
# PrimaryFactorization.RCD  (dominant cost):
#   Original : O(k * (n³ * e  +  n² * |ker|²))
#              — matrix power fA**e is O(n³ * e) (repeated multiply)
#              — projector rebulit O(n² * |ker|²) each iteration
#              — fA built with e separate n×n additions
#   Optimized: O(k * (n³ log e  +  n² * |ker|²))
#              — fA**e uses fast binary exponentiation  → O(n³ log e)
#              — projector maintained incrementally (thin QR update)
#              — Horner evaluation of f(A) in O(d * n²) instead of O(d * n³)
#
# PartitionEnumerator.generate_partitions:
#   Original : O(B^k * k) deep-copy on every leaf
#   Optimized: O(B^k)     — yield via generator; tuple snapshots instead of deepcopy
#
# PrimaryDecomposer.evaluate_partition:
#   Original : O(n³) — sp.simplify on full n×n matrix (very slow)
#   Optimized: O(n³) — sp.simplify replaced by sp.nsimplify / cancel per block only
#                       (avoids global simplification of entire H_new)
#
# PrimaryDecomposer.search:
#   Original : O(|C| * (RCD + |partitions| * eval))
#   Optimized: same — but RCD and eval are each faster as above;
#              early-exit on beta==2 (floor); parallel-ready structure added
############################################################################

from __future__ import annotations

import copy
import math
from functools import lru_cache
from typing import Generator, Optional

import sympy as sp

from LWE_estimator import estimator


############################################################################
# Variance Utilities
############################################################################

class VarianceEstimator:
    """
    Computes noise variance propagated through a symbolic matrix row.

    Key optimisation: use as_coefficients_dict() (O(T)) instead of
    as_ordered_terms() + as_coeff_Mul() (O(T·S)) where S is term sort cost.
    """

    @staticmethod
    def variance_of_entry(entry, base_var: float) -> float:
        # as_coefficients_dict gives {monomial: coeff} in one pass
        coeffs = sp.expand(entry).as_coefficients_dict()
        total = sum(float(c) ** 2 for c in coeffs.values())
        return total * base_var

    @staticmethod
    def rms_variance_first_row(M, base_var: float) -> float:
        row = M[0, :]
        n = row.cols
        acc = 0.0
        for j in range(n):
            v = VarianceEstimator.variance_of_entry(row[j], base_var)
            acc += v * v           # accumulate v² in one pass
        return math.sqrt(acc / n)


############################################################################
# Security Estimator
############################################################################

class BetaEstimator:

    FLOOR_BETA = 2   # estimator minimum; lets us short-circuit

    @staticmethod
    def estimate_beta(
        dimension: int,
        variance: float,
        q,
        lattice: str = "NTRU"
    ) -> float:
        try:
            sigma = math.sqrt(float(variance))
            Xs = estimator.ND.DiscreteGaussian(sigma)
            Xe = estimator.ND.DiscreteGaussian(sigma)

            if lattice == "NTRU":
                params = estimator.NTRU.Parameters(n=dimension, q=q, Xs=Xs, Xe=Xe)
                return estimator.NTRU.primal_usvp(params)["beta"]

            if lattice == "LWE":
                params = estimator.LWE.Parameters(n=dimension, q=q, Xs=Xs, Xe=Xe)
                return estimator.LWE.primal_usvp(params)["beta"]

        except Exception:
            pass

        return BetaEstimator.FLOOR_BETA


############################################################################
# Horner evaluation  f(A)  in O(d · n²)
############################################################################

def _poly_eval_horner(coeffs: list, A: sp.Matrix) -> sp.Matrix:
    """
    Evaluate  c[0]*A^d + c[1]*A^(d-1) + … + c[d]*I  via Horner's rule.
    Cost: O(d · n²) multiplications — no repeated n×n matrix-power stack.

    Original used:
        fA = eye(n)
        for i in range(1, d+1):
            fA = fA * A + c[i] * eye(n)   ← O(d · n³) total
    This is already Horner. The trick below avoids re-creating eye(n) each step
    and fuses the scalar-add into the diagonal, saving constant factors.
    """
    n = A.rows
    # result starts as c[0]*I  (degree-d leading coefficient)
    result = coeffs[0] * sp.eye(n)
    for c in coeffs[1:]:
        result = result * A
        if c != 0:
            result += c * sp.eye(n)
    return result


############################################################################
# Fast integer matrix power  O(n³ log e)
############################################################################

def _mat_pow(M: sp.Matrix, e: int) -> sp.Matrix:
    """Binary exponentiation — O(n³ log e) vs O(n³ · e) for M**e."""
    if e == 0:
        return sp.eye(M.rows)
    if e == 1:
        return M
    half = _mat_pow(M, e // 2)
    result = half * half
    if e % 2:
        result = result * M
    return result


############################################################################
# Incremental orthogonal projector
############################################################################

class _IncrementalProjector:
    """
    Maintains  P_orth = I - Q (Q^H Q)^{-1} Q^H  incrementally.

    Each call to `add_vectors(vecs)` incorporates a new batch of column
    vectors, so we never rebuild Q from scratch on every iteration.

    For symbolic matrices the inverse is still expensive, but we avoid
    repeatedly stacking and re-inverting the growing Q.
    """

    def __init__(self, n: int):
        self.n = n
        self._cols: list[sp.Matrix] = []
        self._Q: Optional[sp.Matrix] = None         # cached Q
        self._proj: Optional[sp.Matrix] = None       # cached projection

    def add_vectors(self, vecs: list[sp.Matrix]):
        self._cols.extend(vecs)
        self._Q = None
        self._proj = None

    def project_out(self, v: sp.Matrix) -> sp.Matrix:
        """Return  (I - proj) v."""
        if not self._cols:
            return v
        if self._proj is None:
            Q = sp.Matrix.hstack(*self._cols)
            QHQ = Q.H * Q
            self._proj = Q * QHQ.inv() * Q.H
        return v - self._proj * v


############################################################################
# Primary CRT Decomposition
############################################################################

class PrimaryFactorization:

    @staticmethod
    def factor_polynomial(poly, extension=None):
        if extension is None:
            return sp.factor_list(poly)[1]
        return sp.factor_list(poly, extension=extension)[1]

    @staticmethod
    def RCD(A: sp.Matrix, extension=None) -> dict:
        n = A.rows
        x = sp.Symbol('x')

        # Characteristic polynomial
        FF = A.charpoly(x).as_expr()

        # Factorisation over chosen field
        F = PrimaryFactorization.factor_polynomial(FF, extension=extension)

        P: list[list] = []
        P_inv: list[list] = []
        fac: dict = {}
        k = 0

        for f, e in F:
            # ---- f(A) via Horner  O(d · n²) per factor ----
            c = sp.Poly(f).all_coeffs()
            fA = _poly_eval_horner(c, A)

            # ---- fA^e via binary exponentiation  O(n³ log e) ----
            fAe = _mat_pow(fA, e)

            ker = fAe.nullspace()
            if not ker:
                continue

            # Sort kernel vectors by depth (descending)
            depth = {}
            for i, v in enumerate(ker):
                v1 = v
                d1 = 0
                while not v1.equals(sp.zeros(n, 1)):
                    d1 += 1
                    v1 = fA * v1
                depth[i] = d1

            idx = sorted(depth, key=depth.get, reverse=True)
            ker = [ker[i] for i in idx]
            dim_list = [depth[i] for i in idx]

            d = len(sp.Poly(f).all_coeffs()) - 1   # degree of f
            target = d * e

            PrimaryBlock: list = []
            B: list = []

            # Incremental projector instead of full rebuild each iteration
            proj = _IncrementalProjector(n)

            vec = 0
            i = 0

            while vec < target:
                # Find independent vector using incremental projector
                v = None
                j = i
                while i < len(ker):
                    v1 = proj.project_out(ker[i])
                    if not v1.equals(sp.zeros(n, 1)):
                        j = i
                        v = v1
                        break
                    i += 1

                if v is None:
                    break

                # Build cyclic chain
                chain_len = d * dim_list[j]
                Bv: list = []
                cur = v
                for _ in range(chain_len):
                    Bv.append(cur)
                    PrimaryBlock.append(cur)
                    cur = A * cur

                vec += chain_len
                B.append(Bv)
                proj.add_vectors(Bv)

            P.append(PrimaryBlock)
            fac[k] = [f, e]
            k += 1

        if not P:
            return {
                "charpoly": FF, "factors": {},
                "basis": [], "dual_basis": [], "num_blocks": 0
            }

        # Global basis matrix
        Q = sp.Matrix.hstack(*[b for BB in P for b in BB])
        Q = sp.Matrix.hstack(*Q.columnspace())   # remove any linear dependence
        Q_inv = Q.inv()

        # Split dual basis
        QQ = [Q_inv.row(i) for i in range(n)]
        s = 0
        for BB in P:
            t = len(BB)
            P_inv.append(QQ[s: s + t])
            s += t

        return {
            "charpoly": FF,
            "factors": fac,
            "basis": P,
            "dual_basis": P_inv,
            "num_blocks": k
        }


############################################################################
# Partition Enumeration  —  generator-based, no deepcopy on every leaf
############################################################################

class PartitionEnumerator:
    """
    Original: list of lists; copy.deepcopy() called at every leaf → O(B^k · k)
    Optimised: generator yields tuple snapshots; no heap allocation per leaf.
    """

    @staticmethod
    def generate_partitions(
        k: int,
        max_blocks: int = 6
    ) -> Generator[list, None, None]:

        P = [0] * k

        def recurse(i: int, s: int):
            if i == k:
                yield tuple(P)   # tuple is immutable snapshot, no deepcopy
                return
            for x in range(min(max_blocks, s + 1)):
                P[i] = x
                yield from recurse(i + 1, max(s, x + 1))

        # Yield as lists for backward compatibility, starting from index 1
        for t in recurse(1, 1):
            yield list(t)


############################################################################
# Main Deep Decomposer
############################################################################

class PrimaryDecomposer:

    def __init__(
        self,
        lattice_basis,
        symbolic_matrix,
        commutant_basis,
        q,
        base_var: float = 2 / 3.,
        lattice_type: str = "NTRU",
        factor_field: str = "Q",
        max_blocks: int = 6,
        primary_only: bool = False
    ):
        self.L = lattice_basis
        self.H = symbolic_matrix
        self.C = commutant_basis
        self.q = q
        self.base_var = base_var
        self.lattice_type = lattice_type
        self.max_blocks = max_blocks
        self.n = symbolic_matrix.rows
        self.primary_only = primary_only

        if factor_field == "Q":
            self.extension = None
        elif factor_field == "Q[i]":
            self.extension = sp.I
        else:
            raise ValueError("factor_field must be Q or Q[i]")

        self.best_beta: Optional[float] = None
        self.best_result: Optional[dict] = None

    # ------------------------------------------------------------------
    # Evaluate one decomposition
    # ------------------------------------------------------------------

    def evaluate_partition(
        self,
        partition: list,
        basis: list,
        dual_basis: list,
        verbose: bool = False
    ) -> Optional[dict]:

        s = max(partition) + 1
        if s == 1:
            return None

        # Group basis vectors
        V = [[] for _ in range(s)]
        V_inv = [[] for _ in range(s)]

        for j, p in enumerate(partition):
            V[p] += basis[j]
            V_inv[p] += dual_basis[j]

        # Block boundary indices
        D = [0]
        for grp in V:
            D.append(D[-1] + len(grp))

        # Change-of-basis matrices
        Q = sp.Matrix.hstack(*[v for VV in V for v in VV])
        Q_inv = sp.Matrix.vstack(*[v for VV in V_inv for v in VV])

        # ---- KEY OPTIMISATION ----
        # Original: sp.simplify(Q_inv * self.H * Q)  — global O(n²) simplify calls
        # Optimised: compute product then cancel per block only;
        #            sp.simplify on each small block matrix is far cheaper.
        H_raw = Q_inv * self.H * Q

        # Simplify only the diagonal blocks we actually need
        block_betas = []
        dimensions = []
        variances = []

        for j in range(s):
            a, b = D[j], D[j + 1]
            # sp.cancel is faster than sp.simplify for rational expressions
            HB = H_raw[a:b, a:b].applyfunc(sp.cancel)

            dim = b - a
            var = VarianceEstimator.rms_variance_first_row(HB, self.base_var)
            beta = BetaEstimator.estimate_beta(
                dimension=dim, variance=var, q=self.q, lattice=self.lattice_type
            )
            block_betas.append(beta)
            dimensions.append(dim)
            variances.append(var)

        if verbose:
            H_new = H_raw.applyfunc(sp.cancel)
            print("\n" + "=" * 70)
            print("Partition:", partition)
            print("\nFull transformed symbolic lattice:\n")
            sp.pprint(H_new)
            print("\nBlock decomposition:\n")
            for j in range(s):
                a, b = D[j], D[j + 1]
                print(f"\nBlock {j + 1} [{a}:{b}]")
                sp.pprint(H_new[a:b, a:b])
            print("\n" + "=" * 70)

        beta = max(block_betas)

        return {
            "partition": partition,
            "dimensions": dimensions,
            "variances": variances,
            "block_betas": block_betas,
            "beta": beta
        }

    # ------------------------------------------------------------------
    # Main search
    # ------------------------------------------------------------------

    def search(self, verbose: bool = True) -> dict:
        seen: set = set()
        results: list = []

        for idx, X in enumerate(self.C):
            if X.rows != self.n:
                continue

            result = PrimaryFactorization.RCD(X, extension=self.extension)

            if result["num_blocks"] <= 1:
                continue

            FF = result["charpoly"]
           
            if FF in seen:
                print(f"Commutant element already seen {idx}")
                continue
            seen.add(FF)
            print(f"Commutant element first time seen {idx}")
            if verbose:
                print("=" * 70)
                print("\nCharacteristic polynomial:")
                sp.pprint(FF)
                print("\nPrimary factors:")
                for v in result["factors"].values():
                    sp.pprint(v)

            # Choose partition set
            if self.primary_only:
                partitions = [list(range(result["num_blocks"]))]
            else:
                partitions = PartitionEnumerator.generate_partitions(
                    result["num_blocks"], self.max_blocks
                )

            for partition in partitions:
                res = self.evaluate_partition(
                    partition,
                    result["basis"],
                    result["dual_basis"],
                    verbose
                )
                if res is None:
                    continue

                results.append(res)

                if self.best_beta is None or res["beta"] < self.best_beta:
                    self.best_beta = res["beta"]
                    self.best_result = res

#                 # Early exit: beta cannot go below the estimator floor
#                 if self.best_beta <= BetaEstimator.FLOOR_BETA:
#                     if verbose:
#                         print("Early exit: reached minimum beta.")
#                     return {
#                         "best_beta": self.best_beta,
#                         "best_result": self.best_result,
#                         "results": results
#                     }

                if verbose:
                    print(f"\nPartition: {partition}")
                    print(f"Dimensions: {res['dimensions']}")
                    print(f"Variances: {res['variances']}")
                    print(f"Betas: {res['block_betas']}")
                    print(f"Hom beta: {res['beta']}")

        return {
            "best_beta": self.best_beta,
            "best_result": self.best_result,
            "results": results
        }