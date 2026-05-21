# CoNAN

This repository contains the artifacts associated with the article and describes how to run
our prototype structure-aware lattice estimator.

# Short Descriptions of Files

The repository contains the following files/modules:

- `examples.py`: Contains multiplication functions for many examples of algebraic rings.
- `compiler.py`: Compiles the multiplication rules for a specific construction and constructs the lattice.
- `fast_decomposer.py`: Symbolic block decomposition and fast heuristic homomorphism search.
- `Deep_Decomposer.ipynb`: The primary decomposer of the algebra and performs a more extensive but slower search.
- `security_estimator.py`: Used to estimate the concrete blocksize (`beta`) for each found homomorphism (mainly used by the fast decomposer).
- `LWE_estimator_vs_CoNAN.ipynb`: Contains the security estimates for the schemes presented in the paper according to CoNAN
  vs. the lattice estimator from [LWE Estimator](https://github.com/malb/lattice-estimator).

# Installation

```bash
git clone https://github.com/lattice-analyzer/CoNAN.git
cd CoNAN
```

# Requirements

- Python 3.10 or later

Required Python packages:
- `lattice-estimator` from [LWE Estimator](https://github.com/malb/lattice-estimator)

- The Deep_Decomposer.ipynb requires sagemath
# examples.py

This file contains examples of multiplication functions written for several lattice-based constructions.

---

## Example: NTRU Multiplication

```python
# Z[x]/(x^n - 1)

def mul_NTRU(f, g, h, n):

    for i in range(n):

        for j in range(n):

            h[(i+j)%n] += (
                f[i] * g[j]
            )
```

---

## Example: 2RLWE

```python
# Z[x,y]/(x^N1+1, y^N2+1)

def mul_2RLWE(f, g, h, N1, N2):

    for i in range(N2):

        for j in range(N2):

            for k in range(N1):

                for l in range(N1):

                    t = (
                        (-1)**(
                            ((i+j)//N2)
                            +
                            ((k+l)//N1)
                        )
                    )

                    idx = (
                        ((i+j)%N2)*N1
                        +
                        ((k+l)%N1)
                    )

                    h[idx] += (
                        t
                        * f[i*N1+k]
                        * g[j*N1+l]
                    )
```

---

## Multiplication Function Format

A multiplication function must follow the template:

```python
h[...] += c * f[...] * g[...]
```

where:

- `f` and `g` are read-only coefficient vectors,
- `h` is the output vector,
- multiplication is bilinear.

---

# Compiler

The compiler extracts the algebraic structure directly from the multiplication rules.

---

## Constructing Symbolic Matrix Representation

```python
import compiler

n = 512

H = compiler.construct_symbolic_matrix(
    filename="examples.py",
    funcname="mul_NTRU",
    dimension=n,
    mul_type="LL",
    variable="h"
)
```

The output is the symbolic matrix representation corresponding to polynomials from the ring `Z[x]/(x^n - 1)`.

---

## Constructing Lattice Bases

```python
L = compiler.construct_lattice(
    filename="examples.py",
    funcname="mul_NTRU",
    dimension=n,
    mul_type="LL"
)
```

This returns the basis matrices of the left matrix representation.

---

## Matrix Representation

| Type | Description                                  |
|------|----------------------------------------------|
| `LL` | Left matrix representation                   |
| `LR` | Transpose of the left matrix representation  |
| `RL` | Transpose of the right matrix representation |
| `RR` | Right matrix representation                  |

---

`LL` and `RR` multiplication matter for noncommutative algebras.
`LR` and `RL` are used to compute the commutant more efficiently.

---

# Fast Decomposer

The symbolic decomposer searches for block patterns in the symbolic matrix and performs decomposition on the block matrix rather than the input matrix.

It is fast and can run up to higher dimensions.

---

# Security Estimator

Based on the discovered dimension-reduction homomorphisms, the security estimator evaluates the blocksize required to recover
the private key or message through the different attack paths and returns the best global decomposition (including over `Q[i]`)
together with the best integer homomorphism defining the concrete security.

---

## Example

```python
import compiler
import fast_decomposer as fastdec

################## NTRU composite #######################

n = 512
q = 7681
base_var = 4/3.

############################################################################
# Initialize estimator
############################################################################

estimator = SecurityEstimator(
    n=n,
    q=q,
    base_var=base_var,
    mul_function="mul_NTRU",
    level=1,    #### fast decomposer
    lattice="NTRU",
    mul_type="LL",
    filename="examples.py"
)

print("Checking dimension-reduction homomorphisms, this may take a few minutes!")

############################################################################
# Run estimation
############################################################################

result = estimator.estimate_security(
    verbose=True
)

############################################################################
# Print result
############################################################################

print(result)
print("security according to CoNAN: ", result['concrete_beta'] * 0.292)
```

---

## Example Output

```text
{'best_global': {'path': [[512, 1.4884, 'integer', 'combine', {'transform': (Matrix([
[1/4,  3/4, -1/4, -1/4],
[1/4, -1/4,  3/4, -1/4],
[1/4, -1/4, -1/4,  3/4],
[1/4, -1/4, -1/4, -1/4]]), Matrix([
[1, 1, 1,  1],
[1, 0, 0, -1],
[0, 1, 0, -1],
[0, 0, 1, -1]])), 'transform_examples': None}], (128, 5.9536, 'integer', 'solve', {'matrix': Matrix([[H0 + H1 + H2 + H3]])}), [384, 2.9768, 'integer', 'combine', {'transform': (Matrix([
[1/2,  1/2, 1/2],
[  0,    0,   1],
[1/2, -1/2, 1/2]]), Matrix([
[1, -1,  1],
[1,  0, -1],
[0,  1,  0]])), 'transform_examples': None}], (128, 5.9536, 'integer', 'solve', {'matrix': Matrix([[H0 - H1 + H2 - H3]])}), [256, 2.9768, 'integer', 'combine', {'transform': (Matrix([
[ 1/2, 1/2],
[-I/2, I/2]]), Matrix([
[1,  I],
[1, -I]])), 'transform_examples': None}], (128, 5.9536, 'complex', 'solve', {'matrix': Matrix([[H0 - I*H1 - H2 + I*H3]])}), (128, 5.9536, 'complex', 'solve', {'matrix': Matrix([[H0 + I*H1 - H2 - I*H3]])})], 'beta': 187, 'complex': True, 'critical_node': 5, 'tree_index': 1}, 'best_global_integer': {'path': [[512, 1.4884, 'integer', 'combine', {'transform': (Matrix([
[1/2,  1/2],
[1/2, -1/2]]), Matrix([
[1,  1],
[1, -1]])), 'transform_examples': None}], (256, 2.9768, 'integer', 'solve', {'matrix': Matrix([[H0 + H1]])}), (256, 2.9768, 'integer', 'solve', {'matrix': Matrix([[H0 - H1]])})], 'beta': 187, 'complex': False, 'critical_node': 1, 'tree_index': 0}, 'concrete_beta': 187}

security according to CoNAN: 54.604
```

The previous output gives the best found homomorphisms over the complex and integer settings and estimates the concrete security
according to the best found homomorphism using the security estimates for integer-entry lattices by calling the lattice estimator
from [LWE Estimator](https://github.com/malb/lattice-estimator) on the reduced-dimension lattice.

The output contains the transformations required to obtain the homomorphisms and lift back the solution.

The previous example corresponds to Gentry's attack and shows that the security of composite NTRU
is only about 54 bits (Core-SVP) compared to the 118 bits predicted by the standard lattice estimator.

The example takes around 2 minutes on a standard desktop.

---

# Deep Decomposer

The deep decomposer operates directly on the original symbolic matrix representation without
looking for block patterns.

It is slower than the fast decomposer but extensively searches for homomorphisms.

The code of the Deep Decomposer in not fully optimized and Our current implementation can run up to `n=128`
and we have implemented is sage to avoid the overhead o  `Sympy`

---

# Deep Decomposer Example

You can check examples in Primary-Decomposer.ipynb

# Project Structure

```text
CoNAN/
├── examples.py
├── compiler.py
├── fast_decomposer.py
├── security_estimator.py
├── Deep-Decomposer.ipynb
├── LWE_estimator_vs_CoNAN.ipynb
└── README.md
```
