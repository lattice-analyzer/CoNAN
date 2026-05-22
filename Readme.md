# CoNAN

This repository contains the artifacts associated with the paper and describes how to run
our prototype structure-aware lattice estimator.

# Short Description of Files

The repository contains the following files/modules:

- `examples.py`: Contains multiplication functions for several examples of algebraic rings.
- `compiler.py`: Compiles the multiplication rules for a specific construction and constructs the corresponding lattice.
- `fast_decomposer.py`: Performs symbolic block decomposition and fast heuristic homomorphism search.
- `security_estimator.py`: Estimates the concrete blocksize (`beta`) for each discovered homomorphism (mainly used by the fast decomposer).
- `LWE_estimator_vs_CoNAN.ipynb`: Contains the security estimates for the schemes presented in the paper according to CoNAN
  versus the lattice estimator from [LWE Estimator](https://github.com/malb/lattice-estimator).
- `Primary_Decomposer_nonopt.sage`: The primary decomposer of the algebra performing a more extensive but slower search (non-optimized, typically practical up to `n=128`).
- `Primary_Decomposer_opt.sage`: A more optimized version of the primary decomposer capable of handling larger dimensions.
- `CoNAN_TimeComplexity.ipynb`: A discussion on the time complexity of CoNAN algorithm.
# Installation

```bash
git clone https://github.com/lattice-analyzer/CoNAN.git
cd CoNAN
```

# Requirements

- Python 3.10 or later
- SageMath (required to run the Primary Decomposer)

Required Python packages:
- `lattice-estimator` from [LWE Estimator](https://github.com/malb/lattice-estimator)


- The Deep_Decomposer.ipynb requires sagemath
# Examples.py


- `Deep_Decomposer.ipynb` also requires SageMath.

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

## Constructing Symbolic Matrix Representations

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

The output is the symbolic matrix representation corresponding to the ring `Z[x]/(x^n - 1)`.

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

This returns the basis matrices corresponding to the left matrix representation.

---

## Matrix Representation Types

| Type | Description                                  |
|------|----------------------------------------------|
| `LL` | Left matrix representation                   |
| `LR` | Transpose of the left matrix representation  |
| `RL` | Transpose of the right matrix representation |
| `RR` | Right matrix representation                  |

---

`LL` and `RR` representations are relevant for noncommutative algebras.
`LR` and `RL` are mainly used to compute the commutant more efficiently.

---

# Fast Decomposer

The symbolic decomposer searches for block patterns in the symbolic matrix and performs decomposition on the resulting block matrix rather than directly on the original matrix.

It is significantly faster and can scale to higher dimensions.

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

The previous output gives the best discovered homomorphisms over both the complex and integer settings and estimates the concrete security
according to the best discovered homomorphism by calling the lattice estimator
from [LWE Estimator](https://github.com/malb/lattice-estimator) on the reduced-dimension lattice.

The output also contains the transformations required to obtain the homomorphisms and lift back the solution.

The previous example corresponds to Gentry's attack and shows that the security of composite NTRU
is only about 54 bits (Core-SVP), compared to the 118 bits predicted by the standard lattice estimator.

The example takes approximately 2 minutes on a standard desktop.

---

# Primary Decomposer

The primary decomposer operates directly on the original symbolic matrix representation without
searching for block patterns.

It is slower than the fast decomposer but performs a more extensive search for homomorphisms.

We implemented it in SageMath to avoid the overhead of the `SymPy` library.

We provide two versions of the decomposer:

- `Primary_Decomposer_nonopt.sage`: Easier to understand and closer to the underlying algorithmic logic, but not fully optimized.
- `Primary_Decomposer_opt.sage`: More optimized and capable of handling larger dimensions.

---

# Primary Decomposer Example

```sage
sage: load("Primary_Decomposer_nonopt.sage")
sage: hom = homomorphism('examples.py', 'mul_NTRU', [64], 'LL', False)
```

or to run multiple workers in parallel:

```sage
sage: hom = homomorphism_parallel(
       filename    = 'examples.py',
       funcname    = 'mul_NTRU',
       args        = [128],       # add more parameters if your function require more than (f, g, h, n) as parameters
       mul_type    = 'LL',
       do_print    = False,
       num_workers = 8,
)
```

You can inspect the discovered homomorphisms by printing:

```sage
print(hom[i])
```

`Primary_Decomposer_opt.sage` can be used in a similar way while targeting larger dimensions and will compute blocksize esimation
per founded homomorphisms.

For instance,
```sage
sage: load("Primary_Decomposer_opt.sage")
sage: homs=homomorphism("examples.py", "mul_BQTRU",[256], mul_type='LL', do_print=False,
....:         num_workers=1, q=512, base_var=2/3, lattice='NTRU',
....:         task_timeout=300, field='QQ')

```
takes a few minutes on a standard desktop and estimate betas for found homomorphisms.
# Project Structure

```text
CoNAN/
├── examples.py
├── compiler.py
├── fast_decomposer.py
├── security_estimator.py
├── Primary_Decomposer_nonopt.sage
├── Primary_Decomposer_opt.sage
├── LWE_estimator_vs_CoNAN.ipynb
├── CoNAN_TimeComplexity.ipynb
└── README.md
```
