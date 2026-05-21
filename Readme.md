# CoNAN

This repository contains the artifacts associated to the article and describe how to run
our prototype of the structure-aware lattice estimator.

# Short Descriptions of files

The repository contains the following files/Modules:
- `examples.py`: contains multiplication functions for many examples of algebraic rings.
- `compiler.py`: Compiles the multiplication rules for a specific construction and construct the lattice.
- `fast-decomposer.py`: Symbolic block decomposition and fast heuristic homomorphism search.
- `deep_decomposer.py`: The primary decomposer of the algebra and perform extensive and slower search.
- `LWE_estimator_vs_CoNAN.ipynb`: Contains the security estimate for the schemes presented in the paper according to CoNAN
vs. the lattice estimator from [LWE Estimator](https://github.com/malb/lattice-estimator).


# Installation

```bash
git clone <repository>
cd CoNAN
```

# Requirements
- python 3.10 or later

- `sympy`
- `numpy`
- `sage`
- `lattice-estimator from `[LWE Estimator](https://github.com/malb/lattice-estimator)



# Example.py
This file contains examples of the multiplication functions written for some lattice-based constructions.

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

# Compiler.py

The compiler extracts algebra structure directly from multiplication routines written in a restricted bilinear form.

The compiler supports:

---

The compiler parses the multiplication rule symbolically and reconstructs the full algebra structure tensor.


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

The output is the symbolic matrix representation of polynomials from the ring `Z[x]/(x^n - 1)`

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

## Matrix representation

| Type | Description                                  |
|---|----------------------------------------------|
| `LL` | Left matrix representation                   |
| `LR` | Transpose of the left matrix representaiton  |
| `RL` | Transpose of the right matrix representation |
| `RR` | Right matrix representation                  |

---
`LL` multiplication and `RR` multiplication matters for noncommutative algebra.
`LR` and `RL` are used to compute the commutant in more efficient way.

---
# Fast Decomposer

The symbolic decomposer searches for block decompositions directly from the symbolic block
form of the matrix representation. It is fast and can run upto higher dimensions

---

## Example

```python
import compiler
import fast_decomposer as fastdec
################## NTRU composite #######################
n = 16
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

print("Checking dimension reduction homomorphisms, may take few minutes!")
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
print("security according to CoNAN: ", result['concrete_beta']*0.292)

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
security according to CoNAN:  54.604
```

The output example gives the best found homomorphism over complex and integer and estimate the 
concrete security according to the best found homomorphism as per estimated for integer entries lattices
by calling the lattice estimator of [LWE Estimator](https://github.com/malb/lattice-estimator) for the reduced dimension lattice.

The output contains the transformer to get the homomorphisms and lift back the solution.

---

# Deep Decomposer

The fast decomposer operates directly on original symbolic matrix representation without
looking for block patterns. 

It is slower than the fast decomposer but look for more possible homomorphism.

Our current implementation is slow and can be further optimized/

---

# Deep Decomposer Example

```python
import compiler
import fast_decomposer as fastdec

############################################################################
# Parameters
############################################################################

n = 8
q = 1024
base_var = 2/3.

############################################################################
# Construct symbolic matrix
############################################################################

H = compiler.construct_symbolic_matrix(
    filename="examples.py",
    funcname="mul_NTRU",
    dimension=n,
    mul_type='LL',
    variable='h'
)

############################################################################
# Initialize symbolic decomposer
############################################################################

decomposer = fastdec.SymbolicDecomposer(
    symbolic_matrix=H,
    n=n,
    q=q,
    base_var=base_var
)

############################################################################
# Search decompositions
############################################################################

decomposer.get_full_trees(
    verbose=True
)
```

---

# Fast Decomposer Output

The fast decomposer prints:

- detected symbolic block structures,
- candidate homomorphisms,
- transformed symbolic lattices,
- estimated attack costs.

Example:

```text
Partition:
[0, 0, 1]

Full transformed symbolic lattice:

[h0+h1+h2+h3            0              0             0]
[0            h0-h1+h2-h3              0             0]
[0                       0         h0-h2      -h1+h3]
[0                       0         h1-h3       h0-h2]
```








# Project Structure

```text
CoNAN/
│
├── compiler.py
├── fast_decomposer.py
├── deep_decomposer.py
├── examples.py
├── LWE_estimator_vs_CoNAN.ipynb
└── README.md
```

