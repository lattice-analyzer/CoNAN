
# Primary_Decomposer little more optimized and support parallel.
# Need more optimization.

import re
import os
import ast
import time
import math
import multiprocessing as mp
from functools import lru_cache


K = QuadraticField(-1, 'i')          # Gaussian rationals Q(i)
_i = K.gen()                          



def extract_operator(filename, funcname, args):
    """
    Cost: O(n³).
    """
  
    if not os.path.isabs(filename) and not os.path.exists(filename):
        filename = os.path.join(os.getcwd(), filename)
    with open(filename, "r") as f:
        source = f.read()
    tree  = ast.parse(source)
    nodes = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name == funcname]
    if len(nodes) != 1:
        return None
    source_code = ast.unparse(nodes[0])

    lines    = source_code.strip().split('\n')
    newlines = [f'def {funcname}_modified(f, g, h, args):']
    for i in range(1, len(lines)):
        line = lines[i]
        x = re.fullmatch(
            r'([ \t]*)(h)([ \t]*)(\[.*?\])([ \t]*)(\+=|-=|=|:=)'
            r'([ \t]*-?[ \t]*((?:\d+|[a-zA-Z_]\w*)[ \t]*\*)?[ \t]*)'
            r'(f)([ \t]*)(\[.*?\])([ \t]*)(\*)([ \t]*)(g)([ \t]*)(\[.*?\])', line)
        d = None if x is None else {i: x.group(i) for i in range(1, 18)}
        newline = (line if x is None else
                   f'{d[1]}{d[2]}{d[3]}{d[11]}{d[17]}{d[4]}{d[5]}{d[6]}'
                   f'{d[7]}{d[9]}{d[10]}{d[11]}{d[12]}{d[13]}{d[14]}{d[15]}{d[16]}{d[17]}')
        newlines.append(newline)
    modified_code = '\n'.join(newlines)

    # Inject 'n' and any other args into the exec namespace so that
    # the generated function can reference them as global variables.
    # Without this, mul_NTRU_modified raises NameError: name 'n' is not defined
    # because the compiled code sees 'n' as a free variable.
    n = args[0]
    namespace = {'n': n, 'args': args}
    exec(compile(f'{source}\n\n{modified_code}', "<generated>", "exec"), namespace)
    mul_extracted = namespace[f'{funcname}_modified']
    f_in = [1] * n
    g_in = [1] * n
    P    = [[[0] * n for _ in range(n)] for _ in range(n)]
    mul_extracted(f_in, g_in, P, args)
    return P



def is_associative(P, n):
    """O(n⁴) — validation only, we don't call it"""
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for s in range(n):
                    if sum(P[i][j][r]*P[r][k][s] - P[j][k][r]*P[i][r][s]
                           for r in range(n)) != 0:
                        return False
    return True




def has_identity(P, n, tol=1e-10):
    """
    Find the identity element of the algebra defined by structure tensor P.

    """
   
    Ab = matrix(QQ, [[P[i//n][k][i % n] if k < n
                      else (1 if (i//n == i % n) else 0)
                      for k in range(n + 1)]
                     for i in range(n * n)])
    N = Ab.right_kernel().basis()
    if len(N) != 1:
        return None
    if abs(N[0][-1]) > tol:
        return matrix(QQ, [[-N[0][i] / N[0][-1] for i in range(n)]])
    return None



# extract_lattice  —  unchanged (O(n³), not a bottleneck)


def extract_lattice(P, n, mul_type='LL'):
    """Extract left/right multiplication matrices from structure tensor P."""
    if mul_type == 'LL':
        return [matrix(QQ, [[P[i][k][j] for j in range(n)]
                            for i in range(n)]) for k in range(n)]
    if mul_type == 'LR':
        return [matrix(QQ, [[P[j][k][i] for j in range(n)]
                            for i in range(n)]) for k in range(n)]
    if mul_type == 'RL':
        return [matrix(QQ, [[P[k][i][j] for j in range(n)]
                            for i in range(n)]) for k in range(n)]
    if mul_type == 'RR':
        return [matrix(QQ, [[P[k][j][i] for j in range(n)]
                            for i in range(n)]) for k in range(n)]



# scale_up  O(n) 



def scale_up(v):

    try:
        first = v[0]
        if first.parent() is ZZ or first.parent() is QQ:
            denoms = [x.denominator() for x in v]
            L = denoms[0]
            for d in denoms[1:]:
                L = lcm(L, d)
            v2 = L * v
            parts = [abs(x.numerator()) for x in v2 if x != 0]
            if not parts:
                return v2
            G = parts[0]
            for p in parts[1:]:
                G = gcd(G, p)
            return v2 / G
    except Exception:
        pass

    denoms = []
    for x in v:
        r, im = x.real(), x.imag()
        denoms.append(r.denominator())
        denoms.append(im.denominator())

    L = denoms[0]
    for d in denoms[1:]:
        L = lcm(L, d)
    v = L * v

    parts = []
    for x in v:
        rp = x.real().numerator()
        ip = x.imag().numerator()
        if rp != 0: parts.append(abs(rp))
        if ip != 0: parts.append(abs(ip))

    if not parts:
        return v
    G = parts[0]
    for p in parts[1:]:
        G = gcd(G, p)
    return v / G


def _ker_zz_to_K_basis(fAe_zz, n):

    return list(fAe_zz.right_kernel().basis())



def common_center_basis(A, k, n):
    M = Matrix(QQ, k * n * n, n * n)
    for t in range(k):
        At = A[t]
        base = t * n * n
        for i in range(n):
            for j in range(n):
                row = base + n * i + j
                for r in range(n):
                    M[row, n * r + j] += At[i, r]
                    M[row, n * i + r] -= At[r, j]
    B = []
    for b in M.right_kernel().basis():
        B.append(matrix(QQ, n, n, scale_up(b)))
    return B, len(B)




def PD(A, n):
    """
    Primary decomposition of a single matrix A.
    Uses Horner evaluation of f^e(A) instead of generic f(A)**e.
    """
    R = QQ['x']
    x = R.gen()
    cp  = A.charpoly(x)
    cpK = cp.change_ring(K)
    F   = cpK.factor()
    P   = []
    b   = []
    for f, e in F:
        # Horner evaluation of (f^e)(A)  — same speedup as in PD2
        fe_coeffs = list(reversed((f**e).list()))   # low→high degree
        d = len(fe_coeffs) - 1
        fAe = fe_coeffs[0] * identity_matrix(QQ, n)
        for t in range(1, d + 1):
            fAe = fAe * A + fe_coeffs[t] * identity_matrix(QQ, n)
        N = fAe.right_kernel().basis()
        P.extend([scale_up(v) for v in N])
        b.append(e * f.degree())
    return matrix(K, [[P[j][i] for j in range(n)] for i in range(n)]), b, cpK




def _sage_matrix_to_numpy(A_sage, n):
    """
    Fast conversion of a Sage QQ/ZZ matrix to a numpy float64 array.

    """
    import numpy as np
    flat = [float(x) for x in A_sage.list()]
    return np.array(flat, dtype=np.float64).reshape(n, n)


def _charpoly_via_eigvals(A_sage, n, field='QQ'):
    """
    Compute the characteristic polynomial and its factored form from
    numpy eigenvalues, without calling SymPy or Sage's Berkowitz algorithm.

    """
    import numpy as np
    from collections import Counter

    TOL = 0.3   # safe: Gaussian integers are distance ≥ 1 apart

    
    A_np = _sage_matrix_to_numpy(A_sage, n)

   
    eigs = np.linalg.eigvals(A_np)

    # ── Step 3: round to nearest Gaussian integer and validate ──────────
    rounded = []
    for e in eigs:
        re_r = round(e.real)
        im_r = round(e.imag)
        if abs(e.real - re_r) > TOL or abs(e.imag - im_r) > TOL:
            raise ValueError(
                f"Eigenvalue {e} is not close to a Gaussian integer "
                f"(residual {abs(e - complex(re_r, im_r)):.4f} > {TOL}). "
                "Falling back to exact charpoly."
            )
        rounded.append((int(re_r), int(im_r)))

    counts = Counter(rounded)   # { (re, im): multiplicity }

    Rz = ZZ['x']
    xz = Rz.gen()
    RK = K['x']
    xK = RK.gen()

    zz_product = Rz(1)
    consumed   = set()
    k_extra_factors = []   # (root_in_K, multiplicity) for unpaired roots

    for (re_r, im_r), mult in counts.items():
        if (re_r, im_r) in consumed:
            continue
        conj = (re_r, -im_r)

        if im_r == 0:
            
            zz_product *= (xz - ZZ(re_r)) ** mult
            consumed.add((re_r, 0))

        elif conj in counts:
            
            mult_pos = mult
            mult_neg = counts[conj]
            e_pair   = min(mult_pos, mult_neg)   # should be equal for QQ matrix

            quad = (xz - ZZ(re_r))**2 + ZZ(im_r)**2
            zz_product *= quad ** e_pair

            consumed.add((re_r,  im_r))
            consumed.add((re_r, -im_r))

            for (r2, i2), extra in [((re_r, im_r),  mult_pos - e_pair),
                                     ((re_r, -im_r), mult_neg - e_pair)]:
                if extra > 0:
                    k_extra_factors.append(
                        (K(r2) + K(i2) * _i, extra))
        else:
            
            k_extra_factors.append(
                (K(re_r) + K(im_r) * _i, mult))
            consumed.add((re_r, im_r))

    
    cp_K = RK(zz_product)
    for root, mult in k_extra_factors:
        cp_K *= (xK - root) ** mult

    

    if field == 'QQ[i]':
        
        factors = []
        for (re_r, im_r), mult in counts.items():
            root = K(re_r) + K(im_r) * _i
            factors.append((xK - root, mult))
        
        return cp_K, factors, cp_K

    else:  # field == 'QQ'
        # Factors live in QQ['x'].
        RQ  = QQ['x']
        xQ  = RQ.gen()
        factors = []
        consumed2 = set()

        for (re_r, im_r), mult in counts.items():
            if (re_r, im_r) in consumed2:
                continue
            conj = (re_r, -im_r)

            if im_r == 0:
                fQ = (xQ - QQ(re_r)) ** 1   
                factors.append((fQ, mult))
                consumed2.add((re_r, 0))

            elif conj in counts:
                e_pair = min(mult, counts[conj])
                # Irreducible quadratic (x-a)²+b² over QQ
                fQ = (xQ - QQ(re_r))**2 + QQ(im_r)**2
                factors.append((fQ, e_pair))
                consumed2.add((re_r,  im_r))
                consumed2.add((re_r, -im_r))
                
            else:
           
                consumed2.add((re_r, im_r))

        cp_QQ = RQ(zz_product)   # the QQ charpoly (real part only)
        return cp_QQ, factors, cp_K


def _horner_numpy(coeffs_lohi, A_np, n):
    """
    Evaluate polynomial  p(A)  in numpy float64 via Horner's method.
    """
    import numpy as np
    d   = len(coeffs_lohi) - 1
    I_f = np.eye(n, dtype=np.float64)
    res = float(coeffs_lohi[d]) * I_f
    for t in range(d - 1, -1, -1):
        res = res @ A_np
        c = float(coeffs_lohi[t])
        if c != 0.0:
            res += c * I_f
    return res


def _numpy_to_sage_int_matrix(arr, n):
    """
    Convert a numpy float64 matrix (whose true values are integers) to a
    Sage matrix over ZZ.  
    """
    flat = [int(round(arr[i, j])) for i in range(n) for j in range(n)]
    return matrix(ZZ, n, n, flat)


def PD2(B, i, n, mul_type, unity, field='QQ'):
    """
    Primary decomposition using commutant element B[i].

    """
    import numpy as np

    A = B[i]
    M = A if mul_type[1] == 'L' else A.T

    # ── Step 1: charpoly + factor list ─────────────────────────────────
    try:
        cp_base, factors, cp_K = _charpoly_via_eigvals(A, n, field=field)
    except Exception:
        # Exact fallback: Sage Berkowitz + factor in the chosen ring
        R  = QQ['x']
        x  = R.gen()
        cp = A.charpoly(x)
        if field == 'QQ[i]':
            cp_K    = cp.change_ring(K)
            cp_base = cp_K
            factors = list(cp_K.factor())
        else:
            cp_base = cp
            cp_K    = cp.change_ring(K)
            factors = list(cp.factor())

    # Irreducibility check 
   
    if len(factors) == 1:
        return identity_matrix(QQ, n), [n], cp_K

    
    M_np = _sage_matrix_to_numpy(M, n)

    b = []
    P = []

    for f, e in factors:
        fe_poly = f ** e

        
        base_ring = f.parent().base_ring()
        coeffs_raw = list(fe_poly.list())   

        use_numpy = False
        if base_ring in (ZZ, QQ):
            coeffs_QQ = [QQ(c) for c in coeffs_raw]
            if all(c.denominator() == 1 for c in coeffs_QQ):
                use_numpy    = True
                coeffs_float = [float(c) for c in coeffs_QQ]

        if use_numpy:
            
            fAe_np  = _horner_numpy(coeffs_float, M_np, n)
            fAe_zz  = _numpy_to_sage_int_matrix(fAe_np, n)
            ker_basis = fAe_zz.right_kernel().basis()

        elif base_ring is K or base_ring == K:
            
            coeffs_K = [K(c) for c in coeffs_raw]
            M_K      = M.change_ring(K)
            fAe      = coeffs_K[-1] * identity_matrix(K, n)
            for t in range(len(coeffs_K) - 2, -1, -1):
                fAe = fAe * M_K
                if coeffs_K[t] != K(0):
                    fAe += coeffs_K[t] * identity_matrix(K, n)
            ker_basis = fAe.right_kernel().basis()

        else:
           
            coeffs_QQ2 = [QQ(c) for c in coeffs_raw]
            fAe = coeffs_QQ2[-1] * identity_matrix(QQ, n)
            for t in range(len(coeffs_QQ2) - 2, -1, -1):
                fAe = fAe * M
                if coeffs_QQ2[t] != 0:
                    fAe += coeffs_QQ2[t] * identity_matrix(QQ, n)
            ker_basis = fAe.right_kernel().basis()

        P.extend([scale_up(v) for v in ker_basis])
        b.append(e * f.degree())

    if not P:
        return identity_matrix(QQ, n), [n], cp_K

   
    flat = [P[j][ii] for ii in range(n) for j in range(n)]
    return matrix(K, n, n, flat), b, cp_K




def combine(P, n, b, s, k, i, p, m, PP):
    """
    Enumerate all partitions of primary blocks and collect non-trivial ones.

    """
    if i == k:
        if m == 1:
            return
        D = [0] * (s + 1)
        for j in range(s):
            D[j + 1] = D[j] + b[j]

        ss = [0] * m
        V_cols = [[] for _ in range(m)]
        for j in range(s):
            ss[p[j]] += b[j]
            V_cols[p[j]].extend(P[D[j]:D[j + 1]])

        DD = [0] * (m + 1)
        for j in range(m):
            DD[j + 1] = DD[j] + ss[j]

        
        M_diag = Matrix(K, n, n)
        for j in range(m):
            for jj in range(DD[j], DD[j + 1]):
                M_diag[jj, jj] = j

       
        V_flat  = [v for VV in V_cols for v in VV]
        V_mat   = Matrix(K, [[v[idx] for v in V_flat] for idx in range(n)])
        M_full  = V_mat * M_diag * V_mat.inverse()

        
        V_split = [(M_full - j * identity_matrix(K, n)).right_kernel()
                   for j in range(m)]
        V_all   = [v for VV in V_split for v in VV]

        
        V_final = Matrix(K, [[P[j][idx] for j in range(n)] for idx in range(n)])
        PP.append([V_final, ss])

    else:
        m1 = min(k, m + 1)
        for x in range(m1):
            p[i] = x
            combine(P, n, b, s, k, i + 1, p, max(m, x + 1), PP)




def gaussian_heuristic(lattice_dim, log_vol):
    """
    Gaussian heuristic for the expected shortest vector length in a lattice


    """
    import math
    d = float(lattice_dim)
    return math.sqrt(d / (2.0 * math.pi * math.e)) * math.exp(log_vol / d)



def estimate_beta(dimension, variance, q, n, lattice='NTRU'):
    import math

    fold        = n / dimension
    folded_var  = float(variance) * fold

    lattice_dim    = 2 * dimension
    log_vol        = dimension * math.log(float(q))
    short_vec_norm = math.sqrt(folded_var * float(lattice_dim))
    gh             = gaussian_heuristic(lattice_dim, log_vol)

    if short_vec_norm >= gh:
        return None

    try:
        from LWE_estimator import estimator
        sigma = math.sqrt(folded_var)
        Xs    = estimator.ND.DiscreteGaussian(sigma)
        Xe    = estimator.ND.DiscreteGaussian(sigma)
        if lattice == 'NTRU':
            params = estimator.NTRU.Parameters(n=dimension, q=q, Xs=Xs, Xe=Xe)
            return float(estimator.NTRU.primal_usvp(params)["beta"])
        if lattice == 'LWE':
            params = estimator.LWE.Parameters(n=dimension, q=q, Xs=Xs, Xe=Xe)
            return float(estimator.LWE.primal_usvp(params)["beta"])
    except Exception as exc:
        return None


def block_betas(P_mat, b_list, L, n, q, base_var=2/3, lattice='NTRU'):
    import math
    betas  = []
    fold_var = []
    offset = 0
    for dim in b_list:
        fold           = n / dim
        folded_var     = base_var * fold
        lattice_dim    = 2 * dim
        log_vol        = dim * math.log(float(q))
        short_vec_norm = math.sqrt(folded_var * float(lattice_dim))
        gh             = gaussian_heuristic(lattice_dim, log_vol)

        if short_vec_norm < gh:
            beta = estimate_beta(dim, base_var, q, n, lattice)
        else:
            beta = None

        betas.append(beta)
        fold_var.append(folded_var)
        offset += dim
    return betas, fold_var

def _worker_PD2(args):
   
    
    i, n, mul_type_str, unity_list, B_list_flat, n_matrices, field = args

   
    unity = matrix(QQ, 1, n, unity_list)
    B     = [matrix(QQ, n, n, B_list_flat[t]) for t in range(n_matrices)]

    try:
        P, b, f = PD2(B, i, n, mul_type_str, unity, field=field)
        key     = tuple(f.coefficients(sparse=False))
        return {
            'idx':     i,
            'P':       P,
            'b':       b,
            'key':     key,
            'skipped': False,
            'reason':  None,
        }
    except Exception as exc:
        return {
            'idx':     i,
            'P':       None,
            'b':       None,
            'key':     None,
            'skipped': True,
            'reason':  str(exc),
        }


# homomorphism  —  parallel version



def homomorphism(filename, funcname, args, mul_type, do_print,
                 num_workers=1, q=None, base_var=2/3,
                 lattice='NTRU', task_timeout=300, field='QQ', level=2):
    """
    Compute all non-trivial homomorphisms via primary decomposition.
    """
    n  = args[0]
    mt = {'L', 'R'}
    if level not in (2, 3):
        print('invalid level'); return None
    if (not isinstance(mul_type, str) or len(mul_type) != 2
            or mul_type[0] not in mt or mul_type[1] not in mt):
        print('invalid multiplication type'); return None

    # Setup (runs in main process) 
    M     = extract_operator(filename, funcname, args)
    unity = has_identity(M, n)
    L     = extract_lattice(M, n, mul_type)
    opp   = mul_type[0] + ('R' if mul_type[1] == 'L' else 'L')
    C     = extract_lattice(M, n, opp)

    t   = n
    pol = set()
    cnt = 0
    hom = []

   
    unity_list = list(unity[0]) if unity is not None else [0] * n

    # Pre-deduplicate by charpoly BEFORE spawning workers.
 
    print(f"[CoNAN] Pre-filtering {t} elements by charpoly ...")
    import numpy as np

    # Serialise ALL n matrices once here, shared across task tuples.
    B_flat_all = [list(C[jj].list()) for jj in range(n)]

    unique_indices = []
    seen_keys      = set()
    for idx in range(t):
        try:
            A_np    = _sage_matrix_to_numpy(C[idx], n)
            eigs    = np.linalg.eigvals(A_np)
            rounded = tuple(sorted(
                (int(round(e.real)), int(round(e.imag))) for e in eigs))
            key = rounded
        except Exception:
            key = idx   # fallback: treat as unique
        if key not in seen_keys:
            seen_keys.add(key)
            unique_indices.append(idx)

    print(f"[CoNAN] {len(unique_indices)} unique charpolys from {t} elements")


    task_args = []
    for idx in unique_indices:
        task_args.append((idx, n, mul_type, unity_list, B_flat_all, n, field))

    nw = min(max(1, num_workers), len(task_args))

    print(f"[CoNAN] n={n}  workers={nw}  tasks={len(task_args)}  "
          f"level={level}  lattice={lattice}  q={q}  field={field}")

   
    if nw == 1:
        results = []
        for ta in task_args:
            r = _worker_PD2(ta)
            results.append(r)
    else:
      
        results = []
        with mp.Pool(processes=nw) as pool:
            for r in pool.imap_unordered(_worker_PD2, task_args,
                                          chunksize=1):
                results.append(r)
            pool.terminate()

    # ── Aggregate results 
    for r in results:
        if r is None or r['skipped']:
            if r and r['reason']:
                continue
                #some are skipped for some reason#print(f"  C{r['idx']} => Skipped ({r['reason']})")
            

        i   = r['idx']
        P   = r['P']
        b   = r['b']
        key = r['key']

        if len(b) == 1:
            print(f'C{i} => Only 1 block')
            continue
        if key in pol:
            print(f'C{i} => Repeated characteristic polynomial')
            continue
        pol.add(key)
        print(f'C{i} => New characteristic polynomial  blocks={len(b)}  dims={b}')

        PP = []
        if level == 2:
            PP = [[P, b]]
        else:
            k = min(6, len(b))
            combine([P.column(j) for j in range(n)],
                    n, b, len(b), k, 1, [0]*k, 1, PP)

        for X in PP:
            P_hom = X[0]
            b_hom = X[1]

            # 
            if q is not None:
                betas, fold_var = block_betas(P_hom, b_hom, L, n, q,
                                    base_var, lattice)
                print("folded_var", fold_var)
                print(f"  Partition dims={b_hom}  "
                      f"betas={[f'{bb:.1f}' if bb else 'N/A' for bb in betas]}"
                      )
                
            else:
                betas = [None] * len(b_hom)

            hom.append({'matrix': P_hom, 'dims': b_hom, 'betas': betas})
            cnt += 1

            if do_print:
                print(f'\nHomomorphism {cnt}  dims={b_hom}')
                print(P_hom)
                P_inv  = P_hom.inverse()
                L_new  = [P_inv * LL * P_hom for LL in L]
                # print_matrix_symbolic(L_new, n, n, ...) — user-defined, kept as-is

    print(f'\n[CoNAN] Done. {cnt} homomorphism(s) found.')
    return hom



