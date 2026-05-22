import re
import os
import ast
import time
import multiprocessing as mp


#### This file is non optimized and can run up to n = 128
#### No need to get beta estimation for small dimensions

K = QuadraticField(-1, 'i')


def extract_operator(filename, funcname, args):
    
    if not os.path.isabs(filename) and not os.path.exists(filename):
        filename = os.path.join(os.getcwd(), filename)
    with open(filename, "r") as f:
        source = f.read()
    tree = ast.parse(source)
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == funcname]
    if len(nodes) != 1:
        return None
    source_code = ast.unparse(nodes[0])

    lines = source_code.strip().split('\n')
    newlines = [f'def {funcname}_modified(f, g, h, args):']
    for i in range(1, len(lines)):
        line = lines[i]
        x = re.fullmatch(r'([ \t]*)(h)([ \t]*)(\[.*?\])([ \t]*)(\+=|-=|=|:=)([ \t]*-?[ \t]*((?:\d+|[a-zA-Z_]\w*)[ \t]*\*)?[ \t]*)(f)([ \t]*)(\[.*?\])([ \t]*)(\*)([ \t]*)(g)([ \t]*)(\[.*?\])', line)
        d = None if x is None else {i: x.group(i) for i in range(1, 18)}
        newline = line if x is None else f'{d[1]}{d[2]}{d[3]}{d[11]}{d[17]}{d[4]}{d[5]}{d[6]}{d[7]}{d[9]}{d[10]}{d[11]}{d[12]}{d[13]}{d[14]}{d[15]}{d[16]}{d[17]}'
        newlines.append(newline)
    modified_code = '\n'.join(newlines)

    # Inject all args into namespace so variables like n, q etc are visible
    n = args[0]
    namespace = {'n': n, 'args': args}
    for idx, val in enumerate(args):
        namespace[f'args_{idx}'] = val
    exec(compile(f'{source}\n\n{modified_code}', "<generated>", "exec"), namespace)
    mul_extracted = namespace[f'{funcname}_modified']
    f = [1] * n
    g = [1] * n
    P = [[[0 for k in range(n)] for j in range(n)] for i in range(n)]
    mul_extracted(f, g, P, args)
    return P


def is_associative(P, n):
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for s in range(n):
                    if sum(P[i][j][r]*P[r][k][s]-P[j][k][r]*P[i][r][s] for r in range(n)) != 0:
                        return False
    return True


def has_identity(P, n, tol=1e-10):
    Ab = matrix(QQ, [[P[i//n][k][i%n] if k < n else 1 if (i//n == i%n) else 0 for k in range(n+1)] for i in range(n*n)])
    N = Ab.right_kernel().basis()
    if len(N) != 1:
        return None
    if abs(N[0][-1]) > tol:
        return matrix(QQ, [[-N[0][i]/N[0][-1] for i in range(n)]])
    return None


def extract_lattice(P, n, mul_type='LL'):
    if mul_type == 'LL':
        return [matrix(QQ, [[P[i][k][j] for j in range(n)] for i in range(n)]) for k in range(n)]
    if mul_type == 'LR':
        return [matrix(QQ, [[P[j][k][i] for j in range(n)] for i in range(n)]) for k in range(n)]
    if mul_type == 'RL':
        return [matrix(QQ, [[P[k][i][j] for j in range(n)] for i in range(n)]) for k in range(n)]
    if mul_type == 'RR':
        return [matrix(QQ, [[P[k][j][i] for j in range(n)] for i in range(n)]) for k in range(n)]
    return None


def print_matrix_basis(L, t, n, fname, s1, s2):
    print(f'Basis Matrices for scheme \'{fname}\', Lattice {s1}({s2}) = {s2}0*{s1}0 + ... + {s2}{t-1}*{s1}{t-1}, where')
    X = [str(L[i]).split('\n') for i in range(n)]
    A = [f'{s1}{i} = ' for i in range(t)]
    B = [''.ljust(len(A[i])) for i in range(t)]
    for i in range(n):
        print(f'{"  " if i < n-1 else ", "}'.join([f'{A[j] if i == 0 else B[j]}{X[i][j]}' for j in range(t)]))


def print_matrix_symbolic(L, t, n, fname, s1, s2):
    h = var([f"{s2}{i}" for i in range(t)])
    H = [[str(expand(sum(L[k][i,j]*h[k] for k in range(t)))) for j in range(n)] for i in range(n)]
    m = max(len(H[i//n][i%n]) for i in range(n*n))
    for i in range(n):
        print(f'[{" ".join([H[i][j].rjust(m) for j in range(n)])}]')


def scale_up(v):
    L = lcm([lcm(x.real().denominator(), x.imag().denominator()) for x in v])
    v = L*v
    G = gcd([x for x in v])
    v = v/G
    return v


def common_center_basis(A, k, n):
    M = Matrix(QQ, k*n*n, n*n)
    for t in range(k):
        for i in range(n):
            for j in range(n):
                for r in range(n):
                    M[t*n*n + n*i + j, n*r + j] += A[t][i, r]
                    M[t*n*n + n*i + j, n*i + r] -= A[t][r, j]
    B = []
    for b in M.right_kernel().basis():
        B.append(matrix(QQ, n, n, scale_up(b)))
    return B, len(B)


def PD(A, n):
    R.<x> = QQ[]
    cp = A.charpoly(x)
    cpK = cp.change_ring(K)
    F = cpK.factor()
    P = []
    b = []
    for f,e in F:
        N = (f(A) ** e).right_kernel().basis()
        P.extend([scale_up(v) for v in N])
        b.append(e*f.degree())
    return matrix(K, [[P[j][i] for j in range(n)] for i in range(n)]), b, cpK


def PD2(B, i, n, mul_type, unity):
    A = B[i]
    M = A if mul_type[1] == 'L' else A.T
    R.<x> = QQ[]
    cp = A.charpoly(x)
    cpK = cp.change_ring(K)
    F = cpK.factor()
    if len(F) == 1:
        return identity_matrix(QQ, n), [n], cpK
    b = []
    P = []
    for f,e in F:
        c = (f**e).list()
        d = len(c) - 1
        y = zero_matrix(QQ, 1, n)
        for t in range(d,-1,-1):
            y = y*M + c[t]*unity
        fAe = zero_matrix(QQ, n, n)
        for j in range(n):
            fAe += y[0,j]*B[j]
        P.extend([scale_up(v) for v in fAe.right_kernel().basis()])
        b.append(d)
    return matrix(K, [[P[j][i] for j in range(n)] for i in range(n)]), b, cpK


def combine(P, n, b, s, k, i, p, m, PP):
    if i == k:
        if m == 1:
            return
        D = [0] * (s+1)
        for j in range(s):
            D[j+1] = D[j] + b[j]
        ss = [0] * m
        V = [[] for j in range(m)]
        for j in range(s):
            ss[p[j]] += b[j]
            V[p[j]].extend(P[D[j]:D[j+1]])
        DD = [0] * (m+1)
        for j in range(m):
            DD[j+1] = DD[j] + ss[j]
        M = Matrix(K, n, n)
        for j in range(m):
            for jj in range(DD[j],DD[j+1]):
                M[jj,jj] = j
        V = [v for VV in V for v in VV]
        V = Matrix(K, [[P[j][i] for j in range(n)] for i in range(n)])
        M = V*M*V.inverse()
        V = [(M - j*identity_matrix(n)).right_kernel() for j in range(m)]
        V = [v for VV in V for v in VV]
        PP.append([Matrix(K, [[P[j][i] for j in range(n)] for i in range(n)]), ss])
    else:
        m1 = min(k,m+1)
        for x in range(m1):
            p[i] = x
            combine(P, n, b, s, k, i+1, p, max(m,x+1), PP)


def homomorphism(filename, funcname, args, mul_type, do_print, level=2):
    n = args[0]
    mt = set(['L', 'R'])
    if level != 2 and level != 3:
        print('invalid level')
        return None
    if type(mul_type) != str or len(mul_type) != 2 or mul_type[0] not in mt or mul_type[1] not in mt:
        print('invalid multiplication type')
        return None
    M = extract_operator(filename, funcname, args)
    unity = has_identity(M, n)
    L = extract_lattice(M, n, mul_type)
    C = extract_lattice(M, n, f'{mul_type[0]}{"R" if mul_type[1] == "L" else "L"}')
    t = n
    pol = set()
    cnt = 0
    hom = []
    for i in range(t):
        P, b, f = PD2(C, i, n, mul_type, unity)
        key = tuple(f.coefficients(sparse=False))
        if len(b) == 1:
            print(f'C{i} => Only 1 block')
            continue
        if key in pol:
            print(f'C{i} => Repeated characteristic Polynomial')
            continue
        pol.add(key)
        print(f'C{i} => New characteristic Polynomial')
        PP = []
        if level == 2:
            PP = [[P, b]]
        else:
            k = min(6, len(b))
            combine([P.column(i) for i in range(n)], n, b, len(b), k, 1, [0]*k, 1, PP)
        for X in PP:
            hom.append(X)
            if do_print:
                P = X[0]
                b = X[1]
                cnt += 1
                print(f'\nHomomorphism - {cnt} found')
                print(b)
                print(P)
                P_inv = P.inverse()
                L_new = [P_inv * LL * P for LL in L]
                print_matrix_symbolic(L_new, n, n, f'{funcname}_hom{cnt}', "L'", 'h')
    print('\n\nAll done!!!')
    return hom




def estimate_beta(dim, q, base_var=2/3, lattice='NTRU'):

    try:
        from LWE_estimator import estimator
        import math
        sigma  = math.sqrt(float(base_var))
        Xs     = estimator.ND.DiscreteGaussian(sigma)
        Xe     = estimator.ND.DiscreteGaussian(sigma)
        if lattice == 'NTRU':
            params = estimator.NTRU.Parameters(n=dim, q=q, Xs=Xs, Xe=Xe)
            return float(estimator.NTRU.primal_usvp(params)["beta"])
        if lattice == 'LWE':
            params = estimator.LWE.Parameters(n=dim, q=q, Xs=Xs, Xe=Xe)
            return float(estimator.LWE.primal_usvp(params)["beta"])
    except Exception as e:
        return None



def _pd2_worker(task):

    i, n, mul_type, unity_rows, C_rows = task

    
    K_worker  = QuadraticField(-1, 'i')
    unity     = matrix(QQ, unity_rows)
    C         = [matrix(QQ, rows) for rows in C_rows]

    try:
        P, b, f = PD2(C, i, n, mul_type, unity)

        
        P_rows = [[(str(P[r, c].real()), str(P[r, c].imag()))
                   for c in range(n)] for r in range(n)]
        key    = tuple(f.coefficients(sparse=False))

        return {
            'i':      i,
            'P_rows': P_rows,
            'b':      b,
            'key':    key,
            'error':  None,
        }

    except Exception as e:
        return {
            'i':      i,
            'P_rows': None,
            'b':      None,
            'key':    None,
            'error':  str(e),
        }



def homomorphism_parallel(filename, funcname, args, mul_type, do_print,
                          num_workers=1, q=None, base_var=2/3,
                          lattice='NTRU',level=2, timeout=300):
  

    n  = args[0]
    mt = set(['L', 'R'])

    if level != 2 and level != 3:
        print('invalid level')
        return None
    if (type(mul_type) != str or len(mul_type) != 2
            or mul_type[0] not in mt or mul_type[1] not in mt):
        print('invalid multiplication type')
        return None

    
    M     = extract_operator(filename, funcname, args)
    unity = has_identity(M, n)
    L     = extract_lattice(M, n, mul_type)
    opp   = mul_type[0] + ('R' if mul_type[1] == 'L' else 'L')
    C     = extract_lattice(M, n, opp)
    t     = n

    print(f'[Parallel] n={n}  workers={num_workers}  level={level}  '
          f'lattice={lattice}  q={q}')

   
    unity_rows = [[unity[0, j] for j in range(n)]]          # 1×n row
    C_rows     = [[[C[k][r, c] for c in range(n)]
                   for r in range(n)] for k in range(n)]    # n×n×n

 
    tasks = [(i, n, mul_type, unity_rows, C_rows) for i in range(t)]

   
    if num_workers == 1:
        raw_results = [_pd2_worker(task) for task in tasks]
    else:
        raw_results = [None] * t
        with mp.Pool(processes=num_workers) as pool:
            handles = [(i, pool.apply_async(_pd2_worker, (tasks[i],)))
                       for i in range(t)]
            for i, handle in handles:
                try:
                    raw_results[i] = handle.get(timeout=timeout)
                except mp.TimeoutError:
                    print(f'C{i} => TIMED OUT after {timeout}s — skipped')
                    raw_results[i] = {'i': i, 'P_rows': None, 'b': None,
                                      'key': None, 'error': 'timeout'}
                except Exception as e:
                    print(f'C{i} => ERROR: {e}')
                    raw_results[i] = {'i': i, 'P_rows': None, 'b': None,
                                      'key': None, 'error': str(e)}
            pool.terminate()

    
    pol = set()
    cnt = 0
    hom = []

    for res in raw_results:
        i = res['i']

        if res['error']:
            print(f'C{i} => Error: {res["error"]}')
            continue

        b   = res['b']
        key = res['key']

        if len(b) == 1:
            print(f'C{i} => Only 1 block')
            continue
        if key in pol:
            print(f'C{i} => Repeated characteristic Polynomial')
            continue
        pol.add(key)
        print(f'C{i} => New characteristic Polynomial')

       
        K_local = QuadraticField(-1, 'i')
        _i      = K_local.gen()
        P = matrix(K_local, n, n,
                   [QQ(re) + QQ(im) * _i
                    for r in res['P_rows'] for (re, im) in r])

        
        PP = []
        if level == 2:
            PP = [[P, b]]
        else:
            k = min(6, len(b))
            combine([P.column(j) for j in range(n)],
                    n, b, len(b), k, 1, [0]*k, 1, PP)

        for X in PP:
            hom.append(X)

           
            #if q is not None:
            #    betas = [estimate_beta(dim, q, base_var, lattice)
            #             for dim in X[1]]
            #    beta_str = [f'{bb:.1f}' if bb is not None else 'N/A'
            #                for bb in betas]
            #    print(f'  dims={X[1]}  betas={beta_str}')

            if do_print:
                P_out = X[0]
                b_out = X[1]
                cnt  += 1
                print(f'\nHomomorphism - {cnt} found')
                print(b_out)
                print(P_out)
                P_inv = P_out.inverse()
                L_new = [P_inv * LL * P_out for LL in L]
                print_matrix_symbolic(L_new, n, n,
                                      f'{funcname}_hom{cnt}', "L'", 'h')

    print('\n\nAll done!!!')
    return hom



#   HOW TO RUN 
#   1) no beta estimation
#   hom = homomorphism('examples.py', 'mul_NTRU', [128], 'LL', False,2)
#
#   2) — parallel with beta estimation (for this version, it is little unnecessary):
#
#   hom = homomorphism_parallel(
#       filename    = 'examples.py',
#       funcname    = 'mul_NTRU',
#       args        = [128],       # add more if mul_NTRU needs more parameters for #  multiplication 	
#       mul_type    = 'LL',
#       do_print    = False,
#       num_workers = 8,
#       q           = 512,
#       base_var    = 2/3,
#       lattice     = 'NTRU',
#       level       = 2,
#       timeout     = 300,
#   )

