################### This file contains multiplication functions associated with various algebras. #####################

import numpy as np
import math
import sympy



# NTRU - Z[X]/<X^N - 1>
def mul_NTRU(f, g, h, n):
    for i in range(n):
        for j in range(n):
            h[(i+j)%n] += f[i]*g[j]
            
            

# NTRU Prime - Z[X]/<X^N - X - 1>
def mul_NTRUPrime(f, g, h, n):
    for i in range(n):
        for j in range(n):
            h[(i+j)%n] += f[i]*g[j]
            if i+j>=n:
                h[(i+j+1)%n] += f[i]*g[j]


# Falcon - Z[X]/<X^N + 1>
def mul_Falcon(f, g, h, n):
    for i in range(n):
        for j in range(n):
            if i+j<n:
                h[(i+j)%n] += f[i]*g[j]
            else:
                h[(i+j)%n] += -f[i]*g[j]


# NTTRU/NTRU+ - Z[X]/<X^N - X^(N/2) + 1>
def mul_NTRUPlus(f, g, h, n):
  # k = i+j
  # k < n => x^k
  # k < 3n/2 => x^(k-n/2) - x^(k-n)
  # else => -x^(k-3n/2)
    for i in range(n):
        for j in range(n):
            if i+j < n:
                h[i+j] += f[i]*g[j]
            elif i+j < 3*n//2:
                h[i+j - n//2] += f[i]*g[j]
                h[i+j - n] += (-1)*f[i]*g[j]
            else:
                h[i+j - 3*n//2] += (-1)*f[i]*g[j]


                

# ETRU - Z[omega][X]/<X^N - 1> => a + b*omega, (a+bw)(c+dw) = (ac-bd) + w(ad+bc-bd)
def mul_ETRU(f, g, h, n):
    N = n // 2
    for i in range(N):
        for j in range(N):
            h[((i+j)%N)*2  ] += f[2*i]*g[2*j]
            h[((i+j)%N)*2  ] += -f[2*i+1]*g[2*j+1]
            h[((i+j)%N)*2+1] += f[2*i]*g[2*j+1]
            h[((i+j)%N)*2+1] += f[2*i+1]*g[2*j]
            h[((i+j)%N)*2+1] += -f[2*i+1]*g[2*j+1]
            
            
            
# DiTRU - Z[X,Y]/<X^N - 1, Y^2 - 1, XY - YX^(N-1)>
def mul_DiTRU(f, g, h, n):
    N = n // 2
    for i in range(N):
        for j in range(N):
            h[(i+j)%N    ] += f[i  ]*g[j  ]   # x^i  * x^j
            h[N+(i+j)%N  ] += f[N+i]*g[j  ]   # yx^i * x^j
            h[N+(N-i+j)%N] += f[i  ]*g[N+j]   # x^i  * yx^j
            h[(N-i+j)%N  ] += f[N+i]*g[N+j]   # yx^i * yx^j
            
            
# GRLWE from dihedral group              
def mul_LWE_DCC(f, g, h, n):
    N = n // 2
    for i in range(N):
        for j in range(N):
            if(i+j<N):
                h[(i+j)%N    ] += f[i  ]*g[j  ]   # x^i  * x^j
                h[N+(i+j)%N  ] += f[N+i]*g[j  ]   # yx^i * x^j
                h[N+(N-i+j)%N] += f[i  ]*g[N+j]   # x^i  * yx^j
                h[(N-i+j)%N  ] += f[N+i]*g[N+j]   # yx^i * yx^j
            else:
                h[(i+j)%N    ] += -f[i  ]*g[j  ]   # x^i  * x^j
                h[N+(i+j)%N  ] += -f[N+i]*g[j  ]   # yx^i * x^j
                h[N+(N-i+j)%N] += -f[i  ]*g[N+j]   # x^i  * yx^j
                h[(N-i+j)%N  ] += -f[N+i]*g[N+j]   # yx^i * yx^j
                


# DiTRU+ - Z[X,Y]/<X^N - 1, Y^2 + 1, XY - YX^(N-1)>
def mul_DiTRUPlus(f, g, h, n):
    N = n // 2
    for i in range(N):
        for j in range(N):
            h[(i+j)%N    ] +=  f[i  ]*g[j  ]   # x^i  * x^j
            h[N+(i+j)%N  ] +=  f[N+i]*g[j  ]   # yx^i * x^j
            h[N+(N-i+j)%N] +=  f[i  ]*g[N+j]   # x^i  * yx^j
            h[(N-i+j)%N  ] += -f[N+i]*g[N+j]   # yx^i * yx^j
            


# NTRU3 - Z[X,Y]/<X^N - 1, Y^3 - 1, XY - YX^s>, s^3 = 1 (mod N)
def mul_NTRU3(f, g, h, n):
    N = n // 3
    s = [a for a in range(1, N) if pow(a, 3, N) == 1]
    for i in range(N):
        for j in range(N):
            h[(i*s[0]+j)%N    ] += f[i    ]*g[j    ]
            h[(i*s[0]+j)%N+N  ] += f[i+N  ]*g[j    ]
            h[(i*s[0]+j)%N+N+N] += f[i+N+N]*g[j    ]
            h[(i*s[1]+j)%N+N  ] += f[i    ]*g[j+N  ]
            h[(i*s[1]+j)%N+N+N] += f[i+N  ]*g[j+N  ]
            h[(i*s[1]+j)%N    ] += f[i+N+N]*g[j+N  ]
            h[(i*s[2]+j)%N+N+N] += f[i    ]*g[j+N+N]
            h[(i*s[2]+j)%N    ] += f[i+N  ]*g[j+N+N]
            h[(i*s[2]+j)%N+N  ] += f[i+N+N]*g[j+N+N]
            

# MNTRU - Z[X,Y]/<X^N - 1, Y^M - 1, XY - YX^s>, s^M = 1 (mod N)
def mul_MNTRU(f, g, h, n):
    t = sympy.factorint(n)
    N = max(t.keys())
    M = min(t.keys())
    x = min([a for a in range(2, N) if pow(a, M, N) == 1])
    s = [pow(x, i, N) for i in range(M)]
    for i1 in range(M):
        for j1 in range(M):
            for i0 in range(N):
                for j0 in range(N):
                    h[((j1+i1)%M)*N + (s[j1]*i0+j0)%N] += f[i1*N+i0]*g[j1*N+j0]
                    


                    

# QTRU  - {f0 + f1i + f2j + f3k | fi in Z[X]/<X^N - 1>, i^2 = j^2 = -1, ij = -ji = k}
# 1  i  j  k
# i -1  k -j
# j -k -1  i
# k  j -i -1
def mul_QTRU(f, g, h, n):
    N = n // 4
    for i1 in range(4):
        for j1 in range(4):
            for i0 in range(N):
                for j0 in range(N):
                    if (i1 == 0 or j1 == 0 or (j1-i1)%3 == 1):
                        h[(i1^j1)*N+(i0+j0)%N] += f[i1*N+i0]*g[j1*N+j0]
                    else:
                        h[(i1^j1)*N+(i0+j0)%N] += -f[i1*N+i0]*g[j1*N+j0]
                        


# SQTRU - {f0 + f1i + f2j + f3k | fi in Z[X]/<X^N - 1>, -i^2 = j^2 = 1, ij = -ji = k}
# 1  i  j  k
# i -1  k -j
# j -k  1 -i
# k  j  i  1
def mul_SQTRU(f, g, h, n):
    N = n // 4
    for i1 in range(4):
        for j1 in range(4):
            for i0 in range(N):
                for j0 in range(N):
                    if j1%2 == 1 and i1%3 != 0:
                        h[(i1^j1)*N+(i0+j0)%N] += (-1)*f[i1*N+i0]*g[j1*N+j0]
                    else:
                        h[(i1^j1)*N+(i0+j0)%N] += f[i1*N+i0]*g[j1*N+j0]
                        


# BQTRU - {f0 + f1i + f2j + f3k | fi in Z[X,Y]/<X^N - 1, Y^N - 1, XY - YX>, i^2 = j^2 = 1,  ij = -ji = k}
# 1  i  j  k
# i  1  k  j
# j -k  1 -i
# k -j  i -1
def mul_BQTRU(f, g, h, n):
    N = int(np.sqrt(n // 4))
    for i2 in range(4):
        for j2 in range(4):
            for i1 in range(N):
                for j1 in range(N):
                    for i0 in range(N):
                        for j0 in range(N):
                            if i2//2 == 1 and j2%2 == 1:
                                h[(i2^j2)*N*N+((i1+j1)%N)*N+(i0+j0)%N] += (-1)*f[i2*N*N+i1*N+i0]*g[j2*N*N+j1*N+j0]
                            else:
                                h[(i2^j2)*N*N+((i1+j1)%N)*N+(i0+j0)%N] += f[i2*N*N+i1*N+i0]*g[j2*N*N+j1*N+j0]
                                

                                
## General quaternion Given in CiC (modify it)                               
def mul_DiTRUplus_strong(f, g, h, n):
    N = n // 2
    for i in range(N):
        for j in range(N):
            if i + j < N:
                h[i+j        ] +=  f[i  ]*g[j  ]   # x^i  * x^j
                h[N+i+j      ] +=  f[N+i]*g[j  ]   # yx^i * x^j
            else:
                h[i+j-N      ] += -f[i  ]*g[j  ]   # x^i  * x^j
                h[i+j        ] += -f[N+i]*g[j  ]   # yx^i * x^j
            if j < i:
                h[N+N-i+j    ] += -f[i  ]*g[N+j]   # x^i  * yx^j
                h[(N-i+j+1)%N] += -f[N+i]*g[N+j]   # yx^i * yx^j
            else:
                h[N-i+j      ] +=  f[i  ]*g[N+j]   # x^i  * yx^j
                h[(j-i+1)%N  ] +=  f[N+i]*g[N+j]   # yx^i * yx^j
                

                
# MNTRU - Z[X,Y]/<X^N - 1, Y^M - 1, XY - YX^s>, s^M = 1 (mod N)
## LWE from semi-direct product type 1
def mul_DCC_semi_direct_type_1(f, g, h, N,M):
    x = min([a for a in range(2, N) if pow(a, M, N) == 1])
    s = [pow(x, i, N) for i in range(M)]
    for i1 in range(M):
        for j1 in range(M):
            for i0 in range(N):
                for j0 in range(N):
                    h[((j1+i1)%M)*N + (s[j1]*i0+j0)%N] += f[i1*N+i0]*g[j1*N+j0]
                    
                    
                    
                    
## Type 2: Semi-direct product
def mul_DCC_semi_direct_type_2(f, g, h, n):

    def get_index_in_G(G, x):
        for i in range(len(G)):
            if G[i] == x:
                return i
        return -1

    print("inside the function!")

    N = int(np.sqrt(2*n))

    # group G
    G = []
    for i in range(N):
        if math.gcd(i, N) == 1:
            G.append(i)

    # group H
    H = list(range(N))

    for i in range(n):
        for j in range(n):

            element1 = (G[int(i % (N/2))], int(i / (N/2)))
            element2 = (G[int(j % (N/2))], int(j / (N/2)))

            g1, h1 = element1
            g2, h2 = element2

            mul = (g1*g2 % N, (pow(g2, -1, N)*h1 + h2) % N)

            idx = get_index_in_G(G, mul[0]) + mul[1]*int(N/2)

            h[idx] += f[i] * g[j]
            
            
            
def mul_2RLWE(f, g, h, n):
    # f[i * n + k] represents the coefficient of (y^i * x^k)
    N = int(np.sqrt(n))
    for i in range(N):              # y-degree of f
        for j in range(N):          # y-degree of g
            for k in range(N):          # x-degree of f_i(x)
                for l in range(N):      # x-degree of g_j(x)
                    t = ((-1)**((i+j)//N + (k+l)//N))
                    h[((i+j)%N)*N + (k+l)%N] +=  t* f[i*N + k] * g[j*N + l]
                    
                    
# Z[w][X,Y]/<X^N - 1, Y^3 - 1, XY - YX^s>, s^3 = 1 (mod N)
#
# (a+bw)(c+dw) = (ac-bd) + w(ad+bc-bd)

def mul_NTRU3_ETRU(f, g, h, n):

    N = n // 6

    s = [a for a in range(1, N) if pow(a, 3, N) == 1]

    for i in range(N):
        for j in range(N):

            # -------------------------------------------------
            # Y^0 * Y^0 -> Y^0
            # -------------------------------------------------

            h[((i*s[0]+j)%N)*2      ] += f[(i)*2      ] * g[(j)*2]
            h[((i*s[0]+j)%N)*2      ] += -f[(i)*2+1   ] * g[(j)*2+1]

            h[((i*s[0]+j)%N)*2 + 1  ] += f[(i)*2      ] * g[(j)*2+1]
            h[((i*s[0]+j)%N)*2 + 1  ] += f[(i)*2+1   ] * g[(j)*2]
            h[((i*s[0]+j)%N)*2 + 1  ] += -f[(i)*2+1  ] * g[(j)*2+1]


            # -------------------------------------------------
            # Y^1 * Y^0 -> Y^1
            # -------------------------------------------------

            h[((i*s[0]+j)%N + N)*2      ] += f[(i+N)*2      ] * g[(j)*2]
            h[((i*s[0]+j)%N + N)*2      ] += -f[(i+N)*2+1   ] * g[(j)*2+1]

            h[((i*s[0]+j)%N + N)*2 + 1  ] += f[(i+N)*2      ] * g[(j)*2+1]
            h[((i*s[0]+j)%N + N)*2 + 1  ] += f[(i+N)*2+1   ] * g[(j)*2]
            h[((i*s[0]+j)%N + N)*2 + 1  ] += -f[(i+N)*2+1  ] * g[(j)*2+1]


            # -------------------------------------------------
            # Y^2 * Y^0 -> Y^2
            # -------------------------------------------------

            h[((i*s[0]+j)%N + 2*N)*2      ] += f[(i+2*N)*2      ] * g[(j)*2]
            h[((i*s[0]+j)%N + 2*N)*2      ] += -f[(i+2*N)*2+1   ] * g[(j)*2+1]

            h[((i*s[0]+j)%N + 2*N)*2 + 1  ] += f[(i+2*N)*2      ] * g[(j)*2+1]
            h[((i*s[0]+j)%N + 2*N)*2 + 1  ] += f[(i+2*N)*2+1   ] * g[(j)*2]
            h[((i*s[0]+j)%N + 2*N)*2 + 1  ] += -f[(i+2*N)*2+1  ] * g[(j)*2+1]


            # -------------------------------------------------
            # Y^0 * Y^1 -> Y^1
            # -------------------------------------------------

            h[((i*s[1]+j)%N + N)*2      ] += f[(i)*2      ] * g[(j+N)*2]
            h[((i*s[1]+j)%N + N)*2      ] += -f[(i)*2+1   ] * g[(j+N)*2+1]

            h[((i*s[1]+j)%N + N)*2 + 1  ] += f[(i)*2      ] * g[(j+N)*2+1]
            h[((i*s[1]+j)%N + N)*2 + 1  ] += f[(i)*2+1   ] * g[(j+N)*2]
            h[((i*s[1]+j)%N + N)*2 + 1  ] += -f[(i)*2+1  ] * g[(j+N)*2+1]


            # -------------------------------------------------
            # Y^1 * Y^1 -> Y^2
            # -------------------------------------------------

            h[((i*s[1]+j)%N + 2*N)*2      ] += f[(i+N)*2      ] * g[(j+N)*2]
            h[((i*s[1]+j)%N + 2*N)*2      ] += -f[(i+N)*2+1   ] * g[(j+N)*2+1]

            h[((i*s[1]+j)%N + 2*N)*2 + 1  ] += f[(i+N)*2      ] * g[(j+N)*2+1]
            h[((i*s[1]+j)%N + 2*N)*2 + 1  ] += f[(i+N)*2+1   ] * g[(j+N)*2]
            h[((i*s[1]+j)%N + 2*N)*2 + 1  ] += -f[(i+N)*2+1  ] * g[(j+N)*2+1]


            # -------------------------------------------------
            # Y^2 * Y^1 -> Y^0
            # -------------------------------------------------

            h[((i*s[1]+j)%N)*2      ] += f[(i+2*N)*2      ] * g[(j+N)*2]
            h[((i*s[1]+j)%N)*2      ] += -f[(i+2*N)*2+1   ] * g[(j+N)*2+1]

            h[((i*s[1]+j)%N)*2 + 1  ] += f[(i+2*N)*2      ] * g[(j+N)*2+1]
            h[((i*s[1]+j)%N)*2 + 1  ] += f[(i+2*N)*2+1   ] * g[(j+N)*2]
            h[((i*s[1]+j)%N)*2 + 1  ] += -f[(i+2*N)*2+1  ] * g[(j+N)*2+1]


            # -------------------------------------------------
            # Y^0 * Y^2 -> Y^2
            # -------------------------------------------------

            h[((i*s[2]+j)%N + 2*N)*2      ] += f[(i)*2      ] * g[(j+2*N)*2]
            h[((i*s[2]+j)%N + 2*N)*2      ] += -f[(i)*2+1   ] * g[(j+2*N)*2+1]

            h[((i*s[2]+j)%N + 2*N)*2 + 1  ] += f[(i)*2      ] * g[(j+2*N)*2+1]
            h[((i*s[2]+j)%N + 2*N)*2 + 1  ] += f[(i)*2+1   ] * g[(j+2*N)*2]
            h[((i*s[2]+j)%N + 2*N)*2 + 1  ] += -f[(i)*2+1  ] * g[(j+2*N)*2+1]


            # -------------------------------------------------
            # Y^1 * Y^2 -> Y^0
            # -------------------------------------------------

            h[((i*s[2]+j)%N)*2      ] += f[(i+N)*2      ] * g[(j+2*N)*2]
            h[((i*s[2]+j)%N)*2      ] += -f[(i+N)*2+1   ] * g[(j+2*N)*2+1]

            h[((i*s[2]+j)%N)*2 + 1  ] += f[(i+N)*2      ] * g[(j+2*N)*2+1]
            h[((i*s[2]+j)%N)*2 + 1  ] += f[(i+N)*2+1   ] * g[(j+2*N)*2]
            h[((i*s[2]+j)%N)*2 + 1  ] += -f[(i+N)*2+1  ] * g[(j+2*N)*2+1]


            # -------------------------------------------------
            # Y^2 * Y^2 -> Y^1
            # -------------------------------------------------

            h[((i*s[2]+j)%N + N)*2      ] += f[(i+2*N)*2      ] * g[(j+2*N)*2]
            h[((i*s[2]+j)%N + N)*2      ] += -f[(i+2*N)*2+1   ] * g[(j+2*N)*2+1]

            h[((i*s[2]+j)%N + N)*2 + 1  ] += f[(i+2*N)*2      ] * g[(j+2*N)*2+1]
            h[((i*s[2]+j)%N + N)*2 + 1  ] += f[(i+2*N)*2+1   ] * g[(j+2*N)*2]
            h[((i*s[2]+j)%N + N)*2 + 1  ] += -f[(i+2*N)*2+1  ] * g[(j+2*N)*2+1]

