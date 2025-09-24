from time import time
import functools
import itertools
import pickle
import numpy as np
import scipy.sparse as spr
import matplotlib.pyplot as plt
import picos
import KetSugar as ks

def p_thr(k):
    N = 3 #3-qubit GHZ
    sub_a = 2**(N-1)
    sub_b = sub_a -1
    sub_c = sub_b**(1/k)
    #print(sub_a, sub_b, sub_c)
    return sub_c/(sub_a + sub_c)

def GHZ(p):
    ket = np.zeros((8,1), dtype=complex)
    ket[0,0] = 1
    ket[-1,0] = 1
    ket /= np.sqrt(2)
    return p*(ket @ ket.T.conj()) + (1-p)*np.eye(8)/8

def swap_ij(n, i, j):
    left_qubits = i
    central_qubits = (j - i) - 1
    right_qubits = (n - j) - 1
    eye_left = np.eye(1 << left_qubits)
    eye_center = np.eye(1 << central_qubits)
    eye_right = np.eye(1 << right_qubits)    
    ket_01 = ks.kron(eye_left, ks.LO, eye_center, ks.HI, eye_right)
    ket_10 = ks.kron(eye_left, ks.HI, eye_center, ks.LO, eye_right)
    bra_01 = ks.dagger(ket_01)
    bra_10 = ks.dagger(ket_10)
    swap = np.eye(1 << n, dtype=complex) \
        - ket_01 @ bra_01 - ket_10 @ bra_10 \
        + ket_01 @ bra_10 + ket_10 @ bra_01
    return swap
#nine-qubit swaps
swaps = {(i,j) : swap_ij(9, i, j) for i,j in [(1,3), (2,6), (5,7)]}


p_tgt =0.295
ghz_1copy = GHZ(p_tgt)
ghz_3copy = ks.kron(*[ghz_1copy]*3)
u_swap = swaps[(2,6)] @ swaps[(5,7)] @ swaps[(1,3)]
ghz = ks.ApplyOp(ghz_3copy, u_swap)

nonzeros = (np.abs(ghz) > 1e-6)
P = picos.Problem()
REF = picos.Constant(ghz)
W = 0
cntr = 0
w_variables = dict()
for row in range(512):
    for col in range(row+1):
        if nonzeros[row, col]:
            cntr += 1
            print(f'\r{row}, {col}, {cntr}', end="")
            wij_label = f'w_{row}_{col}'
            wij = picos.RealVariable(wij_label)
            w_variables[wij_label] = wij
            mij_npy = np.zeros((512,512))            
            mij_npy[row, col] = 1
            if row != col:
                mij_npy[col, row] = 1
            mij = picos.Constant(f"m_{row}_{col}", mij_npy)
            W = W + mij*wij
W = W.renamed("W")
P.add_constraint(picos.trace(W) == 1)
biparts = [
    [0,1,2],
    [3,4,5],
    [6,7,8]
]
for k in [0,1,2]:
    Pkdiag = picos.RealVariable(f'P_{k}', shape = (512,), lower=0, upper=1)    
    Pk = picos.diag(Pkdiag)
    print(Pk)
    Qk = picos.partial_transpose(W - Pk, biparts[k], [2,2,2,2,2,2,2,2,2])
    print(Qk)
    P.add_constraint(Qk >> 0)

print("Calculating fidelity")
F = 0
for row in range(512):
    for col in range(row+1):
        print(f'\r G:{row}, {col}', end="")
        wij_label = f'w_{row}_{col}'
        if wij_label in w_variables:
            F = F + w_variables[wij_label] * ghz[row, col]
            if col != row:
                F = F + w_variables[wij_label] * ghz[col, row]
print("Setting the goal...")
P.set_objective('min', F)

mosek_params = {
    #"MSK_IPAR_NUM_THREADS" : 1,
    "MSK_IPAR_LOG" : 50,
    #"MSK_IPAR_MIO_MEMORY_EMPHASIS_LEVEL" : 1
}

print("Starting...")
LIM = 64*1024
t0 = time()
#this works but witness is positive
S = P.solve(solver = "mosek", verbosity = 2, mosek_params = mosek_params, dualize=True, treememory=64*1024)
#S = P.solve(solver = "mosek", verbosity = 2, dualize=False)
t1 = time()
print(t1-t0, "sec")
print("done")

P0a = P.get_variable('P_0').np
P1a = P.get_variable('P_1').np
P2a = P.get_variable('P_2').np
np.savez("w9nd.npz", W = np.array(W.value), w = F.value.real, P0 = P0a, P1 = P1a, P2 = P2a)


#--- Results in ---
# Optimizer started.
# Presolve started.
# Linear dependency checker started.
# Linear dependency checker terminated.
# Eliminator started.
# Freed constraints in eliminator : 0
# Eliminator terminated.
# Eliminator - tries                  : 1                 time                   : 0.00
# Lin. dep.  - tries                  : 1                 time                   : 0.00
# Lin. dep.  - primal attempts        : 1                 successes              : 1
# Lin. dep.  - dual attempts          : 0                 successes              : 0
# Lin. dep.  - primal deps.           : 0                 dual deps.             : 0
# Presolve terminated. Time: 0.00
# GP based matrix reordering started.
# GP based matrix reordering terminated.
# Optimizer  - threads                : 24
# Optimizer  - solved problem         : the primal
# Optimizer  - Constraints            : 2292
# Optimizer  - Cones                  : 1
# Optimizer  - Scalar variables       : 3074              conic                  : 2
# Optimizer  - Semi-definite variables: 3                 scalarized             : 393984
# Factor     - setup time             : 0.14
# Factor     - dense det. time        : 0.05              GP order time          : 0.00
# Factor     - nonzeros before factor : 1.84e+06          after factor           : 1.84e+06
# Factor     - dense dim.             : 0                 flops                  : 1.77e+09
# ITE PFEAS    DFEAS    GFEAS    PRSTATUS   POBJ              DOBJ              MU       TIME
# 0   3.0e+00  1.0e+00  1.0e+00  0.00e+00   -0.000000000e+00  -0.000000000e+00  1.0e+00  0.20
# 1   1.3e+00  4.4e-01  2.9e-01  1.01e+00   2.667820706e-03   4.201958151e-04   4.4e-01  0.42
# 2   2.7e-02  8.8e-03  7.4e-04  1.01e+00   1.633673239e-04   1.921873481e-03   8.8e-03  0.64
# 3   5.6e-03  1.9e-03  3.0e-05  1.49e+00   8.654994565e-04   1.938999998e-03   1.9e-03  0.87
# 4   1.2e-03  3.9e-04  4.1e-06  2.77e+00   1.903216694e-03   1.931632206e-03   3.9e-04  1.11
# 5   1.7e-04  5.5e-05  2.1e-07  1.05e+00   1.682342426e-03   1.686390335e-03   5.5e-05  1.35
# 6   3.9e-05  1.3e-05  2.0e-08  1.01e+00   8.705070413e-04   8.723432146e-04   1.3e-05  1.59
# 7   1.2e-05  4.1e-06  3.3e-09  1.00e+00   4.893297127e-04   4.900136338e-04   4.1e-06  1.84
# 8   4.3e-06  1.4e-06  6.5e-10  1.00e+00   -1.580450113e-04  -1.577689308e-04  1.4e-06  2.06
# 9   1.8e-07  6.1e-08  5.6e-12  1.00e+00   -3.707956880e-04  -3.707837760e-04  6.1e-08  2.30
# 10  3.6e-09  9.6e-12  1.1e-17  1.00e+00   -3.853006210e-04  -3.853006191e-04  9.7e-12  2.52
# Optimizer terminated. Time: 2.53
