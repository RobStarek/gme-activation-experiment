"""
## Description of the algorithm
**In a nutshell:**

We use the same algo as here [doi:10.1038/nphys1781](https://www.nature.com/articles/nphys1781), but instead of subtracting one pure separable state at time, we subtract biseparable mixtures. 
Instead of searching for maximum by partial eigenprojections, we utilize generalized Bloch angle parameterization.
The challenge is to find the mixture to be subtracted, but here we use prior information about components which we prepare and find find the constituents in the mixture by biasing the current matrix toward the theoretical components. This is the key modification because if we would subtract just pure separable states, it would actually increase purity of the remainder, violating one of the requirements of the original algorithm.

**Inputs:**

* Density matrix $\rho_0$ to be tested
* Maximum number of iterations permitted
* (implicit) List of theoretical pure separable states
* (implicit) List of possible partitions of the components

**Procedure:**
- 1 - find maximally overlapping biseparable mixture $\eta=\sum\limits_{j}^{8} |\psi_{j}\rangle\langle\psi_{j}|w_j$
    - 1.1 - for each theoretical constituent term, make a biased version of $\rho_i$, i.e.
    $$
    \tilde{\rho}_{ij} = b\rho_i + (1-b)|\psi_{j}\rangle\langle\psi_{j}|,
    $$
    where theoretical state $|\psi_{j}\rangle$ is separable with some bipartition $k$.
    - 1.2. - for each biased density matrix, find maximally overlapping pure state $|\tilde{\psi_{j}} \rangle $, separable with some bipartition $k$ by maximization
    $$
    \max |\langle\tilde{\psi}_{j}| \tilde{\rho}_{ij} |\tilde{\psi}_{j}\rangle|
    $$
    over $|\tilde{\psi}_{j}\rangle$, this is done using generalized Bloch angles.
    - 1.3. optimize weight $w_j$ of mixture to maximize overlap with $\rho_i$.

- 2 - subtract $\epsilon_o \eta$ from $\rho_i$ to minimize purity of the trace-normalized remainder, 
$$
\epsilon_o = \argmin_\epsilon \mathrm{Tr}\left[\left(\frac{\rho_i - \epsilon\eta}{\mathrm{Tr}[\rho_i - \epsilon\eta]}\right)^2\right]
$$
- 3 - assign the normalized remainder to $\rho_{i+1}$, 
$$
\rho_{i+1} := \left(\frac{\rho_i - \epsilon_o \eta}{\mathrm{Tr}[\rho_i - \epsilon_o \eta]}\right)
$$
- 4 - if purity of $rho_{i+1}$ is below threshold 1/7, then stop iteration and yield that state was biseparable, otherwise go to point 1. If number of iteration exceeds limit, stop iteration and state inconclusive result.
"""

from enum import IntEnum
from typing import Tuple, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import Tuple

#numerical tolerance
EPSILON = 1e-10

class partitions(IntEnum):
    """Bipartitions"""
    A_BC = 0
    AB_C = 1
    AC_B = 2


LO = np.array([[1],[0]])
HI = np.array([[0],[1]])
HLO = (LO+HI)*(2**-.5)
HHI = (LO-HI)*(2**-.5)
CLO = (LO+1j*HI)*(2**-.5)
CHI = (LO-1j*HI)*(2**-.5)


def dagger(x : np.ndarray) -> np.ndarray:
    """
    Hermite conjugation of x.
    """
    return x.T.conjugate()

def ketbra(x : np.ndarray, y : np.ndarray) -> np.ndarray:
    """
    Outer product of two ket-vectors -> C-matrix
    """
    return np.dot(x, y.T.conjugate())

def kron(*arrays) -> np.ndarray:
    """
    Multiple Kronecker (tensor) product.
    Multiplication is performed from left.    
    """
    E = np.eye(1, dtype=complex)
    for M in arrays:
        E = np.kron(E,M)
    return E

def BinKet(i : int = 0,imx : int =1) -> np.ndarray:
    """
    Computational base states i in imx+1-dimensional vectors.
    """
    ket = np.zeros((imx+1,1), dtype=complex)
    ket[i] = 1
    return ket

def ApplyOp(Rho : np.ndarray,M : np.ndarray) -> np.ndarray:
    """
    Calculate M.Rho.dagger(M).
    """
    return M @ Rho @ M.T.conjugate()

def ExpectationValue(Ket : np.ndarray, M : np.ndarray) -> np.ndarray:
    """
    Expectation value <bra|M|ket>.
    """
    return (Ket.T.conjugate() @ M @ Ket)[0,0]

def purity(rho):
    #more efficient equivalent of return np.trace(rho @ rho)
    return (rho.T.ravel() @ rho.ravel()).real

def swap_ij(n : int, i : int, j : int) -> np.ndarray:
    """Generate swap matrix of qubits i,j in n-qubit space."""
    left_qubits = i
    central_qubits = (j - i) - 1
    right_qubits = (n - j) - 1
    eye_left = np.eye(1 << left_qubits)
    eye_center = np.eye(1 << central_qubits)
    eye_right = np.eye(1 << right_qubits)    
    ket_01 = kron(eye_left, LO, eye_center, HI, eye_right)
    ket_10 = kron(eye_left, HI, eye_center, LO, eye_right)
    bra_01 = dagger(ket_01)
    bra_10 = dagger(ket_10)
    swap = np.eye(1 << n, dtype=complex) \
        - ket_01 @ bra_01 - ket_10 @ bra_10 \
        + ket_01 @ bra_10 + ket_10 @ bra_01
    return swap

##--------------
# Theoretical prior information
Sgate = np.array([[1, 0], [0, 1j]])
SI = np.eye(2)
SZ = np.diag([1,-1])
SX = np.array([[0,1],[1,0]])
SY = np.array([[0,-1j],[1j,0]])

prep_gate = kron(Sgate, Sgate, SZ)
bellp = (BinKet(0b00, 3) + BinKet(0b11, 3))/np.sqrt(2)
bellm = (BinKet(0b00, 3) - BinKet(0b11, 3))/np.sqrt(2)
bell02p = (BinKet(0b000, 7) + BinKet(0b101, 7)+BinKet(0b010, 7) + BinKet(0b111, 7))/2
bell02m = (BinKet(0b000, 7) - BinKet(0b101, 7)-BinKet(0b010, 7) + BinKet(0b111, 7))/2

#single-copy constituent states as ket vectors
theo_kets_3q = {
    'a1' : kron(HLO, bellp),
    'a2' : kron(HHI, bellm),
    'a3' : prep_gate @ kron(HLO, bellp),
    'a4' : prep_gate @ kron(HHI, bellm),
    'b1' : bell02p,
    'b2' : bell02m,
    'b3' : prep_gate @ bell02p,
    'b4' : prep_gate @ bell02m,
    'c1' : BinKet(0b001, 7),
    'c2' : BinKet(0b110, 7)
}
theo_rhos_3q = {key : ketbra(ket, ket) for key, ket in theo_kets_3q.items()}


def _gen_bloch_angles_to_ket(chi, theta1, theta2, phi1, phi2, gamma):
    # Precompute trigonometric values
    c_chi = np.cos(chi / 2)
    s_chi = np.sin(chi / 2)
    c_t1 = np.cos(theta1 / 2)
    s_t1 = np.sin(theta1 / 2)
    c_t2 = np.cos(theta2 / 2)
    s_t2 = np.sin(theta2 / 2)
    exp_igamma2 = np.exp(1j * gamma / 2)
    exp_migamma2 = np.exp(-1j * gamma / 2)
    exp_miphi_sum = np.exp(-1j * (phi1 + phi2) / 2)
    exp_miphi_diff = np.exp(-1j * (phi1 - phi2) / 2)
    exp_piphi_diff = np.exp(1j * (phi1 - phi2) / 2)
    exp_piphi_sum = np.exp(1j * (phi1 + phi2) / 2)
    a = (c_chi * c_t1 * c_t2 * exp_igamma2 + s_chi * s_t1 * s_t2 * exp_migamma2) * exp_miphi_sum
    b = (c_chi * c_t1 * s_t2 * exp_igamma2 - s_chi * s_t1 * c_t2 * exp_migamma2) * exp_miphi_diff
    c = (c_chi * s_t1 * c_t2 * exp_igamma2 - s_chi * c_t1 * s_t2 * exp_migamma2) * exp_piphi_diff
    d = (c_chi * s_t1 * s_t2 * exp_igamma2 + s_chi * c_t1 * c_t2 * exp_migamma2) * exp_piphi_sum
    return a, b, c, d

def gen_bloch_ket(
    chi: float, 
    theta1: float, 
    theta2: float, 
    phi1: float, 
    phi2: float, 
    gamma: float
) -> np.ndarray:
    return np.array(_gen_bloch_angles_to_ket(chi, theta1, theta2, phi1, phi2, gamma)).reshape((-1,1))


#Ket constructors
def bloch_ket(theta, phi):
    """
    Return a qubit column-vector with Bloch sphere coordinates theta, phi.
    theta - lattitude measured from |0> on Bloch sphere
    phi - longitude measured from |+> on Bloch sphere (phase)
    """
    return np.array([[np.cos(theta/2)], [np.sin(theta/2)*np.exp(1j*phi)]])


def find_overlapping_state(rho: np.ndarray, partition=partitions.A_BC) -> Tuple[np.ndarray, float]:
    """
    Find a pure state separable to given partitions that highly overlaps the checked rho.

    Returns:
        Tuple[np.ndarray, float]: The density matrix of the found pure state and its (negative) fidelity.
    """
    #for simplicity it is writtin for A|BC partition. 
    #any other partition is done by swapping qubits to A|BC partition and then back
    opswap = reordering.get(partition, np.eye(8))
    rho = ApplyOp(rho, opswap)

    def foo(x):
        """Fidelity of ket defined using gen. bloch angles to the entered state."""
        x1, x2, x3, x4, x5, x6, x7, x8 = x
        veca = bloch_ket(x1, x2)
        vecbc = gen_bloch_ket(x3, x4, x5, x6, x7, x8)
        vec = np.kron(veca, vecbc)
        return -ExpectationValue(vec, rho).real
    #
    res = minimize(foo, x0=np.random.random(8)*np.pi, method='Nelder-Mead')
    fid = res['fun']
    x1, x2, x3, x4, x5, x6, x7, x8 = res['x'] + (np.random.random(8)-0.5)*np.pi/128

    veca = bloch_ket(x1, x2)
    vecbc = gen_bloch_ket(x3, x4, x5, x6, x7, x8)
    vec = np.kron(veca, vecbc)
    rho = ketbra(vec, vec)
    rho = ApplyOp(rho, opswap.T)
    return rho, fid

def mapper(
    rho: np.ndarray, 
    rho_ab: np.ndarray, 
    eps: float, 
    renorm: bool = True
) -> np.ndarray:
    """
    Subtracts a scaled biseparable mixture (rho_ab) from the input density matrix (rho).
    
    Parameters:
    ----------
    rho : np.ndarray
        The input density matrix.
    rho_ab : np.ndarray
        The biseparable mixture to subtract.
    eps : float
        Subtraction weight.
    renorm : bool, optional
        If True, normalize the result to have trace 1 (default: True).
    
    Returns:
    -------
    np.ndarray
        The resulting (optionally normalized) density matrix after subtraction.
    """
    r = (rho - rho_ab * eps)
    if not renorm:
        return r
    return r / np.trace(r)


def find_starting_point(
    rho: np.ndarray,
    rho_ab: np.ndarray,
    min_exp: int = -10,
    max_exp: int = -1,
    pts: int = 91
) -> float:
    """
    Finds an optimal starting value for the parameter `eps` by searching over a logarithmic range,
    applying the `mapper` function to `rho` and `rho_ab`, and selecting the value that minimizes
    the purity of mapped matrix, subject to positive eigenvalue constraints.
    Args:
        rho (np.ndarray): The input density matrix.
        rho_ab (np.ndarray): The auxiliary density matrix used in the mapping.
        min_exp (int, optional): The minimum exponent for the logarithmic search range (default: -10).
        max_exp (int, optional): The maximum exponent for the logarithmic search range (default: -1).
        pts (int, optional): The number of points in the logarithmic search range (default: 91).
    Returns:
        float: The selected starting value for `eps` that satisfies the eigenvalue constraint and
               minimizes the trace of the squared mapped matrix.
    """
    eps = np.logspace(min_exp, max_exp, pts)
    arr1 = []
    #trace decreasing search
    for e in eps:
        rho2 = mapper(rho, rho_ab, e, True)
        mineigs = np.min(np.linalg.eigvalsh(rho2))   
        if mineigs < -EPSILON:
            break
        f = np.trace(rho2 @ rho2).real
        arr1.append((e, mineigs, f))
    if len(arr1)==0:
        arr1 = [(1e-10, 0, 0)]            
    arr1 = np.array(arr1)    
    subarr = arr1[arr1[:,1] > - EPSILON, :] 
    imin = np.argmin(subarr[:, 2])
    return subarr[imin, 0]

reordering = {
    partitions.AB_C : swap_ij(3, 0, 2),
    partitions.AC_B : swap_ij(3, 0, 1),
    partitions.A_BC : np.eye(8)
}

def find_closest_theo(rho):
    arr = [np.trace(theo_rhos_3q[key] @ rho).real for key in theo_rhos_3q]
    key = list(theo_rhos_3q.keys())[np.argmax(arr)]
    return theo_rhos_3q[key], key
        

def check_bisep(
    initial_rho: np.ndarray,  
    maximum_rounds: int, 
) -> Tuple[bool, float, np.ndarray, float]:
    """
    Check the biseparability of a given 3-qubit density matrix.

    Biseparability refers to whether a quantum state can be expressed as a mixture 
    of separable states across different partitions of the qubits. This function 
    iteratively subtracts biseparable mixtures from the input density matrix until 
    either the purity of the remainder falls below a threshold or the maximum number 
    of iterations is reached.

    Parameters:
    ----------
    initial_rho : np.ndarray
        The input 3-qubit density matrix to check for biseparability.
    theo_rhos_3q : dict[str, np.ndarray]
        A dictionary containing theoretical biseparable states for different partitions.
    maximum_rounds : int
        Maximum number of iterations allowed for the process.

    Returns:
    -------
    Tuple[bool, float, np.ndarray, float]
        - `is_biseparable` (bool): Indicates whether the input density matrix is biseparable.
        - `final_purity` (float): Purity of the remainder after the iterative process.
        - `final_remainder` (np.ndarray): The normalized remainder density matrix.
        - `original_purity` (float): Purity of the input density matrix.

    Notes:
    -----
    - If `DEBUG` is enabled (in function definition), the function visualizes the current state of the density matrix, 
      remainder, biseparable mixture, and normalized remainder using `matplotlib`.
    - The process terminates either when the purity of the remainder falls below `PUR_THR` 
      or when the maximum number of iterations (`maximum_rounds`) is reached.
    """
    DEBUG = False
    PUR_THR = 1/7 + EPSILON #for qubit|(qubit x qubit) systems
    rho = np.copy(initial_rho)
    pur = purity(rho)
    counter = 0
    pur00 = pur    

    for counter in range(maximum_rounds):
        
        _rhos = []
        for key in list(theo_rhos_3q.keys())[:-2]:
            pt = partitions.AC_B if 'b' in key else partitions.A_BC
            _rho_abi, fid = find_overlapping_state(rho*0.999 + 0.001*theo_rhos_3q[key], pt)
            _rhos.append(_rho_abi*np.abs(fid))
        _rhos = np.array(_rhos)
        def cost_function(x):
            """Overlap of bisep mixture to be subtracted with the previous remainder"""
            sm = sum([w*r for w, r in zip(x, _rhos)])
            sm /= np.trace(sm)
            return float(-np.trace(sm @ rho).real)
        bnds = [(0,1)]*len(_rhos)
        optres = minimize(cost_function, x0= np.ones(8)*0.125, bounds = bnds)
        rho_ab = sum([w*r for w, r in zip(optres['x'], _rhos)])
        rho_ab = rho_ab/np.trace(rho_ab)
        eps = find_starting_point(rho, rho_ab)
        remainder = mapper(rho, rho_ab, eps, False)
        nremainder = remainder/np.trace(remainder)
        pur0 = purity(rho/np.trace(rho))
        pur = purity(nremainder)
        if DEBUG and (counter% 100)==0:
            plt.matshow(np.hstack([
                np.abs(rho), 
                np.abs(remainder), 
                rho_ab.real, 
                np.abs(nremainder)
                ]))
            plt.title(f'step={counter}, eps={eps:.1e}, P={pur0:.3e}->{pur:.3e}, (rho|rem|ovl|new)')
            plt.colorbar()
            plt.show()              
            answer = input()
            if answer=='q':
                break  
        if pur <= PUR_THR:
            print(f"\nFinished in round {counter}")
            return True, pur, nremainder, pur00
        counter += 1        
        print(f'{counter}: P={pur:.3e}, F={fid*(-1):.3e}, eps={eps:.3e}', end='\r')
        rho = nremainder          
    print(f"Counter: {counter} exceeded limit.")
    return False, pur, nremainder, pur00