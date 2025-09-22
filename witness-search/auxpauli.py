from functools import reduce
import numpy as np
import KetSugar as ks


def gen_base10_to_base_m(m):
    """
    Get a function that maps base10 integer to 
    list of base m representation (most significant first)
    """
    def _f(i, digits):
        powers = (m**np.arange(digits))[::-1]
        iact = i
        indices = []
        for j in range(digits):
            idx = iact // powers[j]
            indices.append(idx)
            iact -= (indices[-1]*powers[j])
        return indices
    return _f

def gen_base_m_to_base10(m):
    """
    Get a function that maps list of base m digits (most significant first)
    to base10 integer.
    """    
    def _f(args, n):
        powers = (m**np.arange(n))[::-1]
        return np.array(args) @ powers
    return _f

#auxiliary conversion functions
base10_to_base4 = gen_base10_to_base_m(4)
base10_to_base6 = gen_base10_to_base_m(6)
base6_to_base10 = gen_base_m_to_base10(6)

def spawn_generalized_pauli_operators(*basis_mats):
    """
    Generate list of generalized Pauli operators
    constructed from given basis vectors.
    Args:
        *basis_mats ... n 2x2 array of basis column vectors
    Returns:
        4**n operators
        indexing:        
        j=0 identity
        j=1 sigma z
        j=2 sigma x
        j=3 sigma y
        I = sum_j i_j*4**(n-j-1)
    """
    isqrt = 2**(-.5)
    pauli_ops_base = []
    for basis in basis_mats:
        low = basis[:,0].reshape((2,1))
        high = basis[:,1].reshape((2,1))
        kz0 = low
        kz1 = high
        kx0 = (low+high)*isqrt
        kx1 = (low-high)*isqrt
        ky0 = (low+1j*high)*isqrt
        ky1 = (low-1j*high)*isqrt
        pauli_ops_base.append([np.eye(2)] + [ks.ketbra(ket1, ket1) - ks.ketbra(ket2, ket2) for ket1, ket2 in [(kz0, kz1), (kx0, kx1), (ky0, ky1)]])
    
    n = len(basis_mats)
    gammas = []
    for i in range(4**n):
        indices = base10_to_base4(i, n)
        operators = [pauli_ops_base[j][idx] for j, idx in enumerate(indices)]                    
        gamma = reduce(np.kron, operators)
        gammas.append(gamma)
    return gammas


def iterate_generalized_pauli_operators(*basis_mats, **kwargs):
    """
    Generate list of generalized Pauli operators
    constructed from given basis vectors.
    Args:
        *basis_mats ... n 2x2 array of basis column vectors
    Returns:
        4**n operators
        indexing:        
        j=0 identity
        j=1 sigma z
        j=2 sigma x
        j=3 sigma y
        I = sum_j i_j*4**(n-j-1)
    """
    n = len(basis_mats)
    start = kwargs.get('start', 0)
    stop = kwargs.get('stop', 4**n)
    isqrt = 2**(-.5)
    pauli_ops_base = []
    for basis in basis_mats:
        low = basis[:,0].reshape((2,1))
        high = basis[:,1].reshape((2,1))
        kz0 = low
        kz1 = high
        kx0 = (low+high)*isqrt
        kx1 = (low-high)*isqrt
        ky0 = (low+1j*high)*isqrt
        ky1 = (low-1j*high)*isqrt
        pauli_ops_base.append([np.eye(2)] + [ks.ketbra(ket1, ket1) - ks.ketbra(ket2, ket2) for ket1, ket2 in [(kz0, kz1), (kx0, kx1), (ky0, ky1)]])
    

    for i in range(start, stop):
        indices = base10_to_base4(i, n)
        operators = [pauli_ops_base[j][idx] for j, idx in enumerate(indices)]                    
        gamma = reduce(np.kron, operators)
        yield(i, gamma)
    