import itertools
from enum import IntEnum
import numpy as np
import matplotlib.pyplot as plt
import h5py
from concurrent.futures import ProcessPoolExecutor

import departed
from bisep_checker import (
    check_bisep,
    partitions,
    EPSILON,
    dagger,
    ketbra,
    kron,
    theo_rhos_3q,
    BinKet,
    HLO,
    HHI,
    LO,
    HI,
    CLO,
    CHI,
    purity,
)


def psd_projection(r: np.ndarray) -> np.ndarray:
    # force Hermiteanity
    hr = (r + r.T.conj()) * 0.5
    # do spectral decomposition
    evals, evecs = np.linalg.eigh(hr)
    # clip negative eigenvalues, 1e-16 is numerical epsilon
    devals = np.diag(np.clip(evals.real, 1e-16, None))
    # project to PSD matrix
    rp = evecs @ devals @ evecs.T.conj()
    # renorm and return
    return rp / np.trace(rp.real)

# check PPT
def ppt(rho, pt : partitions):
    if pt == partitions.A_BC:
        mask1 = [1,0,0]
    elif pt == partitions.AC_B:
        mask1 = [0,1,0]
    elif pt == partitions.AB_C:
        mask1 = [0,0,1]        
    else:
        print("Ooops!")
        raise    
    rhopt = departed.ptranspose(rho, [2]*3, mask1)
    mineig = np.min(np.linalg.eigvalsh(rhopt))
    return mineig

def test_function(rho):
    p = 0
    rho_tested = (1-p)*rho + p*np.eye(8)/8
    suc, pur_last, rmd, pur_first, counter = check_bisep(rho_tested, 250)
    return suc, pur_last, pur_first, counter
    
if __name__ == '__main__':
    with h5py.File("mc_rhos_200shots.h5") as h5f:
        rhos_single_copy_024 = np.array(h5f["rhos_024"])
        rhos_single_copy_531 = np.array(h5f["rhos_531"])
        rho_seed_024 = np.array(h5f["rhos_024_orig"])
        rho_seed_531 = np.array(h5f["rhos_531_orig"])
    
    rho_seed_024 = np.array([psd_projection(_r) for _r in rho_seed_024])
    rho_seed_531 = np.array([psd_projection(_r) for _r in rho_seed_531])
    
    q = 0.06
    wghts = [(1-q)/8]*8 + [q/2]*2

    rho_exp_024 = sum([w*r for r, w in zip(rho_seed_024, wghts)])
    rho_exp_531 = sum([w*r for r, w in zip(rho_seed_531, wghts)])
    
    wghts = np.array(wghts)
    rhos_single_copy_024_summed = np.sum(rhos_single_copy_024 * wghts.reshape((1,10,1,1)), axis=1)
    rhos_single_copy_531_summed = np.sum(rhos_single_copy_531 * wghts.reshape((1,10,1,1)), axis=1)
        

    n_cpu = 24
    with ProcessPoolExecutor(n_cpu) as pool:
        results_024 = list(pool.map(test_function, rhos_single_copy_024_summed))
    with ProcessPoolExecutor(n_cpu) as pool:
        results_531 = list(pool.map(test_function, rhos_single_copy_531_summed))
    results_024 = np.array(results_024)
    results_531 = np.array(results_531)
    seed_res_024 = test_function(rho_exp_024)
    seed_res_531 = test_function(rho_exp_531)
    print("Saving")
    np.savez('mc_bisep_results_0_200shots.npz', results_024 = results_024, results_531 = results_531, seed_res_024 = seed_res_024, seed_res_531 = seed_res_531)    
    fraction_024 = np.sum(results_024[:,0].astype(int))/results_024.size
    fraction_531 = np.sum(results_531[:,0].astype(int))/results_531.size
    print('fraction 024', fraction_024)
    print('fraction 531', fraction_531)
    print("Done.")
