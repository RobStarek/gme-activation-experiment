# %%
import itertools
import numpy as np
# import MaxLik as ml
import matplotlib.pyplot as plt
import dvml
import logging

logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('numpy').setLevel(logging.WARNING)
logging.getLogger('dvml').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

LO = np.array([[1],[0]])
HI = np.array([[0],[1]])
HLO = (LO+HI)*(2**-.5)
HHI = (LO-HI)*(2**-.5)
CLO = (LO+1j*HI)*(2**-.5)
CHI = (LO-1j*HI)*(2**-.5)

def kron(*arrays) -> np.ndarray:
    """
    Multiple Kronecker (tensor) product.
    Multiplication is performed from left.    
    """
    E = np.eye(1, dtype=complex)
    for M in arrays:
        E = np.kron(E,M)
    return E

# eigenstates = {
#     'x' : (HHI, HLO),
#     'y' : (CLO, CHI),
#     'z' : (LO, HI)
# }

# rpv = np.array([
#     [kron(*(eigenstates[key][k] for k, key in zip(j, bases)))]
#     for bases in itertools.product('zyx', repeat=3) 
#     for j in itertools.product((0,1), repeat=3)
#     ])
# rpv = rpv.reshape((216,8,1))


eigenbase_dict = {
    'I' : ((LO, HI), (1,1)),
    'X' : ((HLO, HHI), (1,-1)), #no sign flip
    'Y' : ((CLO, CHI),(1,-1)),
    'Z' : ((LO, HI), (1,-1)),
}

def base_string_to_proj(string : str) -> list:
    """
    Input measurement string and get projection operators corresponding to that string, 
    all combinations that can happen.    
    """
    eigenvects = [eigenbase_dict.get(b)[0] for b in string]
    proj_kets = [kron(*vecs) for i, vecs in enumerate(itertools.product(*eigenvects))]
    return proj_kets

lintrap_order = 'XYZ'
tomo_str_gen = (''.join(mstr) for mstr in itertools.product(lintrap_order, repeat=3))
meas3 = np.array([base_string_to_proj(mstr) for mstr in tomo_str_gen])
meas3 = meas3.reshape((27*8, 8, 1))

# %%
state_order = ['a1','a2','a3','a4','b1','b2','b3','b4','c1','c2']
data024 = np.load('newtomo/3Tomo_024.npy')
data531 = np.load('newtomo/3Tomo_531.npy')

new_keys = [f'a{i}' for i in range(1,5)] + [f'b{i}' for i in range(1,5)] + [f'c{i}' for i in range(1,3)]
index_map = {key : state_order.index(key) for key in new_keys}
print(index_map)

# %%
tomos_024 = np.array([data024[index_map[s]].ravel() for s in state_order])
tomos_531 = np.array([data531[index_map[s]].ravel() for s in state_order])

# %%
SAMPLES = 100
rec = dvml.Reconstructer(meas3, max_iters = 1000, thres=1e-12, batch_size = 2*10*SAMPLES)
rho_seeds_024 = rec.reconstruct(tomos_024) 
rho_seeds_531 = rec.reconstruct(tomos_531) 

# %%
#bootstrapping
SHOTS = 200


rng = np.random.default_rng()

mc_tomo024 = rng.multinomial(SHOTS, tomos_024.reshape((10,27,8)), size=(SAMPLES,10,27)).reshape((SAMPLES, 10, -1))
mc_tomo531 = rng.multinomial(SHOTS, tomos_531.reshape((10,27,8)), size=(SAMPLES,10,27)).reshape((SAMPLES, 10, -1))

tomo_batch = np.vstack([mc_tomo024.reshape((-1,216)), mc_tomo531.reshape((-1,216))])
print(tomo_batch.shape)

# %%
mc_rhos_batch = rec.reconstruct(tomo_batch)

# %%
mc_rhos_batch = mc_rhos_batch.reshape((2, SAMPLES, 10, 8, 8))

# %%

# %%
import h5py
with h5py.File('mc_rhos_200shots.h5', 'w') as h5f:
    h5f.create_dataset('rhos_024', data = mc_rhos_batch[0])
    h5f.create_dataset('rhos_531', data = mc_rhos_batch[1])
    h5f.create_dataset('rhos_024_orig', data = rho_seeds_024)
    h5f.create_dataset('rhos_531_orig', data = rho_seeds_531)
    print('done')


