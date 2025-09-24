# gme-activation-experiment

Repository for data and scripts regarding the GME activation experiment on trapped ion platform.

---

## Project Structure Overview

### `ion-experiment/`
Scripts, notebooks, and data for experimental analysis and state reconstruction.
- **Notebooks:**  
  - `witnessing-two-copy-gme.ipynb` — Main analysis of two-copy GME activation.
  - `extrapolate-three-copies.ipynb` — Extrapolation to three-copy scenarios.
  - `reconstruct.ipynb` — State reconstruction from tomograms.
- **Scripts:**  
  - `bisep_checker.py`, `check_biseparability.py` — Biseparability checking tools.
  - `MaxLik.py` — Maximum likelihood state estimation.
  - `auxpauli.py`, `KetSugar.py` — Pauli operator and quantum state utilities.
- **Data:**  
  - `data/` — Experimental and processed data files (HDF5, NPY, NPZ).
  - `newtomo/` — Raw and processed tomogram files.
  - `bisep-check/` — Monte Carlo biseparability checks and results.
  - `old/` — Legacy scripts and data.

### `witness-search/`
Scripts and notebooks for witness operator search and construction.
- `find-two-copy-witness.ipynb` — SDP search for two-copy witness.
- `find-three-copy-witness.py` — Three-copy witness search.
- `auxpauli.py`, `KetSugar.py` — Shared quantum mechanics utilities.
- `two-copy-witness.h5` — Precomputed witness operator and measurement table.

---

## Data Files

- **Tomograms:**  
  - `three_qubit_tomograms.h5` - quantum state tomography of individual copies, metadata is in h5 attributes of the datasets
  - `wit_data_27_05_25_Block2.h5`- actual two-copy witness measurement, shape is (100,17,64), where the first axis represents various constituent state, second axis represents witness measurements, and the last axis contains measured probabilities serialized from outcome `000000` to outcome `111111`.
- **Witness Operators:**  
  - `two-copy-witness.h5`, `w9.npy`, `w9.npz`
- **Reconstructed States:**  
  - `reconstructed_three_qubit_states.h5`, `temp_single_copy_q050.npy`

---

## Usage

- Start with notebooks in `ion-experiment/` for data analysis and visualization.
- Use scripts in `witness-search/` for witness construction and SDP optimization.
- Refer to `data/` and `newtomo/` for input tomograms and results.

---

*For more details, see comments in individual scripts and notebooks.*