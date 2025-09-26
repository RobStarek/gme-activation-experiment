# gme-activation-experiment

![optional cover image](ghibli_cover.png)

Repository for data and Python scripts regarding the GME activation experiment on trapped ion platform.

* TODO: Arxiv number
* TODO: Zenodo link

---

## Project Structure Overview

### `ion-experiment/`
Scripts, notebooks, and data for experimental analysis and state reconstruction.
- **Notebooks:**  
  - `witnessing-two-copy-gme.ipynb` — Main analysis of two-copy GME activation experiment.
  - `extrapolate-three-copies.ipynb` — Extrapolation to three-copy scenario.
  - `reconstruct.ipynb` — State reconstruction from tomograms.
  - `characterize-reconstruction.ipynb` — Calculate figures of merit of the single-copy states.
- **Scripts:**  
  - `bisep_checker.py`, `check_biseparability.py` — biseparability checking.
  - `MaxLik.py` — Maximum likelihood state estimation.
  - `auxpauli.py`, `KetSugar.py` — Pauli operator and quantum state utilities.
- **Data:**  
  - `data/` — Experimental and processed data files (HDF5, NPY, NPZ), and intermediate results.
  

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
  - `wit_data_27_05_25_Block2.h5`- actual two-copy witness measurement
- **Witness Operators:**  
  - `two-copy-witness.h5`, `w9.npy`, `w9.npz`
- **Reconstructed States:**  
  - `reconstructed_three_qubit_states.h5`


- **Structure of state tomograms:**
  - The H5 file contains two dataset: `ions_024` and `ions_531` for the first and the second copy, respectively.
  - Each dataset has shape (10, 27, 8), wherein the first index selects the prepared constituent state (see `state_order` attribute), the second index selects the tomographic measurement (see `tomo_order` attribute), and the last index selects the measurement outcome (`000`, `001`, ..., `111`). The entries in the dataset are measured outcomes probabilities for each state and measurement. They were calculated as number of counts devided by number of shots and stored. We used 200 shots per configuration.

- **Structure of witness tomogram:**
  - The H5 file contains 100 named datasets, corrsponding to the combination of used constituent states, from `a1,a1` to `c2,c2`.
  - Each dataset is table 17x64, where the first index selects the witness measurement in accordance to `two_copy_table_meas` dataset of the `two-copy-witness.h5` file, and the second index determines the measurement outcome (`000000`, `000001`, ..., `111111`).

The data files can be conveniently inspected using [MyHDF5](https://myhdf5.hdfgroup.org) tool.

---

## Usage

- Start with notebooks in `ion-experiment/` for data analysis and visualization.
- Use scripts in `witness-search/` for witness construction and SDP optimization.
- Refer to `data/`for input tomograms and results.

---

*For more details, see comments in individual scripts and notebooks.*
