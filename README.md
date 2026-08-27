# QK — Projected Quantum Kernels

This repository contains the reference implementation of **Projected Quantum Kernels (PQK)** for classification tasks, together with quantum feature maps, quantum kernel-based classifiers (QSVM/PQK-SVC), and one-class variants for anomaly detection, built on top of [Qiskit](https://www.ibm.com/quantum/qiskit).

It is the base code used to produce the results presented in the following two papers:

## 📄 Related Publications

1. **Conference paper** — F. D'Amore, L. Mariani, C. Mastroianni, J. Settino, A. Vinci, *"Projected Quantum Kernel for IoT Data Analysis"*, in *2024 IEEE Conference on Pervasive and Intelligent Computing (PICom)*, Boracay Island, Philippines, 2024, pp. 173–177.
   DOI: [10.1109/PICom64201.2024.00032](https://ieeexplore.ieee.org/abstract/document/10795417)

2. **Journal paper** — F. D'Amore, L. Mariani, C. Mastroianni, F. Plastina, L. Salatino, J. Settino, A. Vinci, *"Assessing projected quantum kernels for the classification of IoT data"*, *Array*, vol. 29, 2026, art. no. 100695.
   DOI: [10.1016/j.array.2026.100695](https://www.sciencedirect.com/science/article/pii/S2590005626000184)

If you use this code in your research, please cite the paper(s) above, and/or the software itself using the metadata in [CITATION.cff](CITATION.cff).

## Overview

The project explores **quantum kernel methods** for supervised (classification) and unsupervised (anomaly detection) machine learning, focusing on the IoT/environmental sensing use case (occupancy detection from environmental data). It implements:

- Several **quantum feature maps / encodings** (angle encoding, ZZ feature map, IQP-style, Trotterized, data re-uploading, etc.) for different numbers of qubits and entanglement layouts.
- **Projected Quantum Kernels (PQK)**, computed by projecting quantum states onto reduced/measurable observables and then applying a classical kernel (e.g. RBF, linear) on the projected features.
- Classical **Quantum Support Vector Classifiers (QSVM)** and **PQK-based SVCs**, including one-class variants (OC-SVM) for anomaly detection.
- Utilities for hyperparameter grid search, cross-validation, and result/metric collection (accuracy, F1, p-values).

## Repository Structure

```
main.py                 Entry point / quick sanity check
pqk/                     Core library: circuits, kernels, PQK-SVC, callbacks, measures
qencodings/              Standalone scripts defining specific quantum encodings (angle, ZZ, IQP, Trotter, data re-uploading, ...)
qfm/                     Quantum feature map building blocks (circuits, callbacks, trainable feature maps)
jobs/                    Experiment scripts (training, grid search, statistical tests) and their outputs
notebooks/               Exploratory / validation notebooks
data/                    Datasets and preprocessing notebooks
results/                 Accuracy / F1 result files produced by the experiments
plots/                   Generated plots
origin/                  Original/raw source data
```

Key modules in [pqk/](pqk):
- [Circuits.py](pqk/Circuits.py) — parametrized quantum circuits / encodings
- [PQK_SVC.py](pqk/PQK_SVC.py) / [PQK_SVC_PE.py](pqk/PQK_SVC_PE.py) — Projected Quantum Kernel SVC implementations
- [TrainablePQK_SVC.py](pqk/TrainablePQK_SVC.py) — trainable/parametrized version of the PQK-SVC
- [CKernels.py](pqk/CKernels.py) — classical kernel functions applied on projected features
- [QMeasures.py](pqk/QMeasures.py) — quantum measurement/estimation routines
- [QEncoding.py](pqk/QEncoding.py) / [QEncodingMC.py](pqk/QEncodingMC.py) — quantum encoding helpers
- [QKCallback.py](pqk/QKCallback.py) — training callbacks

## Requirements & Installation

Requires Python 3.10+ (tested with the pinned versions in [requirements.txt](requirements.txt)).

```bash
python -m venv .venv
source .venv/bin/activate

# CPU-only
pip install -r requirements.txt

# or, for GPU-accelerated simulation
pip install -r requirements_gpu.txt
```

## Usage

Experiment scripts are located under [jobs/](jobs), e.g.:

```bash
python jobs/run_PQK.py          # train/evaluate a PQK-SVC on the IoT dataset
python jobs/run_QSVM.py         # train/evaluate a quantum kernel SVM
python jobs/run_OCSVM_AD.py     # one-class quantum kernel SVM for anomaly detection
python jobs/run_PQK_GS.py       # grid search over PQK hyperparameters
```

Datasets used by the experiments are in [data/](data) (e.g. `env.sel3.sk_sc.csv`, the scaled environmental/occupancy dataset).

## Citation

See [CITATION.cff](CITATION.cff) for the software citation metadata, and the [Related Publications](#-related-publications) section above for the associated papers.

## License

See [LICENSE](LICENSE).
