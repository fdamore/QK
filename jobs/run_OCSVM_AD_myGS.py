import sys
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from qiskit_algorithms.utils import algorithm_globals

current_wd = os.getcwd()
sys.path.append(current_wd)

from pqk.Circuits import Circuits
from pqk.QMeasures import QMeasures
from pqk.PQK_SVC import PQK_OCSVC
from pqk.aux_funcs import *

# -----------------------------
# Global seed
# -----------------------------
seed = 123
np.random.seed(seed)
algorithm_globals.random_seed = seed

# -----------------------------
# Synthetic dataset
# -----------------------------
n_norm = 100
n_anom = 10
n_features = 2

X_norm = 0.5 * np.random.randn(n_norm, n_features)
X_anom = np.random.uniform(low=-4, high=4, size=(n_anom, n_features))

scaler = StandardScaler()
X_norm = scaler.fit_transform(X_norm)
X_anom = scaler.transform(X_anom)

# -----------------------------
# PQK encoding & measurement
# -----------------------------
full_ent=True
encoding_key = 'xyz'
my_obs_key = 'XYZ'
measure_fn_key = 'CPU'
cvfolds = 5

encoding_dict = {
    'xyz': Circuits.xyz_encoded(full_ent=full_ent, n_wire=n_features),   # change to 3d ? 
    'zz': Circuits.zzfeaturemap(full_ent=full_ent, n_wire=n_features), 
    'x': Circuits.x_encoded(full_ent=full_ent, n_wire=n_features), 
    'spiral': Circuits.spiral_encoding(full_ent=full_ent, n_wire=n_features, n_windings=1),
    'uniform': Circuits.uniform_bloch_encoding(full_ent=full_ent, n_wire=n_features),
    'IQP': Circuits.IQP_HuangE2(n_wire=n_features),
    'Trotter': Circuits.Trotter_HuangE3(n_wire=n_features),
    }   

pauli_meas_dict = {
    'XYZ' : generate_my_obs(['X','Y','Z'], n_qub=n_features),
    'XY' : generate_my_obs(['X','Y'], n_qub=n_features),
    'X' : generate_my_obs(['X'], n_qub=n_features),
    'Y' : generate_my_obs(['Y'], n_qub=n_features),
    'Z' : generate_my_obs(['Z'], n_qub=n_features),
    'BLOCH_XYZ' : generate_my_obs(['X','Y','Z'], n_qub=n_features//2), #for the uniform bloch:   (couldnt make it work yet - Luca)
    'NON_LOCAL_XX' : generate_my_obs(['X','Y','Z','XX'], n_qub=n_features),
    'ADJAC_2QUB' : adjacent_qub_obs(['X','Y','Z'], n_qub=n_features, n_measured_qub=2),
    'ADJAC_XX' : adjacent_qub_obs(['X'], n_qub=n_features, n_measured_qub=2),
    'ADJAC_YY' : adjacent_qub_obs(['Y'], n_qub=n_features, n_measured_qub=2),
    'ADJAC_ZZ' : adjacent_qub_obs(['Z'], n_qub=n_features, n_measured_qub=2),
}
pauli_meas_dict['ADJAC_2QUB_EXTRA'] = pauli_meas_dict['XYZ'] + pauli_meas_dict['ADJAC_2QUB']

measure_fn_dict = {
    'CPU' : QMeasures.StateVectorEstimator,
    'GPU' : QMeasures.GPUAerStateVectorEstimator,
    'GPUfakenoise' : QMeasures.GPUAerVigoNoiseStateVectorEstimator,
}

clear_cache = False
my_obs = pauli_meas_dict[my_obs_key]
print(f'Selected observables: {my_obs}')

# -----------------------------
# CV setup
# -----------------------------
kf = KFold(n_splits=cvfolds, shuffle=True, random_state=seed)

nus = [0.01, 0.05, 0.1]
gammas = [0.01, 0.1, 1.0]
param_grid = [(nu, gamma) for nu in nus for gamma in gammas]

results = []

# Log fold-by-fold
log_fold = []

# -----------------------------
# Grid search con tqdm
# -----------------------------
for nu, gamma in tqdm(param_grid, desc="Grid search (nu,gamma)"):
    f1_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(tqdm(kf.split(X_norm), total=cvfolds, desc=f"Folds nu={nu} gamma={gamma}", leave=False)):
        X_train = X_norm[train_idx]
        X_val_norm = X_norm[val_idx]
        X_val = np.vstack([X_val_norm, X_anom])
        y_val = np.hstack([np.ones(len(X_val_norm)), -np.ones(len(X_anom))])

        model = PQK_OCSVC(
            circuit=encoding_dict[encoding_key],
            obs=my_obs,
            measure_fn=measure_fn_dict[measure_fn_key],
            nu=nu,
            gamma=gamma,
            fit_clear=True,
            pqk_verbose=False
        )

        t0 = time.time()
        model.fit(X_train)
        t1 = time.time()

        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred, pos_label=-1)
        f1_scores.append(f1)

        # log fold
        log_fold.append({
            'nu': nu,
            'gamma': gamma,
            'fold': fold_idx+1,
            'f1': f1,
            'fit_time_s': t1-t0
        })

    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores, ddof=1)
    results.append((nu, gamma, mean_f1, std_f1))

# -----------------------------
# Best parameters
# -----------------------------
best = max(results, key=lambda x: x[2])
best_nu, best_gamma = best[0], best[1]

# -----------------------------
# Salva log fold-by-fold
# -----------------------------
df_log = pd.DataFrame(log_fold)
log_file = f'pqk_ocs_fold_log.csv'
df_log.to_csv(log_file, index=False)

# -----------------------------
# Fold migliore
# -----------------------------
best_fold = df_log[(df_log['nu']==best_nu) & (df_log['gamma']==best_gamma)].sort_values('f1', ascending=False).iloc[0]
best_fold_f1 = best_fold['f1']
best_fold_idx = best_fold['fold']

# Ricalcola X_train/X_val della fold migliore
for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_norm)):
    if fold_idx+1 == best_fold_idx:
        X_train = X_norm[train_idx]
        X_val_norm = X_norm[val_idx]
        X_val = np.vstack([X_val_norm, X_anom])
        y_val = np.hstack([np.ones(len(X_val_norm)), -np.ones(len(X_anom))])
        break

# Calcola accuracy sulla fold migliore
model = PQK_OCSVC(
    circuit=encoding_dict[encoding_key],
    obs=my_obs,
    measure_fn=measure_fn_dict[measure_fn_key],
    nu=best_nu,
    gamma=best_gamma,
    fit_clear=True,
    pqk_verbose=False
)
model.fit(X_train)
y_pred = model.predict(X_val)
best_fold_acc = np.mean(y_pred == y_val)

# -----------------------------
# Risultati finali
# -----------------------------
print(f"Best parameters: nu={best_nu}, gamma={best_gamma}, mean F1={best[2]:.4f} ± {best[3]:.4f}")
print(f"Best fold: {best_fold_idx}, F1={best_fold_f1:.4f}, Accuracy={best_fold_acc:.4f}")
print(f"Log fold-by-fold salvato in {log_file}")
