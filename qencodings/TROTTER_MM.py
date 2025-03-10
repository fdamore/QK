import os
import sys

current_wd = os.getcwd()
sys.path.append(current_wd)

from pqk.Circuits import Circuits
import pandas as pd
from pqk.aux_funcs import adjacent_qub_obs, generate_my_obs
from pqk.QEncoding import QEncoding

source_file = 'data/env.sel3.sk_sc.csv'
env = pd.read_csv(source_file)

#DEFINE design matrix
Y = env['occupancy']
X = env[['illuminance', 'blinds','lamps','rh', 'co2', 'temp']]

X__np = X.to_numpy()
y_np = Y.to_numpy()

#shape of ndarray
print('Shape of data to encode:')
print(f'Data: {X__np.shape}')
print(f'Label: {y_np.shape}')

#generate obs
obs_m1 = generate_my_obs(['X','Y','Z'], n_qub=6)
obs_m2 = adjacent_qub_obs(['X','Y','Z'], n_qub=6, n_measured_qub=2)
obs_m1_m2 = obs_m1 + obs_m2

print(f'Observations: used to project quantum states: {obs_m1_m2}')
print(f'Number of observations: {len(obs_m1_m2)}')

#used circuits
qc =  Circuits.Trotter_HuangE3(n_wire=6)
print(qc.draw('text'))

q_enc = QEncoding(data=X__np, obs=obs_m1_m2, qcircuit=qc)
en = q_enc.encode()

q_enc.save_encoding(file_name='QC_TROTTER_OBS_MM.csv',y_label= y_np)


