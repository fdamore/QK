import os
import sys

current_wd = os.getcwd()
sys.path.append(current_wd)

from pqk.Circuits import Circuits
import pandas as pd
from pqk.aux_funcs import adjacent_qub_obs
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
obs = adjacent_qub_obs(['X','Y','Z'], n_qub=6, n_measured_qub=2)
print(f'Observations: used to project quantum states: {obs}')

#used circuits
qc =  Circuits.xyz_encoded(n_wire=6, full_ent=False, dr_layers=2, dr_sep = 'cnot')
print(qc.draw('text'))

q_enc = QEncoding(data=X__np, obs=obs, qcircuit=qc)
en = q_enc.encode()

q_enc.save_encoding(file_name='QC_3D_OBS_M2_L2.csv',y_label= y_np)


