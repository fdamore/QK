import os
import sys

current_wd = os.getcwd()
sys.path.append(current_wd)

from pqk.Circuits import Circuits
import pandas as pd
from pqk.aux_funcs import generate_my_obs
from pqk.QEncoding import QEncoding

source_file = 'data/env.sel3.2pi_minmax.csv'
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
obs = generate_my_obs(['X','Y','Z'], n_qub=3)
print(f'Observations: used to project quantum states: {obs}')

#used circuits
qc =  Circuits.uniform_bloch_encoding(n_wire=3, full_ent=False)
print(qc.draw('text'))

q_enc = QEncoding(data=X__np, obs=obs, qcircuit=qc)
en = q_enc.encode()

q_enc.save_encoding(file_name='QC_UB_OBS_M1_ENT_FALSE.csv',y_label= y_np)


