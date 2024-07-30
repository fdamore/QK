import sys
import os
import time
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from qiskit_algorithms.utils import algorithm_globals

#define working directory and package for QK
current_wd = os.getcwd()
sys.path.append(current_wd)

from qke.CircuitContainer import CircuitContainer
from qke.Circuits import Circuits
from qke.QMeasures import QMeasures
from qke.CKernels import CKernels



#set the seed
np.random.seed(123)
algorithm_globals.random_seed = 123


my_obs = ['XIIIII', 'IXIIII','IIXIII', 'IIIXII','IIIIXI','IIIIIX','YIIIII', 'IYIIII','IIYIII', 'IIIYII','IIIIYI','IIIIIY','ZIIIII', 'IZIIII','IIZIII', 'IIIZII','IIIIZI','IIIIIZ']
c = CircuitContainer(qtemplate=Circuits.xyz_encoded, full_ent=False, nwire=6, obs=my_obs, measure_fn=QMeasures.StateVectorEstimator, c_kernel=CKernels.rbf)



#load dataset with panda
#data are scaled outside the notebook
f_rate = 1 #rate of data sampling fot testing pourpose
data_file_csv = 'data/env.sel3.scaled.csv'
env = pd.read_csv(data_file_csv).sample(frac=f_rate, random_state=123)  

#DEFINE design matrix
Y = env['occupancy']
X = env[['illuminance', 'blinds','lamps','rh', 'co2', 'temp']]

#split design matrix (25% of the design matrix used for test)
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=123)
#WARNING: convert data to numpy. Quantum stuff (Qiskit) do not like PANDAS
X_train_np = X_train.to_numpy()
y_train_np = y_train.to_numpy()

X_test_np = X_test.to_numpy()
y_test_np = y_test.to_numpy()

#check the shape of test and training dataset
print(f'File used for this run: {data_file_csv}')
print(f'Fraction rate used for this run: {f_rate * 100}%')
print(f'Shape of dataset: {env.shape}')
print(f'Training shape dataset {X_train_np.shape}')
print(f'Label for traing {y_train_np.shape}')
print(f'Test shape dataset {X_test_np.shape}')
print(f'Label for test {y_test_np.shape}')

#get time
t_start = time.time()

svm_quantum = SVC(kernel=CircuitContainer.kernel_matrix).fit(X_train_np, y_train_np);
print(f'Sanity check. Dict len after training: {len(c.fm_dict)}')

#get time training
t_training = time.time()

#result...
predictions = svm_quantum.predict(X_test_np)
score = accuracy_score(predictions, y_test)

#final time (trainign + predict)
t_final = time.time()

print(f'*******SCORE: {score}')
print(f'Time training: {t_training - t_start} seconds. Final time {t_final - t_start} seconds')
print(f'Sanity check. Dict len after prediction: {len(c.fm_dict)}')

c.save_feature_map(prefix='run_xyz_')