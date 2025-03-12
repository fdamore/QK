import datetime
import sys
import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from qiskit_algorithms.utils import algorithm_globals

#define working directory and package for QK
current_wd = os.getcwd()
sys.path.append(current_wd)

from pqk.PQK_SVC import PQK_SVC
from pqk.Circuits import Circuits
from pqk.QMeasures import QMeasures
from pqk.CKernels import CKernels



#set the seed
np.random.seed(123)
algorithm_globals.random_seed = 123

# Best paramenter: {'C': 32.0, 'gamma': 0.01}
my_obs = ['XIIIII', 'IXIIII','IIXIII', 'IIIXII','IIIIXI','IIIIIX','YIIIII', 'IYIIII','IIYIII', 'IIIYII','IIIIYI','IIIIIY','ZIIIII', 'IZIIII','IIZIII', 'IIIZII','IIIIZI','IIIIIZ']

q_c = Circuits.xyz_encoded(full_ent=False, n_wire=6)

pqk = PQK_SVC(C=32, gamma=0.01, circuit=q_c, obs=my_obs, measure_fn=QMeasures.StateVectorEstimator, c_kernel='linear')

#my_obs = ['XIIIII', 'IXIIII','IIXIII', 'IIIXII','IIIIXI','IIIIIX']
#my_obs = ['YIIIII', 'IYIIII','IIYIII', 'IIIYII','IIIIYI','IIIIIY']
#my_obs = ['ZIIIII', 'IZIIII','IIZIII', 'IIIZII','IIIIZI','IIIIIZ']


#print metadata
pqk.metadata()

#load dataset with panda
#data are scaled outside the notebook
f_rate = 0.01 #rate of data sampling fot testing pourpose
data_file_csv = 'data/env.sel3.sk_sc.csv'
env = pd.read_csv(data_file_csv).sample(frac=f_rate, random_state=123)  

#DEFINE design matrix
Y = env['occupancy']
X = env[['illuminance', 'blinds','lamps','rh', 'co2', 'temp']]

#split design matrix (25% of the design matrix used for test)
#X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=123)

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=123, shuffle=False) #used to crate qenconding

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

svm_quantum = pqk.fit(X_train_np, y_train_np);
print(f'Sanity check. Dict len after training: {len(pqk._fm_dict)}')

#get time training
t_training = time.time()

#result...
predictions = svm_quantum.predict(X_test_np)
score = accuracy_score(predictions, y_test)

#final time (trainign + predict)
t_final = time.time()

print(f'*******SCORE: {score}')
print(f'Time training: {t_training - t_start} seconds. Final time {t_final - t_start} seconds')
print(f'Sanity check. Dict len after prediction: {len(pqk._fm_dict)}')

#time simulation info
datetime_object = datetime.datetime.fromtimestamp(t_final)
formatted_datetime = datetime_object.strftime("%Y-%m-%d %H-%M-%S")
print(f'Simulation end at: {formatted_datetime}')

#save encondings
pqk.save_feature_map(prefix='xyz_')
pqk.save_latent_space(prefix='xyz_', suffix='temp.csv', y = np.append(y_train_np, y_test_np))