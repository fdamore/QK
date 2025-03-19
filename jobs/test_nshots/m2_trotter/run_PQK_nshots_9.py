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

print(current_wd)

from pqk.PQK_SVC_PE import PQK_SVC_PE
from pqk.Circuits import Circuits
from pqk.CKernels import CKernels

#set the seed
np.random.seed(123)
algorithm_globals.random_seed = 123

#quantum stuff: observable, circuit to encoding and number type of function used to measure that uses shots
my_obs = ['XXIIII', 'IXXIII','IIXXII', 'IIIXXI','IIIIXI','XIIIIX',
          'YYIIII', 'IYYIII','IIYYII', 'IIIYYI','IIIIYY','YIIIIY',
          'ZZIIII', 'IZZIII','IIZZII', 'IIIZZI','IIIIZZ','ZIIIIZ']
q_c = Circuits.Trotter_HuangE3(n_wire=6)

#load dataset with panda
#data are scaled outside the notebook
f_rate = 1 #rate of data sampling fot testing pourpose
data_file_csv = './data/env.sel3.sk_sc.csv'
env = pd.read_csv(data_file_csv).sample(frac=f_rate, random_state=123)  

#DEFINE design matrix
Y = env['occupancy']
X = env[['illuminance', 'blinds','lamps','rh', 'co2', 'temp']]


#split design matrix using similar 
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=2580, test_size=280,random_state=123)
#WARNING: convert data to numpy. Quantum stuff (Qiskit) do not like PANDAS
X_train_np = X_train.to_numpy()
y_train_np = y_train.to_numpy()

X_test_np = X_test.to_numpy()
y_test_np = y_test.to_numpy()

#deinf the number of the shots
n_shots_list = range(400,450,1)#run_1

#get time
t_start = time.time()

list_score = []

#Best parameter: {'C': 4.0, 'gamma': 5.0, 'kernel': 'rbf'}
#define hyperparamenter PQK_M2_TROTTER_18_obs
C_ = 4
gamma_ = 5

#clear the cache
clear_cache = True

#check the shape of test and training dataset
print(f'File used for this run: {data_file_csv}')
print(f'Fraction rate used for this run: {f_rate * 100}%')
print(f'Shape of dataset: {env.shape}')
print(f'Training shape dataset {X_train_np.shape}')
print(f'Label for traing {y_train_np.shape}')
print(f'Test shape dataset {X_test_np.shape}')
print(f'Label for test {y_test_np.shape}')
print(f'Clear cache? {clear_cache}')
print(f'C: {C_}')
print(f'Gamma: {gamma_}')
print(f'OBS: {my_obs}')
print(f'Quantum C: {q_c.name}')


for n_shot_ in n_shots_list:
    
    pqk = PQK_SVC_PE(C=C_, fit_clear=clear_cache, gamma=gamma_, circuit=q_c, obs=my_obs, c_kernel='rbf', nshots=n_shot_, shots_seed=123)
        
    svm_quantum = pqk.fit(X_train_np, y_train_np)
    #result...
    predictions = svm_quantum.predict(X_test_np)
    score = accuracy_score(predictions, y_test)

    list_score.append(score)

    
#get time training
t_training = time.time()


#final time (trainign + predict)
t_final = time.time()


print(f'Final time {t_final - t_start} seconds')

#save info.
np.savetxt("nhsots_9.txt", np.array(n_shots_list))
np.savetxt("scores_9.txt", np.array(list_score))
