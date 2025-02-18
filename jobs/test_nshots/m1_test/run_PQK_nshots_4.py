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

from pqk.PQK_SVC_PE import PQK_SVC_PE
from pqk.Circuits import Circuits
from pqk.QMeasures import QMeasures
from pqk.CKernels import CKernels

#set the seed
np.random.seed(123)
algorithm_globals.random_seed = 123

#quantum stuff: observable, circuit to encoding and number type of function used to measure that uses shots
my_obs = ['XIIIII', 'IXIIII','IIXIII', 'IIIXII','IIIIXI','IIIIIX',
          'YIIIII', 'IYIIII','IIYIII', 'IIIYII','IIIIYI','IIIIIY',
          'ZIIIII', 'IZIIII','IIZIII', 'IIIZII','IIIIZI','IIIIIZ']
q_c = Circuits.xyz_encoded(full_ent=False, n_wire=6)

#load dataset with panda
#data are scaled outside the notebook
f_rate = 1 #rate of data sampling fot testing pourpose
data_file_csv = 'data/env.sel3.sk_sc.csv'
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

#check the shape of test and training dataset
print(f'File used for this run: {data_file_csv}')
print(f'Fraction rate used for this run: {f_rate * 100}%')
print(f'Shape of dataset: {env.shape}')
print(f'Training shape dataset {X_train_np.shape}')
print(f'Label for traing {y_train_np.shape}')
print(f'Test shape dataset {X_test_np.shape}')
print(f'Label for test {y_test_np.shape}')


#deinf the number of the shots
n_shots_list = range(150,200,1)#run_1

#get time
t_start = time.time()

list_score = []

#define hyperparamenter PQK_M1_3D_FALSE
C_ = 128
gamma_ = 1e-7

for n_shot_ in n_shots_list:
    
    pqk = PQK_SVC_PE(C=C_, gamma=gamma_, circuit=q_c, obs=my_obs, c_kernel=CKernels.rbf, nshots=n_shot_, shots_seed=123)
        
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
np.savetxt("nhsots_4.txt", np.array(n_shots_list))
np.savetxt("scores_4.txt", np.array(list_score))
