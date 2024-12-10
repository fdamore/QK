import sys
import os

#define working directory and package for QK
current_wd = os.getcwd()
sys.path.append(current_wd)

import time
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from qiskit_machine_learning.algorithms.classifiers import QSVC
from qiskit_machine_learning.kernels.algorithms.quantum_kernel_trainer import QuantumKernelTrainer
from qiskit_algorithms.optimizers import SPSA
from qiskit_machine_learning.utils.loss_functions import SVCLoss

from pqk.TrainablePQK_SVC import TrainablePQK_SVC
from pqk.QKCallback import QKCallback
from pqk.QMeasures import QMeasures
from pqk.CKernels import CKernels
from pqk.Circuits import Circuits
from qiskit_algorithms.utils import algorithm_globals



#set the seed
seed=123
np.random.seed(seed)
algorithm_globals.random_seed = seed


my_obs = ['XIIIII', 'IXIIII','IIXIII', 'IIIXII','IIIIXI','IIIIIX','YIIIII', 'IYIIII','IIYIII', 'IIIYII','IIIIYI','IIIIIY','ZIIIII', 'IZIIII','IIZIII', 'IIIZII','IIIIZI','IIIIIZ']
#my_obs = ['XIIIII', 'IXIIII','IIXIII', 'IIIXII','IIIIXI','IIIIIX']
#my_obs = ['YIIIII', 'IYIIII','IIYIII', 'IIIYII','IIIIYI','IIIIIY']
#my_obs = ['ZIIIII', 'IZIIII','IIZIII', 'IIIZII','IIIIZI','IIIIIZ']

#set the seed
np.random.seed(123)
algorithm_globals.random_seed = 123

#load dataset with panda
#data are scaled outside the notebook
f_rate = 1 #rate of data sampling fot testing pourpose
#data_file_csv = 'data/env.sel3.scaled.csv'
data_file_csv = 'data/env.sel3.sk_sc.csv'
env = pd.read_csv(data_file_csv).sample(frac=f_rate, random_state=seed)    

Y = env['occupancy']
X = env[['illuminance', 'blinds','lamps','rh', 'co2', 'temp']]

#split design matrix (25% of the design matrix used for test)
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=123)

#cast to numpy object
X_train = X_train.to_numpy()
y_train = y_train.to_numpy()
X_test = X_test.to_numpy()
y_test = y_test.to_numpy()

#define the maxiter paramenter
max_iter = 20

#check the shape of test and training dataset
print(f'Using dataset in datafile: {data_file_csv}')
print(f'Fraction rate used for this run: {f_rate * 100}%')
print(f'Max number of iteration used in kernel optimization: {max_iter}')
print(f'Shape of dataset: {env.shape}')
print(f'Training shape dataset {X_train.shape}')
print(f'Label for traing {y_train.shape}')
print(f'Test shape dataset {X_test.shape}')
print(f'Label for test {y_test.shape}')

#build a feature map 
NUM_QBIT = X_train.shape[1]

#define the circuits
encoding_circuit = Circuits.xyz_encoded(n_wire=NUM_QBIT)
trainable_circuit = Circuits.zy_decomposition(param_prefix='tr', n_wire=NUM_QBIT, full_ent=False)
encoding_circuit.barrier()
fm = encoding_circuit.compose(trainable_circuit)
training_params = trainable_circuit.parameters
n_trainables = len(training_params)

#show feature map
print(f'*** TRAINABLE FEATURE MAP used in QSVC')
print(fm.draw())
print(f'Number of trainable paramenters: {n_trainables}')

#define callback
my_callback = QKCallback()

q_kernel = TrainablePQK_SVC(feature_map=fm, training_parameters=training_params, obs=my_obs, measure_fn=QMeasures.StateVectorEstimator, c_kernel=CKernels.rbf)



#print this info
print(f'The QMeasure function used: {q_kernel.measure_fn.__name__}')
print(f'The classical kernel used: {q_kernel.c_kernel.__name__}')
print(f'The observables we use: {my_obs}')

#define updater, loss and inizial param
spsa_opt = SPSA(maxiter=max_iter, learning_rate=0.03, perturbation=0.01, termination_checker=my_callback.callback)
loss_func = SVCLoss(C=1.0)
#init_point=[np.pi/2 for _ in range(NUM_QBIT)] #TODO: try random values (o pi)

init_point = np.random.uniform(size=n_trainables)
print(f'Initial point: {init_point}')


#get time
training_kernel_start = time.time()

qk_trainer = QuantumKernelTrainer(quantum_kernel=q_kernel, loss=loss_func, initial_point= init_point, optimizer=spsa_opt)
qkt_results = qk_trainer.fit(X_train, y_train)
optimized_kernel = qkt_results.quantum_kernel

training_kernel_end = time.time()

#using optimezed kernel in QSVC
qsvc = QSVC(quantum_kernel=optimized_kernel)

#get time
training_svm_start = time.time()

qsvc.fit(X_train, y_train)

#get time training
training_svm_end = time.time()

#result...
predictions = qsvc.predict(X_test)
score = accuracy_score(predictions, y_test)

#final time (trainign + predict)
jobs_final_time = time.time()

#save the callback
my_callback.save(prefix='TR_')

print(f'*******SCORE: {score}')
print(f'Time kernel training: {training_kernel_end - training_kernel_start} seconds.')
print(f'Time training SVM: {training_svm_end - training_svm_start} seconds.')
print(f'Total jobs time: {jobs_final_time - training_kernel_start} seconds.')

