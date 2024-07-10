
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

from qke.TrainableKernelFeatureMap import TrainableKernelFeatureMap
from qke.TrainableCircuits import TrainableCircuits
from qke.QKCallback import QKCallback
from qke.QMeasures import QMeasures
from qke.CKernels import CKernels
from qiskit_algorithms.utils import algorithm_globals

#set the seed
np.random.seed(123)
algorithm_globals.random_seed = 123

#load dataset with panda
#data are scaled outside the notebook
#sclaled_data_file = 'data/env.sel3.scaled.csv'
data_file_csv = 'data/env.sel3.minmax.csv' 
env = pd.read_csv(data_file_csv)  


#DEFINE design matrix

f_rate = 1
env_slice = env.sample(frac=f_rate, random_state=123) #slices the origin dataset

Y = env_slice['occupancy']
X = env_slice[['illuminance', 'blinds','lamps','rh', 'co2', 'temp']]

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
t_circuit = TrainableCircuits.zzfm(n_wire=NUM_QBIT)
#t_circuit = TrainableCircuits.d_stack(n_wire=NUM_QBIT)
#t_circuit = TrainableCircuits.twl_zzfm(n_wire=NUM_QBIT)
#t_circuit = TrainableCircuits.trainable_twl(n_wire=NUM_QBIT)

fm = t_circuit.qc
training_params = t_circuit.training_parameters

#show feature map
print(f'*** TRAINABLE FEATURE MAP used in QSVC')
print(fm.draw())

#define callback
my_callback = QKCallback()

#define the trainable kernel
my_obs = ['ZIIIII', 'IZIIII','IIZIII', 'IIIZII','IIIIZI','IIIIIZ']
nshots = 100 #paramenter using primitive estimator

measure_fn = QMeasures.StateVectorEstimator
kernel = CKernels.linear

#q_kernel = TrainableOuterQuantumKernel(feature_map=fm, training_parameters=training_params)
q_kernel = TrainableKernelFeatureMap(feature_map=fm, training_parameters=training_params)
q_kernel.configure(obs=my_obs, nshots=nshots, q_measure=QMeasures.StateVectorEstimator, c_kernel=CKernels.rbf)

#print this info
print(f'The QMeasure function used: {q_kernel.q_measure.__name__}')
print(f'The classical kernel used: {q_kernel.kernel.__name__}')
print(f'The observables we use: {my_obs}')
print(f'The numbers of shots (if applicable) for (qiskit) primitive estimator: {nshots}')

#define updater, loss and inizial param
spsa_opt = SPSA(maxiter=max_iter, learning_rate=0.03, perturbation=0.01, termination_checker=my_callback.callback)
loss_func = SVCLoss(C=1.0)
#init_point=[np.pi/2 for _ in range(NUM_QBIT)] #TODO: try random values (o pi)

init_point = np.random.uniform(size=NUM_QBIT)
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

print(f'*******SCORE: {score}')
print(f'Time kernel training: {training_kernel_end - training_kernel_start} seconds.')
print(f'Time training SVM: {training_svm_end - training_svm_start} seconds.')
print(f'Total jobs time: {jobs_final_time - training_kernel_start} seconds.')

