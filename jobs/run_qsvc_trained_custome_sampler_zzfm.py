import sys
import os
import time
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from qiskit_machine_learning.algorithms.classifiers import QSVC
from qiskit.circuit import ParameterVector
from qiskit.circuit import QuantumCircuit
from qiskit_machine_learning.kernels.algorithms.quantum_kernel_trainer import QuantumKernelTrainer
from qiskit_algorithms.optimizers import SPSA
from qiskit_machine_learning.utils.loss_functions import SVCLoss
from qiskit_machine_learning.kernels import TrainableFidelityQuantumKernel
from qiskit.circuit.library import ZZFeatureMap

from qiskit_aer.primitives import Sampler as AerSampler # Aer Sampler
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_algorithms.utils import algorithm_globals

import numpy as np

#define working directory and package for QK
current_wd = os.getcwd()
sys.path.append(current_wd)

from qke.QKCallback import QKCallback

#set the seed
np.random.seed(123)

#load dataset with panda
#data are scaled outside the notebook
#sclaled_data_file = 'data/env.sel3.scaled.csv'
data_file_csv = 'data/env.sel3.minmax.csv' 
env = pd.read_csv(data_file_csv)  


#DEFINE design matrix
f_rate = 1
env_slice = env.sample(frac=f_rate) #slices the origin dataset

Y = env_slice['occupancy']
X = env_slice[['illuminance', 'blinds','lamps','rh', 'co2', 'temp']]

#split design matrix (25% of the design matrix used for test)
X_train, X_test, y_train, y_test = train_test_split(X, Y)

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
trainable_fm = QuantumCircuit(NUM_QBIT)

training_params = ParameterVector("θ_par", NUM_QBIT)

# Create an initial rotation layer of trainable parameters
for i, param in enumerate(training_params):
    trainable_fm.ry(param, trainable_fm.qubits[i])

zzfm = ZZFeatureMap(feature_dimension=NUM_QBIT)

fm = trainable_fm.compose(zzfm)

#implementation of fidelity using "ComputeUncompute"
n_shots = 1024
my_sampler = AerSampler(run_options={"shots":n_shots,"seed":123})
my_state_fidelity = ComputeUncompute(sampler=my_sampler)
print(f'*** Number of shots used in Aer {n_shots}')

#show feature map
print(f'*** TRAINABLE FEATURE MAP used in QSVC')
print(fm.draw())


#define callback
my_callback = QKCallback()

#define the trainable kernel
q_kernel = TrainableFidelityQuantumKernel(fidelity=my_state_fidelity, feature_map=fm, training_parameters=training_params)

#define updater, loss and inizial param
spsa_opt = SPSA(maxiter=max_iter, learning_rate=0.03, perturbation=0.01, termination_checker=my_callback.callback)

loss_func = SVCLoss(C=1.0)
init_point=[np.pi/2 for _ in range(NUM_QBIT)]

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

# Using dataset in datafile: data/env.sel3.minmax.csv
# Fraction rate used for this run: 5.0%
# Max number of iteration used in kernel optimization: 2
# Shape of dataset: (2865, 7)
# Training shape dataset (107, 6)
# Label for traing (107,)
# Test shape dataset (36, 6)
# Label for test (36,)
# *** TRAINABLE FEATURE MAP used in QSVC
#      ┌──────────────┐┌──────────────────────────────────────────────┐
# q_0: ┤ Ry(θ_par[0]) ├┤0                                             ├
#      ├──────────────┤│                                              │
# q_1: ┤ Ry(θ_par[1]) ├┤1                                             ├
#      ├──────────────┤│                                              │
# q_2: ┤ Ry(θ_par[2]) ├┤2                                             ├
#      ├──────────────┤│  ZZFeatureMap(x[0],x[1],x[2],x[3],x[4],x[5]) │
# q_3: ┤ Ry(θ_par[3]) ├┤3                                             ├
#      ├──────────────┤│                                              │
# q_4: ┤ Ry(θ_par[4]) ├┤4                                             ├
#      ├──────────────┤│                                              │
# q_5: ┤ Ry(θ_par[5]) ├┤5                                             ├
#      └──────────────┘└──────────────────────────────────────────────┘
# **********************
# Print callback. Iteration 1
# Number of function evaluations: 2
# The paramenters: [1.6506896  1.49090305 1.49090305 1.49090305 1.6506896  1.6506896 ]
# The function value: 48.74391002602404
# The stepsize: 0.19569775539910736
# Whether the step was accepted: True
# **********************
# **********************
# Print callback. Iteration 2
# Number of function evaluations: 4
# The paramenters: [1.75728966 1.384303   1.384303   1.59750311 1.75728966 1.75728966]
# The function value: 48.729226389295036
# The stepsize: 0.26111574693138345
# Whether the step was accepted: True
# **********************
# *******SCORE: 0.6388888888888888
# Time kernel training: 81.507483959198 seconds.
# Time training SVM: 10.034525394439697 seconds.
# Total jobs time: 98.15184950828552 seconds.