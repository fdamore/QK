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
from qiskit_machine_learning.kernels import TrainableFidelityStatevectorKernel
from qiskit_algorithms.utils import algorithm_globals

import numpy as np

#define working directory and package for QK
current_wd = os.getcwd()
sys.path.append(current_wd)

from qke.QKCallback import QKCallback


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
env_slice = env.sample(frac=f_rate) #slices the origin dataset

Y = env_slice['occupancy']
X = env_slice[['illuminance', 'blinds','lamps','rh', 'co2', 'temp']]

#split design matrix (25% of the design matrix used for test)
X_train, X_test, y_train, y_test = train_test_split(X, Y)


#check the shape of test and training dataset
print(f'Using dataset in datafile: {data_file_csv}')
print(f'Fraction rate used for this run: {f_rate * 100}%')
print(f'Shape of dataset: {env.shape}')
print(f'Training shape dataset {X_train.shape}')
print(f'Label for traing {y_train.shape}')

print(f'Test shape dataset {X_test.shape}')
print(f'Label for test {y_test.shape}')

#build a feature map 
NUM_QBIT = X_train.shape[1]
fm = QuantumCircuit(NUM_QBIT)
input_params = ParameterVector("x_par", NUM_QBIT)
training_params = ParameterVector("θ_par", NUM_QBIT)

# Create an initial rotation layer of trainable parameters
for i, param in enumerate(training_params):
    fm.ry(param, fm.qubits[i])

# Create a rotation layer of input parameters
for i, param in enumerate(input_params):
    fm.rz(param, fm.qubits[i])

#show feature map
print(f'*** TRAINABLE FEATURE MAP used in QSVC')
print(fm.draw())

#define callback
my_callback = QKCallback()

#define the trainable kernel
q_kernel = TrainableFidelityStatevectorKernel(feature_map=fm, training_parameters=training_params)

#define updater, loss and inizial param
spsa_opt = SPSA(maxiter=10, learning_rate=0.03, perturbation=0.01, termination_checker=my_callback.callback)
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

# #RUN USING MINMAX
# Using dataset in datafile: data/env.sel3.minmax.csv
# Fraction rate used for this run: 5.0%
# Shape of dataset: (2865, 7)
# Training shape dataset (107, 6)
# Label for traing (107,)
# Test shape dataset (36, 6)
# Label for test (36,)
# *** TRAINABLE FEATURE MAP used in QSVC
#      ┌──────────────┐┌──────────────┐
# q_0: ┤ Ry(θ_par[0]) ├┤ Rz(x_par[0]) ├
#      ├──────────────┤├──────────────┤
# q_1: ┤ Ry(θ_par[1]) ├┤ Rz(x_par[1]) ├
#      ├──────────────┤├──────────────┤
# q_2: ┤ Ry(θ_par[2]) ├┤ Rz(x_par[2]) ├
#      ├──────────────┤├──────────────┤
# q_3: ┤ Ry(θ_par[3]) ├┤ Rz(x_par[3]) ├
#      ├──────────────┤├──────────────┤
# q_4: ┤ Ry(θ_par[4]) ├┤ Rz(x_par[4]) ├
#      ├──────────────┤├──────────────┤
# q_5: ┤ Ry(θ_par[5]) ├┤ Rz(x_par[5]) ├
#      └──────────────┘└──────────────┘
# *******SCORE: 0.7222222222222222
# Time kernel training: 19.50859236717224 seconds.
# Time training SVM: 0.07846307754516602 seconds.
# Total jobs time: 19.65761113166809 seconds.

# RUN
# Fraction rate used for this run: 5.0%
# Shape of dataset: (2865, 7)
# Training shape dataset (107, 6)
# Label for traing (107,)
# Test shape dataset (36, 6)
# Label for test (36,)
# *** TRAINABLE FEATURE MAP used in QSVC
#      ┌──────────────┐┌──────────────┐
# q_0: ┤ Ry(θ_par[0]) ├┤ Rz(x_par[0]) ├
#      ├──────────────┤├──────────────┤
# q_1: ┤ Ry(θ_par[1]) ├┤ Rz(x_par[1]) ├
#      ├──────────────┤├──────────────┤
# q_2: ┤ Ry(θ_par[2]) ├┤ Rz(x_par[2]) ├
#      ├──────────────┤├──────────────┤
# q_3: ┤ Ry(θ_par[3]) ├┤ Rz(x_par[3]) ├
#      ├──────────────┤├──────────────┤
# q_4: ┤ Ry(θ_par[4]) ├┤ Rz(x_par[4]) ├
#      ├──────────────┤├──────────────┤
# q_5: ┤ Ry(θ_par[5]) ├┤ Rz(x_par[5]) ├
#      └──────────────┘└──────────────┘
# *******SCORE: 0.6666666666666666
# Time kernel training: 19.962605953216553 seconds.
# Time training SVM: 0.07925987243652344 seconds.
# Total jobs time: 20.113325119018555 seconds.