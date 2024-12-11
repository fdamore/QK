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
encoding_circuit = Circuits.xyz_encoded(n_wire=NUM_QBIT, full_ent=False)
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



#RUN ON SERVER 2024-12-11 on server callback TR_20241210221314.json
# Using dataset in datafile: data/env.sel3.sk_sc.csv
# Fraction rate used for this run: 100%
# Max number of iteration used in kernel optimization: 20
# Shape of dataset: (2865, 7)
# Training shape dataset (2148, 6)
# Label for traing (2148,)
# Test shape dataset (717, 6)
# Label for test (717,)
# *** TRAINABLE FEATURE MAP used in QSVC
#      ┌───────────┐┌───────────┐┌───────────┐                         ┌───┐ ░ »
# q_0: ┤ Rx(phi_0) ├┤ Ry(phi_0) ├┤ Rz(phi_0) ├──■──────────────────────┤ X ├─░─»
#      ├───────────┤├───────────┤├───────────┤┌─┴─┐                    └─┬─┘ ░ »
# q_1: ┤ Rx(phi_1) ├┤ Ry(phi_1) ├┤ Rz(phi_1) ├┤ X ├──■───────────────────┼───░─»
#      ├───────────┤├───────────┤├───────────┤└───┘┌─┴─┐                 │   ░ »
# q_2: ┤ Rx(phi_2) ├┤ Ry(phi_2) ├┤ Rz(phi_2) ├─────┤ X ├──■──────────────┼───░─»
#      ├───────────┤├───────────┤├───────────┤     └───┘┌─┴─┐            │   ░ »
# q_3: ┤ Rx(phi_3) ├┤ Ry(phi_3) ├┤ Rz(phi_3) ├──────────┤ X ├──■─────────┼───░─»
#      ├───────────┤├───────────┤├───────────┤          └───┘┌─┴─┐       │   ░ »
# q_4: ┤ Rx(phi_4) ├┤ Ry(phi_4) ├┤ Rz(phi_4) ├───────────────┤ X ├──■────┼───░─»
#      ├───────────┤├───────────┤├───────────┤               └───┘┌─┴─┐  │   ░ »
# q_5: ┤ Rx(phi_5) ├┤ Ry(phi_5) ├┤ Rz(phi_5) ├────────────────────┤ X ├──■───░─»
#      └───────────┘└───────────┘└───────────┘                    └───┘      ░ »
# «     ┌──────────────┐┌───────────────┐┌───────────────┐
# «q_0: ┤ Rz(trbeta_0) ├┤ Ry(trgamma_0) ├┤ Rz(trdelta_0) ├
# «     ├──────────────┤├───────────────┤├───────────────┤
# «q_1: ┤ Rz(trbeta_1) ├┤ Ry(trgamma_1) ├┤ Rz(trdelta_1) ├
# «     ├──────────────┤├───────────────┤├───────────────┤
# «q_2: ┤ Rz(trbeta_2) ├┤ Ry(trgamma_2) ├┤ Rz(trdelta_2) ├
# «     ├──────────────┤├───────────────┤├───────────────┤
# «q_3: ┤ Rz(trbeta_3) ├┤ Ry(trgamma_3) ├┤ Rz(trdelta_3) ├
# «     ├──────────────┤├───────────────┤├───────────────┤
# «q_4: ┤ Rz(trbeta_4) ├┤ Ry(trgamma_4) ├┤ Rz(trdelta_4) ├
# «     ├──────────────┤├───────────────┤├───────────────┤
# «q_5: ┤ Rz(trbeta_5) ├┤ Ry(trgamma_5) ├┤ Rz(trdelta_5) ├
# «     └──────────────┘└───────────────┘└───────────────┘
# Number of trainable paramenters: 18
# The QMeasure function used: StateVectorEstimator
# The classical kernel used: rbf
# The observables we use: ['XIIIII', 'IXIIII', 'IIXIII', 'IIIXII', 'IIIIXI', 'IIIIIX', 'YIIIII', 'IYIIII', 'IIYIII', 'IIIYII', 'IIIIYI', 'IIIIIY', 'ZIIIII', 'IZIIII', 'IIZIII', 'IIIZII', 'IIIIZI', 'IIIIIZ']
# Initial point: [0.69646919 0.28613933 0.22685145 0.55131477 0.71946897 0.42310646
#  0.9807642  0.68482974 0.4809319  0.39211752 0.34317802 0.72904971
#  0.43857224 0.0596779  0.39804426 0.73799541 0.18249173 0.17545176]
# **********************
# Print callback. Iteration 1
# Number of function evaluations: 2
# The paramenters: [0.69646919 0.28613933 0.22685145 0.55131477 0.71946897 0.42310646
#  0.9807642  0.68482974 0.4809319  0.39211752 0.34317802 0.72904971
#  0.43857224 0.0596779  0.39804426 0.73799541 0.18249173 0.17545176]
# The function value: 1019.2621378751476
# The stepsize: 0.0
# Whether the step was accepted: True
# **********************
# **********************
# Print callback. Iteration 2
# Number of function evaluations: 4
# The paramenters: [0.69646919 0.28613933 0.22685145 0.55131477 0.71946897 0.42310646
#  0.9807642  0.68482974 0.4809319  0.39211752 0.34317802 0.72904971
#  0.43857224 0.0596779  0.39804426 0.73799541 0.18249173 0.17545176]
# The function value: 1019.2621378751478
# The stepsize: 7.234986049614836e-12
# Whether the step was accepted: True
# **********************
# **********************
# Print callback. Iteration 3
# Number of function evaluations: 6
# The paramenters: [0.69646919 0.28613933 0.22685145 0.55131477 0.71946897 0.42310646
#  0.9807642  0.68482974 0.4809319  0.39211752 0.34317802 0.72904971
#  0.43857224 0.0596779  0.39804426 0.73799541 0.18249173 0.17545176]
# The function value: 1019.2621378751497
# The stepsize: 7.234986049614836e-12
# Whether the step was accepted: True
# **********************
# **********************
# Print callback. Iteration 4
# Number of function evaluations: 8
# The paramenters: [0.69646919 0.28613933 0.22685145 0.55131477 0.71946897 0.42310646
#  0.9807642  0.68482974 0.4809319  0.39211752 0.34317802 0.72904971
#  0.43857224 0.0596779  0.39804426 0.73799541 0.18249173 0.17545176]
# The function value: 1019.262137875148
# The stepsize: 5.787988839691868e-12
# Whether the step was accepted: True
# **********************
# **********************
# Print callback. Iteration 5
# Number of function evaluations: 10
# The paramenters: [0.69646919 0.28613933 0.22685145 0.55131477 0.71946897 0.42310646
#  0.9807642  0.68482974 0.4809319  0.39211752 0.34317802 0.72904971
#  0.43857224 0.0596779  0.39804426 0.73799541 0.18249173 0.17545176]
# The function value: 1019.2621378751493
# The stepsize: 1.446997209922967e-12
# Whether the step was accepted: True
# **********************
# **********************
# Print callback. Iteration 6
# Number of function evaluations: 12
# The paramenters: [0.69646919 0.28613933 0.22685145 0.55131477 0.71946897 0.42310646
#  0.9807642  0.68482974 0.4809319  0.39211752 0.34317802 0.72904971
#  0.43857224 0.0596779  0.39804426 0.73799541 0.18249173 0.17545176]
# The function value: 1019.2621378751497
# The stepsize: 5.787988839691868e-12
# Whether the step was accepted: True
# **********************
# **********************
# Print callback. Iteration 7
# Number of function evaluations: 14
# The paramenters: [0.69646919 0.28613933 0.22685145 0.55131477 0.71946897 0.42310646
#  0.9807642  0.68482974 0.4809319  0.39211752 0.34317802 0.72904971
#  0.43857224 0.0596779  0.39804426 0.73799541 0.18249173 0.17545176]
# The function value: 1019.2621378751492
# The stepsize: 3.617493024807418e-12
# Whether the step was accepted: True
# **********************
# **********************
# Print callback. Iteration 8
# Number of function evaluations: 16
# The paramenters: [0.69646919 0.28613933 0.22685145 0.55131477 0.71946897 0.42310646
#  0.9807642  0.68482974 0.4809319  0.39211752 0.34317802 0.72904971
#  0.43857224 0.0596779  0.39804426 0.73799541 0.18249173 0.17545176]
# The function value: 1019.2621378751489
# The stepsize: 2.170495814884451e-12
# Whether the step was accepted: True
# **********************
# **********************
# Print callback. Iteration 9
# Number of function evaluations: 18
# The paramenters: [0.69646919 0.28613933 0.22685145 0.55131477 0.71946897 0.42310646
#  0.9807642  0.68482974 0.4809319  0.39211752 0.34317802 0.72904971
#  0.43857224 0.0596779  0.39804426 0.73799541 0.18249173 0.17545176]
# The function value: 1019.2621378751488
# The stepsize: 4.340991629768902e-12
# Whether the step was accepted: True
# **********************
# **********************
# Print callback. Iteration 10
# Number of function evaluations: 20
# The paramenters: [0.69646919 0.28613933 0.22685145 0.55131477 0.71946897 0.42310646
#  0.9807642  0.68482974 0.4809319  0.39211752 0.34317802 0.72904971
#  0.43857224 0.0596779  0.39804426 0.73799541 0.18249173 0.17545176]
# The function value: 1019.2621378751493
# The stepsize: 5.787988839691868e-12
# Whether the step was accepted: True
# **********************
# **********************
# Print callback. Iteration 11
# Number of function evaluations: 22
# The paramenters: [0.69646919 0.28613933 0.22685145 0.55131477 0.71946897 0.42310646
#  0.9807642  0.68482974 0.4809319  0.39211752 0.34317802 0.72904971
#  0.43857224 0.0596779  0.39804426 0.73799541 0.18249173 0.17545176]
# The function value: 1019.2621378751489
# The stepsize: 1.2299476284345221e-11
# Whether the step was accepted: True
# **********************
# **********************
# Print callback. Iteration 12
# Number of function evaluations: 24
# The paramenters: [0.69646919 0.28613933 0.22685145 0.55131477 0.71946897 0.42310646
#  0.9807642  0.68482974 0.4809319  0.39211752 0.34317802 0.72904971
#  0.43857224 0.0596779  0.39804426 0.73799541 0.18249173 0.17545176]
# The function value: 1019.2621378751478
# The stepsize: 1.2299476284345221e-11
# Whether the step was accepted: True
# **********************
# **********************
# Print callback. Iteration 13
# Number of function evaluations: 26
# The paramenters: [0.69646919 0.28613933 0.22685145 0.55131477 0.71946897 0.42310646
#  0.9807642  0.68482974 0.4809319  0.39211752 0.34317802 0.72904971
#  0.43857224 0.0596779  0.39804426 0.73799541 0.18249173 0.17545176]
# The function value: 1019.2621378751484
# The stepsize: 1.1575977679383737e-11
# Whether the step was accepted: True
# **********************
# **********************
# Print callback. Iteration 14
# Number of function evaluations: 28
# The paramenters: [0.69646919 0.28613933 0.22685145 0.55131477 0.71946897 0.42310646
#  0.9807642  0.68482974 0.4809319  0.39211752 0.34317802 0.72904971
#  0.43857224 0.0596779  0.39804426 0.73799541 0.18249173 0.17545176]
# The function value: 1019.2621378751489
# The stepsize: 1.9534462333960055e-11
# Whether the step was accepted: True
# **********************
# **********************
# Print callback. Iteration 15
# Number of function evaluations: 30
# The paramenters: [0.69646919 0.28613933 0.22685145 0.55131477 0.71946897 0.42310646
#  0.9807642  0.68482974 0.4809319  0.39211752 0.34317802 0.72904971
#  0.43857224 0.0596779  0.39804426 0.73799541 0.18249173 0.17545176]
# The function value: 1019.2621378751488
# The stepsize: 1.3022974889306704e-11
# Whether the step was accepted: True
# **********************
# **********************
# Print callback. Iteration 16
# Number of function evaluations: 32
# The paramenters: [0.69646919 0.28613933 0.22685145 0.55131477 0.71946897 0.42310646
#  0.9807642  0.68482974 0.4809319  0.39211752 0.34317802 0.72904971
#  0.43857224 0.0596779  0.39804426 0.73799541 0.18249173 0.17545176]
# The function value: 1019.2621378751496
# The stepsize: 1.0128980469460769e-11
# Whether the step was accepted: True
# **********************
# **********************
# Print callback. Iteration 17
# Number of function evaluations: 34
# The paramenters: [0.69646919 0.28613933 0.22685145 0.55131477 0.71946897 0.42310646
#  0.9807642  0.68482974 0.4809319  0.39211752 0.34317802 0.72904971
#  0.43857224 0.0596779  0.39804426 0.73799541 0.18249173 0.17545176]
# The function value: 1019.2621378751487
# The stepsize: 6.511487444653352e-12
# Whether the step was accepted: True
# **********************
# **********************
# Print callback. Iteration 18
# Number of function evaluations: 36
# The paramenters: [0.69646919 0.28613933 0.22685145 0.55131477 0.71946897 0.42310646
#  0.9807642  0.68482974 0.4809319  0.39211752 0.34317802 0.72904971
#  0.43857224 0.0596779  0.39804426 0.73799541 0.18249173 0.17545176]
# The function value: 1019.2621378751497
# The stepsize: 1.3022974889306704e-11
# Whether the step was accepted: True
# **********************
# **********************
# Print callback. Iteration 19
# Number of function evaluations: 38
# The paramenters: [0.69646919 0.28613933 0.22685145 0.55131477 0.71946897 0.42310646
#  0.9807642  0.68482974 0.4809319  0.39211752 0.34317802 0.72904971
#  0.43857224 0.0596779  0.39804426 0.73799541 0.18249173 0.17545176]
# The function value: 1019.2621378751496
# The stepsize: 1.3746473494268187e-11
# Whether the step was accepted: True
# **********************
# **********************
# Print callback. Iteration 20
# Number of function evaluations: 40
# The paramenters: [0.69646919 0.28613933 0.22685145 0.55131477 0.71946897 0.42310646
#  0.9807642  0.68482974 0.4809319  0.39211752 0.34317802 0.72904971
#  0.43857224 0.0596779  0.39804426 0.73799541 0.18249173 0.17545176]
# The function value: 1019.2621378751487
# The stepsize: 3.617493024807418e-12
# Whether the step was accepted: True
# **********************
# *******SCORE: 0.7768479776847977
# Time kernel training: 40570.99979496002 seconds.
# Time training SVM: 853.9637115001678 seconds.
# Total jobs time: 42069.94274735451 seconds.
