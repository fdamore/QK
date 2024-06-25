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
from qiskit.circuit.library import ZZFeatureMap

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

#show feature map
print(f'*** TRAINABLE FEATURE MAP used in QSVC')
print(fm.draw())

#define callback
my_callback = QKCallback()

#define the trainable kernel
q_kernel = TrainableFidelityStatevectorKernel(feature_map=fm, training_parameters=training_params)

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

# #RUN USING MINMAX
# Fraction rate used for this run: 100%
# Max number of iteration used in kernel optimization: 20
# Shape of dataset: (2865, 7)
# Training shape dataset (2148, 6)
# Label for traing (2148,)
# Test shape dataset (717, 6)
# Label for test (717,)
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
# The paramenters: [1.08017212 1.08017212 1.08017212 2.06142053 2.06142053 2.06142053]
# The function value: 689.2994980301917
# The stepsize: 1.2017789650538828
# Whether the step was accepted: True
# **********************
# **********************
# Print callback. Iteration 2
# Number of function evaluations: 4
# The paramenters: [1.77762821 0.38271603 1.77762821 1.36396445 2.75887662 1.36396445]
# The function value: 706.4559501951408
# The stepsize: 1.7084115359876109
# Whether the step was accepted: True
# **********************
# **********************
# Print callback. Iteration 3
# Number of function evaluations: 6
# The paramenters: [ 0.42506146 -0.96985072  3.13019496  0.01139769  1.40630987  0.01139769]
# The function value: 725.2047643320451
# The stepsize: 3.313098386624061
# Whether the step was accepted: True
# **********************
# **********************
# Print callback. Iteration 4
# Number of function evaluations: 8
# The paramenters: [ 0.85752824 -1.40231751  2.69772817  0.44386448  0.97384308  0.44386448]
# The function value: 713.1513325214344
# The stepsize: 1.0593229599579979
# Whether the step was accepted: True
# **********************
# **********************
# Print callback. Iteration 5
# Number of function evaluations: 10
# The paramenters: [-0.54063151 -0.00415776  1.29956842 -0.95429527  2.37200283 -0.95429527]
# The function value: 712.2519704343217
# The stepsize: 3.4247779667679197
# Whether the step was accepted: True
# **********************
# **********************
# Print callback. Iteration 6
# Number of function evaluations: 12
# The paramenters: [ 0.16341949 -0.70820876  0.59551742 -1.65834627  1.66795183 -1.65834627]
# The function value: 722.2170410265662
# The stepsize: 1.724565700956923
# Whether the step was accepted: True
# **********************
# **********************
# Print callback. Iteration 7
# Number of function evaluations: 14
# The paramenters: [ 3.65366556 -4.19845483  4.08576349 -5.14859234 -1.82229423 -5.14859234]
# The function value: 729.1765639391663
# The stepsize: 8.549321947354297
# Whether the step was accepted: True
# **********************
# **********************
# Print callback. Iteration 8
# Number of function evaluations: 16
# The paramenters: [ 4.37551096 -3.47660943  3.3639181  -5.87043774 -2.54413963 -5.87043774]
# The function value: 718.3925449185153
# The stepsize: 1.768152895604204
# Whether the step was accepted: True
# **********************
# **********************
# Print callback. Iteration 9
# Number of function evaluations: 18
# The paramenters: [ 4.49235698 -3.59345545  3.24707208 -5.98728376 -2.66098565 -5.75359171]
# The function value: 723.7497204036288
# The stepsize: 0.2862131299056279
# Whether the step was accepted: True
# **********************
# **********************
# Print callback. Iteration 10
# Number of function evaluations: 20
# The paramenters: [ 4.359265   -3.46036347  3.38016406 -5.85419177 -2.79407764 -5.8866837 ]
# The function value: 722.9879295309056
# The stepsize: 0.32600744647274954
# Whether the step was accepted: True
# **********************
# **********************
# Print callback. Iteration 11
# Number of function evaluations: 22
# The paramenters: [ 4.51213872 -3.6132372   3.53303778 -5.70131805 -2.64120391 -5.73380997]
# The function value: 722.3064787526321
# The stepsize: 0.37446262404606745
# Whether the step was accepted: True
# **********************
# **********************
# Print callback. Iteration 12
# Number of function evaluations: 24
# The paramenters: [ 5.21171404 -2.91366188  4.2326131  -6.40089336 -1.94162859 -5.03423465]
# The function value: 717.9223840395753
# The stepsize: 1.7136025638588246
# Whether the step was accepted: True
# **********************
# **********************
# Print callback. Iteration 13
# Number of function evaluations: 26
# The paramenters: [ 4.47938011 -2.18132795  4.96494703 -7.13322729 -1.20929467 -4.30190073]
# The function value: 721.3837799065784
# The stepsize: 1.793844442250169
# Whether the step was accepted: True
# **********************
# **********************
# Print callback. Iteration 14
# Number of function evaluations: 28
# The paramenters: [ 3.29702526 -0.9989731   6.14730188 -8.31558214 -2.39164951 -5.48425558]
# The function value: 688.4770636783329
# The stepsize: 2.896166075119282
# Whether the step was accepted: True
# **********************
# **********************
# Print callback. Iteration 15
# Number of function evaluations: 30
# The paramenters: [ 3.76290533 -1.46485317  6.61318194 -8.78146221 -1.92576945 -5.95013564]
# The function value: 734.593196429223
# The stepsize: 1.1411684454603883
# Whether the step was accepted: True
# **********************
# **********************
# Print callback. Iteration 16
# Number of function evaluations: 32
# The paramenters: [  1.87344394  -3.35431456   8.50264333 -10.6709236   -3.81523084
#   -4.06067425]
# The function value: 742.1079524849304
# The stepsize: 4.628216295317132
# Whether the step was accepted: True
# **********************
# **********************
# Print callback. Iteration 17
# Number of function evaluations: 34
# The paramenters: [  0.43167103  -1.91254165   7.06087042 -12.11269651  -5.25700375
#   -5.50244716]
# The function value: 721.0031938402448
# The stepsize: 3.531607955287895
# Whether the step was accepted: True
# **********************
# **********************
# Print callback. Iteration 18
# Number of function evaluations: 36
# The paramenters: [  0.1507024   -1.63157302   7.34183906 -12.39366514  -5.53797238
#   -5.22147853]
# The function value: 713.3512555697642
# The stepsize: 0.6882297793579647
# Whether the step was accepted: True
# **********************
# **********************
# Print callback. Iteration 19
# Number of function evaluations: 38
# The paramenters: [  1.25414517  -0.52813025   6.23839628 -13.49710791  -4.43452961
#   -4.11803576]
# The function value: 710.8564591944082
# The stepsize: 2.702871755180082
# Whether the step was accepted: True
# **********************
# **********************
# Print callback. Iteration 20
# Number of function evaluations: 40
# The paramenters: [ -0.21291515  -1.99519057   4.77133596 -12.03004759  -2.96746928
#   -5.58509608]
# The function value: 729.5171586652124
# The stepsize: 3.593549218548337
# Whether the step was accepted: True
# **********************
# *******SCORE: 0.8800557880055788
# Time kernel training: 821.4848947525024 seconds.
# Time training SVM: 20.039472341537476 seconds.
# Total jobs time: 854.3936252593994 seconds.



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