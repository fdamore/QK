import sys
import os
import time
from sklearn.metrics import accuracy_score

#define the trainable kernel
from qiskit_algorithms.optimizers import SPSA
from qiskit_machine_learning.utils.loss_functions import SVCLoss 
from qiskit_machine_learning.algorithms.classifiers import QSVC
from qiskit_machine_learning.kernels.algorithms.quantum_kernel_trainer import QuantumKernelTrainer
from qiskit_machine_learning.datasets import ad_hoc_data
from qiskit_algorithms.optimizers import SPSA
from qiskit_machine_learning.utils.loss_functions import SVCLoss
from qiskit_algorithms.utils import algorithm_globals

import numpy as np

#define working directory and package for QK
current_wd = os.getcwd()
sys.path.append(current_wd)

from pqk import Circuits
from pqk.QKCallback import QKCallback
from pqk.QMeasures import QMeasures
from pqk.TrainableKernelFeatureMap import TrainableKernelFeatureMap
from pqk.CKernels import CKernels




#set the seed(s)
np.random.seed(123)
algorithm_globals.random_seed = 123

adhoc_dimension = 3
X_train, y_train, X_test, y_test, adhoc_total = ad_hoc_data(
    training_size=30,
    test_size=5,
    n=adhoc_dimension,
    gap=0.3,
    plot_data=False,
    one_hot=False,
    include_sample_total=True,
)

max_iter = 10
NUM_QBIT = X_train.shape[1] # nqubit = nuber of dimension (feature)

#check the shape of test> and training dataset
print(f'Max number of iteration used in kernel optimization: {max_iter}')
print(f'Training shape dataset {X_train.shape}')
print(f'Label for traing {y_train.shape}')
print(f'Test shape dataset {X_test.shape}')
print(f'Label for test {y_test.shape}')
print(f'NUM_QUBIT {NUM_QBIT}')

encoding_circuit = Circuits.zzfeaturemap(n_wire=NUM_QBIT)
trainable_circuit = Circuits.x_encoded(n_wire=NUM_QBIT)
encoding_circuit.barrier()
fm = encoding_circuit.compose(trainable_circuit)
training_params = trainable_circuit.parameters

fm.draw()

#define callback
my_callback = QKCallback()


my_obs = ['ZII', 'IZI','IIZ']
nshots = 100 #paramenter using primitive estimator
measure_fn = QMeasures.StateVectorEstimator

#using kernel feature map (change the q_kernel in order to modiy the trainable kernel)
q_kernel = TrainableKernelFeatureMap(feature_map=fm, training_parameters=training_params)
q_kernel.configure(obs=my_obs, nshots=nshots, q_measure=QMeasures.StateVectorEstimator, c_kernel=CKernels.rbf)

#q_kernel = TrainableFidelityQuantumKernel(feature_map=fm, training_parameters=training_params)

#define updater, loss and inizial param
spsa_opt = SPSA(maxiter=max_iter, learning_rate=0.03, perturbation=0.01, termination_checker=my_callback.callback)
loss_func = SVCLoss(C=1.0)
#init_point=[np.pi / 2 for _ in range(NUM_QBIT)] 
init_point = np.random.uniform(size=NUM_QBIT)

print(f'The numbers of shots (if applicable) for (qiskit) primitive estimator: {nshots}')
print(f'Initial point: {init_point}')
#print this info
print(f'The QMeasure function used: {q_kernel.q_measure.__name__}')
print(f'The classical kernel used: {q_kernel.kernel.__name__}')
print(f'The observables we use: {my_obs}')

training_kernel_start = time.time()

qk_trainer = QuantumKernelTrainer(quantum_kernel=q_kernel, loss=loss_func, initial_point= init_point, optimizer=spsa_opt)
qkt_results = qk_trainer.fit(X_train, y_train)
optimized_kernel = qkt_results.quantum_kernel

training_kernel_end = time.time()

print(f'Time kernel training: {training_kernel_end - training_kernel_start} seconds.')

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
print(f'Time training SVM: {training_svm_end - training_svm_start} seconds.')
print(f'Total jobs time: {jobs_final_time - training_kernel_start} seconds.')
