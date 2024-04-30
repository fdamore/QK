#Create a cotainer
from qproc import CircuitContainer
from qproc import Circuits
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from qproc import kernel_matrix
import numpy as np


#set the seed
np.random.seed(123)


my_obs = ['ZIIIII', 'IZIIII','IIZIII', 'IIIZII','IIIIZI','IIIIIZ']

c = CircuitContainer(qtemplate=Circuits.circularEnt, nwire=6, obs=my_obs)


#load dataset with panda
#data are scaled outside the notebook
env = pd.read_csv('data/env.sel2.scaled.csv')  


#DEFINE design matrix
Y = env['occupancy']
#X = env[['illuminance', 'blinds','lamps','co', 'rh', 'co2', 'temp']]
X = env[['illuminance', 'blinds','lamps','rh', 'co2', 'temp']]

#split design matrix (25% of the design matrix used for test)
X_train, X_test, y_train, y_test = train_test_split(X, Y)
#WARNING: convert data to numpy. Quantum stuff (Qiskit) do not like PANDAS
X_train_np = X_train.to_numpy()
y_train_np = y_train.to_numpy()

X_test_np = X_test.to_numpy()
y_test_np = y_test.to_numpy()

#check the shape of test and training dataset
print(f'Training shape dataset {X_train_np.shape}')
print(f'Label for traing {y_train_np.shape}')

print(f'Test shape dataset {X_test_np.shape}')
print(f'Label for test {y_test_np.shape}')

#reduce data for trainingdefine data for training
K = 1000
print(f'USING {K} data point for training')

svm_quantum = SVC(kernel=kernel_matrix).fit(X_train_np[:K], y_train_np[:K]);
#result...
predictions = svm_quantum.predict(X_test_np)
score = accuracy_score(predictions, y_test)
print(f'*******SCORE: {score}')

#LAST RESULT:
##USING 1000 data point for training
##*******SCORE: 0.8222384784198976

#CIRCUIT
#*** Created quantum template for feature map using 6 qubit ***
#     ┌───┐┌───────────┐                         ┌───┐ ░ ┌───┐
#q_0: ┤ H ├┤ Rz(phi_0) ├──■──────────────────────┤ X ├─░─┤ H ├
#     ├───┤├───────────┤┌─┴─┐                    └─┬─┘ ░ ├───┤
#q_1: ┤ H ├┤ Rz(phi_1) ├┤ X ├──■───────────────────┼───░─┤ H ├
#     ├───┤├───────────┤└───┘┌─┴─┐                 │   ░ ├───┤
#q_2: ┤ H ├┤ Rz(phi_2) ├─────┤ X ├──■──────────────┼───░─┤ H ├
#     ├───┤├───────────┤     └───┘┌─┴─┐            │   ░ ├───┤
#q_3: ┤ H ├┤ Rz(phi_3) ├──────────┤ X ├──■─────────┼───░─┤ H ├
#     ├───┤├───────────┤          └───┘┌─┴─┐       │   ░ ├───┤
#q_4: ┤ H ├┤ Rz(phi_4) ├───────────────┤ X ├──■────┼───░─┤ H ├
#     ├───┤├───────────┤               └───┘┌─┴─┐  │   ░ ├───┤
#q_5: ┤ H ├┤ Rz(phi_5) ├────────────────────┤ X ├──■───░─┤ H ├
#     └───┘└───────────┘                    └───┘      ░ └───┘
#*** Required observables: ['ZIIIII', 'IZIIII', 'IIZIII', 'IIIZII', 'IIIIZI', 'IIIIIZ']