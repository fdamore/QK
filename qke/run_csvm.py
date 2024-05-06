#Create a cotainer
import pandas as pd
import time
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


#set the seed
np.random.seed(123)


#load dataset with panda
#data are scaled outside the notebook
env = pd.read_csv('data/env.sel3.scaled.csv')  


#DEFINE design matrix
Y = env['occupancy']
X = env[['illuminance', 'blinds','lamps','rh', 'co2', 'temp']]

#split design matrix (25% of the design matrix used for test)
X_train, X_test, y_train, y_test = train_test_split(X, Y)
#WARNING: convert data to numpy. Quantum stuff (Qiskit) do not like PANDAS
X_train_np = X_train.to_numpy()
y_train_np = y_train.to_numpy()
#LAST RESULT:
##USING 1000 data point for training
##*******SCORE: 0.8222384784198976
X_test_np = X_test.to_numpy()
y_test_np = y_test.to_numpy()

#check the shape of test and training dataset
print(f'Shape of dataset: {env.shape}')
print(f'Training shape dataset {X_train_np.shape}')
print(f'Label for traing {y_train_np.shape}')

print(f'Test shape dataset {X_test_np.shape}')
print(f'Label for test {y_test_np.shape}')


#get time
t_start = time.time()

kernel_type = 'linear'

#try SVM using RBF kernel
svm = SVC(kernel=kernel_type).fit(X_train_np, y_train_np);

#get time training
t_training = time.time()

#result...
predictions = svm.predict(X_test_np)
score = accuracy_score(predictions, y_test)

#final time (trainign + predict)
t_final = time.time()

print(f'Using kernel type: {kernel_type}')
print(f'*******SCORE: {score}')
print(f'Time training: {t_training - t_start} seconds. Final time {t_final - t_start} seconds')

#LAST RESULT: SEL3 - NO DUPLICATED
# Shape of dataset: (2865, 7)
# Training shape dataset (2148, 6)
# Label for traing (2148,)
# Test shape dataset (717, 6)
# Label for test (717,)
# USING 2148 data point for training
# Using kernel type: linear
# *******SCORE: 0.7796373779637378
# Time training: 0.06213259696960449 seconds. Final time 0.06864118576049805 seconds

#RUN WITH KERNEL = LINEAR
# Shape of dataset: (3233, 7)
# Training shape dataset (2424, 6)
# Label for traing (2424,)
# Test shape dataset (809, 6)
# Label for test (809,)
# USING 2424 data point for training
# Using kernel type: linear
# *******SCORE: 0.7391841779975278
# Time training: 0.09719133377075195 seconds. Final time 0.10743522644042969 seconds

#RUN WITH KERNEL = RBF
# Shape of dataset: (3233, 7)
# Training shape dataset (2424, 6)
# Label for traing (2424,)
# Test shape dataset (809, 6)
# Label for test (809,)
# USING 2424 data point for training
# Using kernel type: rbf
# *******SCORE: 0.765142150803461
# Time training: 0.05849957466125488 seconds. Final time 0.07966256141662598 seconds