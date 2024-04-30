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
#LAST RESULT:
##USING 1000 data point for training
##*******SCORE: 0.8222384784198976
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

#get time
t_start = time.time()

#try SVM using RBF kernel
svm = SVC(kernel='rbf').fit(X_train_np[:K], y_train_np[:K]);

#get time training
t_training = time.time()

#result...
predictions = svm.predict(X_test_np)
score = accuracy_score(predictions, y_test)

#final time (trainign + predict)
t_final = time.time()


print(f'*******SCORE: {score}')
print(f'Time training: {t_training - t_start} seconds. Final time {t_final - t_start} seconds')



