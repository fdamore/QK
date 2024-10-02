#Create a cotainer
import pandas as pd
import time
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


#set the seed
np.random.seed(123)

#load dataset with panda
#data are scaled outside the notebook
#sclaled_data_file = 'data/env.sel3.scaled.csv'
f_rate = 0.1 #rate of data sampling. For grid search select 10% of the data
data_file_csv = 'data/env.sel3.scaled.csv' 
env = pd.read_csv(data_file_csv).sample(frac=f_rate, random_state=123)  


#DEFINE design matrix
Y = env['occupancy']
X = env[['illuminance', 'blinds','lamps','rh', 'co2', 'temp']]

#split design matrix (25% of the design matrix used for test)

#WARNING: convert data to numpy. Quantum stuff (Qiskit) do not like PANDAS
X_train_np = X.to_numpy()
y_train_np = Y .to_numpy()

print(X_train_np.var())

#check the shape of test and training dataset
print(f'Shape of dataset: {env.shape}')
print(f'Training shape dataset {X_train_np.shape}')
print(f'Label for traing {y_train_np.shape}')

# define grid search strategy
#Create a dictionary of possible parameters
params_grid = {'C': [0.006, 0.015, 0.03, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256, 512, 1024],
          'gamma': np.array([0.10, 0.15, 0.25, 0.5, 0.75, 1.0, 1.25,1.50, 1.75, 2.0, 2.5, 3.0, 4.0]),          
          'kernel':['rbf'] }

#Create the GridSearchCV object
grid_clf = GridSearchCV(SVC(), params_grid)

#get time
t_start = time.time()


#Fit the data with the best possible parameters
grid_clf = grid_clf.fit(X_train_np, y_train_np)

#get time training
t_training = time.time()

#Print the best estimator with it's parameters
print(f'Best paramenter: {grid_clf.best_params_}')


# Shape of dataset: (286, 7)
# Training shape dataset (286, 6)
# Label for traing (286,)
# Best paramenter: {'C': 8.0, 'gamma': 0.25, 'kernel': 'rbf'}

