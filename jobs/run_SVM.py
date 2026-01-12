#Create a cotainer
from matplotlib.pylab import f
import pandas as pd
import time
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

def get_qencoded_data(f_rate, source_file) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:

    #load dataset with panda
    data_file_csv = source_file
    env = pd.read_csv(data_file_csv).sample(frac=f_rate, random_state=123)      

    #DEFINE design matrix
    Y = env['label']
    X = env[env.columns.difference(['label'])]
    
    #WARNING: convert data to numpy. Quantum stuff (Qiskit) do not like PANDAS
    #X_train_np = X.to_numpy()
    #y_train_np = Y.to_numpy()

    return (X,Y,env)

#set the seed
np.random.seed(123)

#load dataset with panda
#data_file_csv = 'data/env.sel3.scaled.csv' 
#data_file_csv = 'data/env.sel3.sk_sc.csv'
data_file_csv = 'qfm/fm/qencoding/QC_3D_OBS_M2_ENT_TRUE.csv'

origin = False
if origin:
    #DEFINE design matrix
    env = pd.read_csv(data_file_csv)  
    Y = env['occupancy']
    X = env[['illuminance', 'blinds','lamps','rh', 'co2', 'temp']]
else:
    X, Y, env = get_qencoded_data(f_rate=1, source_file=data_file_csv)    

#split design matrix (25% of the design matrix used for test)
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=123, test_size=0.1)
#WARNING: convert data to numpy. Quantum stuff (Qiskit) do not like PANDAS
X_train_np = X_train.to_numpy()
y_train_np = y_train.to_numpy()
#LAST RESULT:
##USING 1000 data point for training
##*******SCORE: 0.8222384784198976
X_test_np = X_test.to_numpy()
y_test_np = y_test.to_numpy()

kernel_type = 'rbf'
C_value = 2.0
gamma_value = 4.0

#check the shape of test and training dataset
print(f'Shape of dataset: {env.shape}')
print(f'Training shape dataset {X_train_np.shape}')
print(f'Label for traing {y_train_np.shape}')
print(f'Test shape dataset {X_test_np.shape}')
print(f'Label for test {y_test_np.shape}')
print(f'Using C value: {C_value} and gamma value: {gamma_value}')


#get time
t_start = time.time()

#try SVM using RBF kernel
#svm = SVC(kernel=kernel_type).fit(X_train_np, y_train_np);
#use paramenter selected in grid search
svm = SVC(kernel=kernel_type, C=C_value, gamma=gamma_value).fit(X_train_np, y_train_np);


#get time training
t_training = time.time()

#result...
predictions = svm.predict(X_test_np)
score = accuracy_score(predictions, y_test)

#final time (trainign + predict)
t_final = time.time()

print(f'Using kernel type: {kernel_type}')
print(f'Using dataset in datafile: {data_file_csv}')
print(f'*******SCORE: {score}')
print(f'Time training: {t_training - t_start} seconds. Final time {t_final - t_start} seconds')