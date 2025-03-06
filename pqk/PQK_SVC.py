import numpy as np
import os
import time
import datetime

from qiskit import QuantumCircuit
from pqk.QMeasures import QMeasures
from pqk.CKernels import CKernels
from sklearn.svm import SVC

import pandas as pd


class PQK_SVC(SVC):


    #dict for latent space
    _fm_dict = {}   

      
    def __init__(self, C = 1, gamma = 0.5, fit_clear = True, obs = ['Z'], measure_fn = QMeasures.StateVectorEstimator, c_kernel = 'rbf',
                 _fm_dict = {}, *,  circuit : QuantumCircuit):
        
        
        super().__init__(C=C, gamma=gamma, kernel=self._kernel_matrix)               

        #clear the cache before fit
        self.fit_clear = fit_clear       

        #define PQK_SVM arameters 
        self.obs = obs       
        self.measure_fn = measure_fn
        self.c_kernel = c_kernel         
        self.circuit = circuit


        #set the initial enconding
        self._fm_dict = _fm_dict 

        print(len(self._fm_dict))      

    def _pqk_compute_kernel(self, x1, x2):
        '''
        Compute the required kernel
        '''
        if self.c_kernel not in ['rbf', 'linear'] or self.c_kernel == 'rbf':
            return CKernels.rbf(x1, x2, self.gamma)
        
        if self.c_kernel == 'linear':
            return CKernels.linear(x1, x2)    
                


    def metadata(self):
        print(f'*** CIRCUITS qubit ***')                
        print(self.circuit.draw())
        print(f'*** Required observables: {self.obs}')
        print(f'*** Measure procedure: {self.measure_fn.__name__}')
        print(f'*** CKernel function used: {self.c_kernel}')
        print(f'Param: {self.get_params}')
        print(f'Qubits: {self.circuit.num_qubits}')
        return ""

    #encode data in parameter    
    def _qEncoding(self, data): 
        #inplace = If False, a copy of the circuit with the bound parameters is returned       
        qc_assigned = self.circuit.assign_parameters(data, inplace = False)
        return qc_assigned;  

    #define quantum feature kernel using CircuitContainer    
    def _qfKernel(self, x1, x2):

        
        obs = self.obs

        #define the key
        k_x1 = str(x1) #get_key(x1) 
        k_x2 = str(x2) #get_key(x2)

        #check the k1 and get feature map
        x1_fm = None
        if k_x1 in self._fm_dict:
            x1_fm = self._fm_dict[k_x1]
        else:
            x1_qc = self._qEncoding(x1)
            x1_fm = self.measure_fn(x1_qc, observables=obs)
            
            self._fm_dict[k_x1] = x1_fm

        #check the k2 and get feature map
        x2_fm = None
        if k_x2 in self._fm_dict:
            x2_fm = self._fm_dict[k_x2]
        else:
            x2_qc = self._qEncoding(x2)
            x2_fm = self.measure_fn(x2_qc, observables=obs)        
            self._fm_dict[k_x2] = x2_fm    

        #compute kernel
        #k_computed = np.dot(x1_fm, x1_fm) #uise this for linear kernel
        k_computed = self._pqk_compute_kernel(x1_fm, x2_fm)
        return k_computed
    
    
    def _kernel_matrix(self, A, B):
        """
        compute the kernel matrix (Gram if A==B)    
        """        
        return np.array([[self._qfKernel(a, b) for b in B] for a in A])
    
    
    def save_feature_map(self, prefix = ''):
        
        '''
        create a csv file with feature maps
        '''
        
        current_timestamp = time.time()
        datetime_object = datetime.datetime.fromtimestamp(current_timestamp)
        formatted_datetime = datetime_object.strftime("%Y%m%d%H%M%S")
        csv_file = '../qfm/' + prefix + str(formatted_datetime) + '.csv'        
        
        main_path = os.path.dirname(__file__)
        file_path = os.path.join(main_path, csv_file)

        #store the features map
        with open(file_path, 'w') as f:            
            f.write(','.join(['key', 'value']))
            f.write('\n') # Add a new line
            for i, (k,v) in enumerate(self._fm_dict.items()):
                l_item = [k,str(v)]
                f.write(','.join(l_item))
                f.write('\n') # Add a new line

        #print the related time stamp. 
        print(f'Timestamp of the file storing data: {formatted_datetime}')
    

    def save_latent_space(self, prefix = '', *, y):
        '''
        Save latent space
        '''
        #create a csv file with feature maps
        current_timestamp = time.time()
        datetime_object = datetime.datetime.fromtimestamp(current_timestamp)
        formatted_datetime = datetime_object.strftime("%Y%m%d%H%M%S")
        csv_file = '../qfm/lt_space/' + prefix + str(formatted_datetime) + '.csv' 

        main_path = os.path.dirname(__file__)
        file_path = os.path.join(main_path, csv_file)               
        
        df = pd.DataFrame(self._fm_dict.values())         
        df['label'] = y

        df.to_csv(file_path, encoding='utf-8', index=False)
            
        #print the related time stamp. 
        print(f'Timestamp of the file storing data: {formatted_datetime}')
    

    def get_q_encoding(self):
        '''
        get latent space
        '''
        return self._fm_dict
    


    def fit(self, X, y):
        
        """
        need to reimplements fit in order to manage the latent cache
        """        

        if len(self.obs) == 0:
            print('WARNING: provide observables')
        
        #clear the cache
        if self.fit_clear:
            self._fm_dict.clear()      

        super().fit(X=X, y=y)       
                
        return self

   
        