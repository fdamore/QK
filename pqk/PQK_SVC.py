from debugpy import configure
import numpy as np

import os
import time
import datetime

from pqk.QMeasures import QMeasures
from pqk.Circuits import Circuits
from pqk.CKernels import CKernels
from sklearn.svm import SVC


class PQK_SVC(SVC):   

    #quantum circuit used to define feature map
    circuit = None

    #quantum template for the circuit
    template = None    

    #number of qubit
    nwire = None

    #list of observables
    obs = None

    #store computed feature map
    fm_dict = {}

    #measure function
    measure_fn = None  

    #the classical kernel used to compute
    c_kernel = None

    
    
    def __init__(self,C = 1, gamma = 0.5, fit_clear = True, print_param = False, nwire = 1, obs = ['Z'], full_ent = True, template = Circuits.xyz_encoded, measure_fn = QMeasures.Aer, c_kernel = CKernels.linear):
        
        super().__init__(C=C, gamma=gamma, kernel=self.__kernel_matrix)

        """
         
        nwire are the number of qubits
        obs are the observation used to project state vector back to classical space
        qtemplate is the circuit template used to  encode data in quantum space
        measure_fn is the measure procedure used to get evs to encoded states
        c_kernel is the used classical kernel
        """

        #clear the cache before fit
        self.fit_clear = fit_clear       

        #define PQK_SVM arameters 
        self.obs = obs
        self.nwire = nwire 
        self.template = template
        self.full_ent =full_ent
        self.measure_fn = measure_fn
        self.c_kernel = c_kernel       
        self.circuit = self.template(self.nwire,  self.full_ent)        


    def metadata(self):
        print(f'*** Quantum template for feature map using {str(self.nwire)} qubit ***')                
        print(self.circuit.draw())
        print(f'*** Required observables: {self.obs}')
        print(f'*** Measure procedure: {self.measure_fn.__name__}')
        print(f'*** CKernel function used: {self.c_kernel.__name__}')
        return ""

    #encode data in parameter    
    def __qEncoding(self, qc, data):         
        qc_assigned = qc.assign_parameters(data, inplace = False)
        return qc_assigned;  

    #define quantum feature kernel using CircuitContainer    
    def __qfKernel(self, x1, x2):

        
        qc_template = self.circuit
        obs = self.obs

        #define the key
        k_x1 = str(x1) #get_key(x1) 
        k_x2 = str(x2) #get_key(x2)

        #check the k1 and get feature map
        x1_fm = None
        if k_x1 in self.fm_dict:
            x1_fm = self.fm_dict[k_x1]
        else:
            x1_qc = self.__qEncoding(qc_template, x1)        
            x1_fm = self.measure_fn(x1_qc, observables=obs)
            
            self.fm_dict[k_x1] = x1_fm

        #check the k2 and get feature map
        x2_fm = None
        if k_x2 in self.fm_dict:
            x2_fm = self.fm_dict[k_x2]
        else:
            x2_qc = self.__qEncoding(qc_template, x2)
            x2_fm = self.measure_fn(x2_qc, observables=obs)        
            self.fm_dict[k_x2] = x2_fm    

        #compute kernel
        #k_computed = np.dot(x1_fm, x1_fm) #uise this for linear kernel
        k_computed = self.c_kernel(x1_fm, x2_fm)
        return k_computed
    
    
    def __kernel_matrix(self, A, B):
        """
        compute the kernel matrix (Gram if A==B)    
        """        
        return np.array([[self.__qfKernel(a, b) for b in B] for a in A]) 
    
    #save my feature map
    def save_feature_map(self, prefix = ''):
        #create a csv file with feature maps
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
            for i, (k,v) in enumerate(self.fm_dict.items()):
                l_item = [k,str(v)]
                f.write(','.join(l_item))
                f.write('\n') # Add a new line

        #print the related time stamp. 
        print(f'Timestamp of the file storing data: {formatted_datetime}')
    
    
    
    def fit(self, X, y):
        
        """
        need to reimplements fit in order to manage the latent cache
        """        

        if len(self.obs) == 0:
            print('WARNING: provide observables')
        
        #clear the cache
        if self.fit_clear:
            self.fm_dict.clear()      

        super().fit(X=X, y=y)
                
        return self

   
        