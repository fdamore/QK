import numpy as np

import os
import time
import datetime
from qke.QMeasures import QMeasures
from qke.Circuits import Circuits
from qke.CKernels import CKernels

from functools import wraps

def singleton(cls):
    @wraps(cls)
    def _wrap(*args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = cls(*args, **kwargs)
        return cls._instance
    return _wrap


#singleton container for circuit
@singleton
class CircuitContainer:   

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
    kernel = None  

    def __init__(self, nwire = 1, obs = ['Z'], full_ent = True, qtemplate = Circuits.ansatz_encoded, measure_fn = QMeasures.Aer, c_kernel = CKernels.linear):
        print('*** Create a Container ***')
        self.build(nwire=nwire, obs=obs,full_ent=full_ent, qtemplate=qtemplate, measure_fn= measure_fn, c_kernel=c_kernel)
    
    def build(self, nwire = 1, obs = ['Z'], full_ent = True, qtemplate = Circuits.ansatz_encoded, measure_fn = QMeasures.Aer, c_kernel = CKernels.linear):
        #define parameters
        self.obs = obs
        self.nwire = nwire 
        self.template = qtemplate
        self.full_ent =full_ent
        self.measure_fn = measure_fn
        self.kernel = c_kernel

        self.circuit = self.template(self.nwire,  self.full_ent)

        #print metadata
        self.metadata()

        if len(self.obs) == 0:
            print('WARNING: provide observables')
        
        #clear the cache
        self.fm_dict.clear()


    def metadata(self):
        print(f'*** Quantum template for feature map using {str(self.nwire)} qubit ***')                
        print(self.circuit.draw())
        print(f'*** Required observables: {self.obs}')
        print(f'*** Measure procedure: {self.measure_fn.__name__}')
        print(f'*** CKernel function used: {self.kernel.__name__}')
        return ""

    #encode data in parameter
    @staticmethod
    def qEncoding(qc, data):               
        qc_assigned = qc.assign_parameters(data, inplace = False)
        return qc_assigned;  

    #define qquantum feature kernel using CircuitContainer
    @staticmethod
    def qfKernel(x1, x2):

        #get info about obs and circuits
        circuit_container = CircuitContainer()  
        qc_template = circuit_container.circuit
        obs = circuit_container.obs

        #define the key
        k_x1 = str(x1) #get_key(x1) 
        k_x2 = str(x2) #get_key(x2)

        #check the k1 and get feature map
        x1_fm = None
        if k_x1 in circuit_container.fm_dict:
            x1_fm = circuit_container.fm_dict[k_x1]
        else:
            x1_qc = CircuitContainer.qEncoding(qc_template, x1)        
            x1_fm = circuit_container.measure_fn(x1_qc, observables=obs)
            
            circuit_container.fm_dict[k_x1] = x1_fm

        #check the k2 and get feature map
        x2_fm = None
        if k_x2 in circuit_container.fm_dict:
            x2_fm = circuit_container.fm_dict[k_x2]
        else:
            x2_qc = CircuitContainer.qEncoding(qc_template, x2)
            x2_fm = circuit_container.measure_fn(x2_qc, observables=obs)        
            circuit_container.fm_dict[k_x2] = x2_fm    

        #compute kernel
        #k_computed = np.dot(x1_fm, x1_fm) #uise this for linear kernel
        k_computed = circuit_container.kernel(x1_fm, x1_fm)
        return k_computed
    
    #compute the kernel matrix (Gram if A==B)
    @staticmethod
    def kernel_matrix(A, B):
        #Compute gram matrix
        return np.array([[CircuitContainer.qfKernel(a, b) for b in B] for a in A]) 
    
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