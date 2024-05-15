
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit.primitives import Estimator as PrimitiveEstimator 
from qiskit.primitives import StatevectorEstimator

import os
import time
import datetime

from qiskit.circuit.library import ZZFeatureMap


from functools import wraps

def singleton(cls):
    @wraps(cls)
    def _wrap(*args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = cls(*args, **kwargs)
        return cls._instance
    return _wrap


#embeddings
class Circuits:


    @staticmethod
    def zzfeaturemap(n_wire, full_ent = True):
        zfm = ZZFeatureMap(feature_dimension=n_wire)
        return zfm


    #cascade embedding
    @staticmethod
    def cascade(n_wire, full_ent = True):

        # Create a new circuit with two qubits
        qc = QuantumCircuit(n_wire)
        
        for i in range(n_wire):

            #add hadamrd
            qc.h(i)

            phi_name = 'phi_'+str(i)
            phi = Parameter(phi_name)
            qc.rx(phi, i)

            if(full_ent):
                qc.cx(i%n_wire, (i+1)%n_wire)

            #add hadamrd
            qc.h(i)
        
        return qc

    #embedding - encoded used in paper Hubregsten et all.
    @staticmethod
    def ansatz_encoded(n_wire, full_ent = True):

        # Create a new circuit with two qubits
        qc = QuantumCircuit(n_wire)
        
        for i in range(n_wire):
                    
            #add hadamrd
            qc.h(i)

        for i in range(n_wire):
            
            phi_name = 'phi_'+str(i)
            phi = Parameter(phi_name)
            qc.rz(phi, i)

        if(full_ent):
            for i in range(n_wire):            
                qc.cx(i%n_wire, (i+1)%n_wire)

        
        qc.barrier()    

        
        for i in range(n_wire):
            #add hadamrd
            qc.h(i)
        
        return qc;
    
    

    @staticmethod
    def x_encoded(n_wire, full_ent = True):

        # Create a new circuit with two qubits
        qc = QuantumCircuit(n_wire)        
        

        for i in range(n_wire):            
            phi_name = 'phi_'+str(i)
            phi = Parameter(phi_name)
            qc.rx(phi, i)      

        if(full_ent):
            for i in range(n_wire):
                qc.cx(i%n_wire, (i+1)%n_wire)    
      
        
        return qc 


    @staticmethod
    def y_encoded(n_wire, full_ent = True):

        # Create a new circuit with two qubits
        qc = QuantumCircuit(n_wire)        
        

        for i in range(n_wire):            
            phi_name = 'phi_'+str(i)
            phi = Parameter(phi_name)
            qc.ry(phi, i)

        if(full_ent):
            for i in range(n_wire):            
                qc.cx(i%n_wire, (i+1)%n_wire)        
      
        
        return qc


    @staticmethod
    def y_encoded_scaled(n_wire, full_ent = True):

        # Create a new circuit with two qubits
        qc = QuantumCircuit(n_wire)               

        for i in range(n_wire):
            
            phi_name = 'phi_'+str(i)
            phi = Parameter(phi_name)
            qc.ry(((phi + 1)/2) * np.pi, i)

        if(full_ent):
            for i in range(n_wire):
                qc.cx(i%n_wire, (i+1)%n_wire)        
      
        
        return qc




    @staticmethod
    def z_encoded(n_wire, full_ent = True):

        # Create a new circuit with two qubits
        qc = QuantumCircuit(n_wire)   
        

        for i in range(n_wire):            
            phi_name = 'phi_'+str(i)
            phi = Parameter(phi_name)
            qc.rz(phi, i) 

        if(full_ent):
            for i in range(n_wire):            
                qc.cx(i%n_wire, (i+1)%n_wire)       
        
        return qc;

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

    def __init__(self, nwire = 1, obs = ['Z'], full_ent = True, qtemplate = Circuits.ansatz_encoded):
        print('*** Create a Container ***')
        self.build(nwire=nwire, obs=obs,full_ent=full_ent, qtemplate=qtemplate)
    
    def build(self, nwire = 1, obs = ['Z'], full_ent = True, qtemplate = Circuits.ansatz_encoded):
        #define parameters
        self.obs = obs
        self.nwire = nwire 
        self.template = qtemplate
        self.full_ent =full_ent

        print(f'*** Created quantum template for feature map using {str(self.nwire)} qubit ***')        
        self.circuit = self.template(self.nwire,  self.full_ent)
        print(self.circuit.draw())
        print(f'*** Required observables: {self.obs}')

        if len(self.obs) == 0:
            print('WARNING: provide observables')
        
        #clear the cache
        self.fm_dict.clear()


    def metadata(self):
        print(f'*** Quantum template for feature map using {str(self.nwire)} qubit ***')                
        print(self.circuit.draw())
        print(f'*** Required observables: {self.obs}')

    
    #save my feature map
    def save_feature_map(self, prefix = ''):
        #create a csv file with feature maps
        current_timestamp = time.time()
        datetime_object = datetime.datetime.fromtimestamp(current_timestamp)
        formatted_datetime = datetime_object.strftime("%Y%m%d%H%M%S")
        csv_file = '../qfm/' + prefix + str(formatted_datetime) + '.csv'        
        
        main_path = os.path.dirname(__file__)
        file_path = os.path.join(main_path, csv_file)
        

        with open(file_path, 'w') as f:            
            f.write(','.join(['key', 'value']))
            f.write('\n') # Add a new line
            for i, (k,v) in enumerate(self.fm_dict.items()):
                l_item = [k,str(v)]
                f.write(','.join(l_item))
                f.write('\n') # Add a new line
            

#encode data in parameter
def qEncoding(qc, data):               
    qc_assigned = qc.assign_parameters(data, inplace = False)
    return qc_assigned;


#alternative key
def get_key(x):
    x_r = x
    l = []
    for itm in x_r:
        l.append(str(itm))
    return '@'.join(l)


    

#define qquantum feature kernel
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
        x1_qc = qEncoding(qc_template, x1)
        x1_fm = evalObsAer(x1_qc, observables=obs)        
        circuit_container.fm_dict[k_x1] = x1_fm

    #check the k2 and get feature map
    x2_fm = None
    if k_x2 in circuit_container.fm_dict:
        x2_fm = circuit_container.fm_dict[k_x2]
    else:
        x2_qc = qEncoding(qc_template, x2)
        x2_fm = evalObsAer(x2_qc, observables=obs)        
        circuit_container.fm_dict[k_x2] = x2_fm    

    #compute kernel
    k_computed = np.dot(x1_fm, x2_fm)
    return k_computed
 

def kernel_matrix(A, B):
    """Compute the matrix whose entries are the kernel
       evaluated on pairwise data from sets A and B."""
    return np.array([[qfKernel(a, b) for b in B] for a in A])


#measure on quantum circuits
def evalObsAer(qc, observables):    

    obs = [SparsePauliOp(label) for label in observables]
    
    estimator = AerEstimator() 
    estimator.options.default_precision = 0  

    obs = [
        observable.apply_layout(qc.layout) for observable in obs
    ]
    
    # One pub, with one circuit to run against observables.
    job = estimator.run([(qc, obs)])
    
    # This is the result of the entire submission.  We submitted one Pub,
    # so this contains one inner result (and some metadata of its own).
    job_result = job.result()
    
     

    return job_result[0].data.evs
    
#measure on quantum circuits
def evalObsPrimitive(qc, observables):         
    
    estimator = PrimitiveEstimator(options={'shots':100}) 
    

    l = []         

    for itm in observables:
        job = estimator.run(qc, itm)
        job_result = job.result()
        l.append(job_result.values[0])   

    #return job_result[0].data.evs
    return np.array(l)

#measure on quantum circuits
def evalObsStateVector(qc, observables):         
    
    estimator = StatevectorEstimator(default_precision=0)     

    obs = [SparsePauliOp(label) for label in observables]

    pub = (qc, obs)
    job = estimator.run([pub])
    result = job.result()[0]
    return result.data.evs
    


if __name__ == "__main__":
    pass
    



    
    
