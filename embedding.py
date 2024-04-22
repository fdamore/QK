
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from sklearn.gaussian_process.kernels import RBF, DotProduct

from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit.primitives import Estimator as PrimitiveEstimator 
from qiskit.primitives import StatevectorEstimator


from functools import wraps

def singleton(cls):
    @wraps(cls)
    def _wrap(*args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = cls(*args, **kwargs)
        return cls._instance
    return _wrap



class QEmbedding:

    #cascade embedding
    @staticmethod
    def createCascadeQuantumEmbedding(n_wire):

        # Create a new circuit with two qubits
        qc = QuantumCircuit(n_wire)
        
        for i in range(n_wire):

            #add hadamrd
            qc.h(i)

            phi_name = 'phi_'+str(i)
            phi = Parameter(phi_name)
            qc.rx(phi, i)

            #entagled? Why not
            qc.cx(i%n_wire, (i+1)%n_wire)

            #add hadamrd
            qc.h(i)
        
        return qc

    #embedding
    @staticmethod
    def createQuantumEmbedding(n_wire):

        # Create a new circuit with two qubits
        qc = QuantumCircuit(n_wire)
        
        for i in range(n_wire):
                    
            #add hadamrd
            qc.h(i)

        for i in range(n_wire):
            
            phi_name = 'phi_'+str(i)
            phi = Parameter(phi_name)
            qc.rz(phi, i)

        for i in range(n_wire):

            #entagled? Why not
            qc.cx(i%n_wire, (i+1)%n_wire)

        
        qc.barrier()    

        
        for i in range(n_wire):

            #add hadamrd
            qc.h(i)
        
        return qc;

#singleton container for circuit
@singleton
class CircuitContainer:

    circuit = None;
    nwire = None;
    f_embedding = None     

    def __init__(self, nwire, f_embedding = QEmbedding.createQuantumEmbedding):
        self.nwire = nwire 
        self.f_embedding = f_embedding 
    
    def get_circuit(self):        
        if not self.circuit:
            print(f'Create template with  {str(self.nwire)} qubit')            
            self.circuit = self.f_embedding(self.nwire)
        return self.circuit
    


    

#encode data in parameter
def qEncoding(qc, data):               
    qc_assigned = qc.assign_parameters(data, inplace = False)
    return qc_assigned;
    

#define qquantum feature kernel
def qfKernel(x1, x2, qc):
    obs=["ZIIIII","IZIIII", "IIZIII", "IIIZII","IIIIZI",'IIIIIZ']  
    #obs=["ZIIIII","IZIIII"]#, "IIZIII", "IIIZII","IIIIZI",'IIIIIZ']  
    #obs=["ZZIIII","IZIIII", "IIZZII", "IIIZII","IIIIZI",'ZIIIIZ'] 

    

    x1_qc = qEncoding(qc, x1)

    f_x1 = evalObsAer(x1_qc, observables=obs)
    #f_x1 = evalObsStateVector(x1_qc, observables=obs)

    x2_qc = qEncoding(qc, x2)
    f_x2 = evalObsAer(x2_qc, observables=obs)
    #f_x2 = evalObsStateVector(x2_qc, observables=obs)

    #compute kernel
    k_computed = np.dot(f_x1, f_x2)
    return k_computed;




#define qquantum feature kernel
def qfSVCKernel(x1, x2):     

    n_qubit = len(x1)           
    
    circuit_container = CircuitContainer(n_qubit)    
    qc_template = circuit_container.get_circuit()   

    return qfKernel(x1, x2, qc_template)    

def kernel_matrix(A, B):
    """Compute the matrix whose entries are the kernel
       evaluated on pairwise data from sets A and B."""
    return np.array([[qfSVCKernel(a, b) for b in B] for a in A])


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
    
