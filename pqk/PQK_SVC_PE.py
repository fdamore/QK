from qiskit import QuantumCircuit
from pqk.PQK_SVC import PQK_SVC
from pqk.QMeasures import QMeasures
from pqk.CKernels import CKernels



class PQK_SVC_PE(PQK_SVC):  
    
    def __init__(self,C = 1, gamma = 0.5, fit_clear = True, obs = ['Z'], c_kernel = CKernels.rbf,*,nshots =100,  circuit : QuantumCircuit):

        super().__init__(C=C, gamma=gamma, fit_clear=fit_clear, obs=obs, c_kernel=c_kernel,circuit=circuit)
       
        """         
        obs are the observation used to project state vector back to classical space
        circuit is the quantum circuit used for encode classical data        
        c_kernel is the used classical kernel        
        nshost is the number of shots the Primitive estimation are using
        """ 

        self.nshots = nshots 

    
    def metadata(self):
        print(f'*** CIRCUITS qubit ***')                
        print(self.circuit.draw())
        print(f'*** Required observables: {self.obs}')     
        print(f'*** Measure procedure: {QMeasures.PrimitiveEstimator.__name__}')
        print(f'*** Measure function used: { self.measure_fn.__name__}')   
        print(f'*** CKernel function used: {self.c_kernel.__name__}')
        print(f'Param: {self.get_params}')
        print(f'Qubits: {self.circuit.num_qubits}')        
        return ""
    


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
            x1_fm = QMeasures.PrimitiveEstimator(observables=obs, qc=x1_qc, nshots = self.nshots)
            
            self._fm_dict[k_x1] = x1_fm

        #check the k2 and get feature map
        x2_fm = None
        if k_x2 in self._fm_dict:
            x2_fm = self._fm_dict[k_x2]
        else:
            x2_qc = self._qEncoding(x2)
            x2_fm = QMeasures.PrimitiveEstimator(observables=obs, qc=x2_qc, nshots = self.nshots)  
            self._fm_dict[k_x2] = x2_fm    

        #compute kernel
        #k_computed = np.dot(x1_fm, x1_fm) #uise this for linear kernel
        k_computed = self.c_kernel(x1_fm, x2_fm)
        return k_computed
    
    
    

   
        