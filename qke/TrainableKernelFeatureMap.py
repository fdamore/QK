import numpy as np
from typing import Sequence

from qiskit_machine_learning.kernels import TrainableKernel, BaseKernel

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector

from qke.qproc import QMeasures




#define my trainable quantum kernel using the outer quantum kernel.
# qke.Circuit should be defined before running the training process
class TrainableKernelFeatureMap(TrainableKernel, BaseKernel):
        
       
        

        def __init__(self,*,
                     feature_map: QuantumCircuit | None = None, 
                     training_parameters: ParameterVector | Sequence[Parameter] | None = None) -> None:
                        
                        super().__init__(feature_map=feature_map,  training_parameters=training_parameters)
                         # Override the number of features defined in the base class.
                        self._num_features = feature_map.num_parameters - self._num_training_parameters
                        self._feature_parameters = [parameter for parameter in feature_map.parameters 
                                                    if parameter not in self._training_parameters]
                        self._parameter_dict = {parameter: None for parameter in self.feature_map.parameters} 

              
      
        
        #the type of measures        
        obs = ['Z']       

        #cache
        fm_dict = {}   

        #primitive estimator use nshots (if usable)
        nshots = 100

        #The quantum measure function used
        q_measure = None

        def configure(self, obs = ['Z'], nshots = 100, q_measure = QMeasures.PrimitiveEstimator):            
            self.obs = obs      
            self.nshots = nshots
            self.q_measure = q_measure
        
        #encode data in parameter
        def qEncoding(self, data):               
             qc_assigned = self._feature_map.assign_parameters(data, inplace = False)
             return qc_assigned

        #define qquantum feature kernel
        def qfKernel(self, x1, x2):

            #get info about obs and circuits                        

            #define the key
            k_x1 = str(x1) #get_key(x1) 
            k_x2 = str(x2) #get_key(x2)

            #check the k1 and get feature map
            x1_fm = None
            if k_x1 in self.fm_dict:
                x1_fm = self.fm_dict[k_x1]
            else:
                x1_qc = self.qEncoding(x1)        
                x1_fm = self.q_measure(x1_qc, observables=self.obs, nshots=self.nshots)
                self.fm_dict[k_x1] = x1_fm

            #check the k2 and get feature map
            x2_fm = None
            if k_x2 in self.fm_dict:
                x2_fm = self.fm_dict[k_x2]
            else:
                x2_qc = self.qEncoding(x2)
                x2_fm = self.q_measure(x2_qc, observables=self.obs, nshots=self.nshots) 
                self.fm_dict[k_x2] = x2_fm    

            #compute kernel
            k_computed = self._compute_kernel_score(x1_fm, x2_fm)
            return k_computed
        
        #compute a kernel function
        def _compute_kernel_score(self, x: np.ndarray, y: np.ndarray) -> float:
            return x.dot(y)

        #hook methods
        def evaluate(self,x_vec: np.ndarray,y_vec: np.ndarray | None = None) -> np.ndarray:
                         
            #x_vec, y_vec = self._validate_input(x_vec, y_vec)
            new_x_vec = self._parameter_array(x_vec)
            if y_vec is not None:
                new_y_vec = self._parameter_array(y_vec)
                return self.kernel_matrix(new_x_vec, new_y_vec)
            else:
                return self.kernel_matrix(new_x_vec, new_x_vec)
                 
             

        def kernel_matrix(self, A, B):
            """Compute the matrix whose entries are the kernel
            evaluated on pairwise data from sets A and B."""
            return np.array([[self.qfKernel(a, b) for b in B] for a in A])    


        
