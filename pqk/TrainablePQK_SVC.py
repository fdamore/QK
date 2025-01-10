import datetime
import os
import time
import numpy as np
from typing import Sequence

from qiskit_machine_learning.kernels import TrainableKernel, BaseKernel

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector

from pqk.QMeasures import QMeasures
from pqk.CKernels import CKernels




#define my trainable quantum kernel using the outer quantum kernel.
# pqk.Circuit should be defined before running the training process
class TrainablePQK_SVC(TrainableKernel, BaseKernel):
        
       
        

        def __init__(self, obs = ['Z'], measure_fn = QMeasures.PrimitiveEstimator, c_kernel = CKernels.linear, save_feature_map = False, *,
                     feature_map: QuantumCircuit | None = None, 
                     training_parameters: ParameterVector | Sequence[Parameter] | None = None) -> None:
                        
                        super().__init__(feature_map=feature_map,  training_parameters=training_parameters)
                         # Override the number of features defined in the base class.
                        self._num_features = feature_map.num_parameters - self._num_training_parameters
                        self._feature_parameters = [parameter for parameter in feature_map.parameters 
                                                    if parameter not in self._training_parameters]
                        self._parameter_dict = {parameter: None for parameter in self.feature_map.parameters}

                        #configure the tranable PQK instance
                        self._fm_dict = {}
                        self.obs = obs               
                        self.measure_fn = measure_fn
                        self.c_kernel = c_kernel
                        self.save_feature_map = save_feature_map


        def metadata(self):
            '''
            Print metadata
            '''
            print(f'*** Quantum template for feature map using {str(self._num_features)} Features ***')                
            print(self.feature_map.draw())
            print(f'*** Required observables: {self.obs}')
            print(f'*** Measure procedure: {self.measure_fn.__name__}')
            print(f'*** CKernel function used: {self.c_kernel.__name__}')            
            return ""
        
        def _qEncoding(self, data):               
             '''
             Encode data in parameter
             '''
             qc_assigned = self._feature_map.assign_parameters(data, inplace = False)
             return qc_assigned

        
        def _qfKernel(self, x1, x2):
            '''
            Define the quantum kernel
            '''                        
            
            #define the key
            k_x1 = str(x1) #get_key(x1) 
            k_x2 = str(x2) #get_key(x2)

            #check the k1 and get feature map
            x1_fm = None
            if k_x1 in self._fm_dict:
                x1_fm = self._fm_dict[k_x1]
            else:
                x1_qc = self._qEncoding(x1)
                x1_fm = self.measure_fn(x1_qc, observables=self.obs)                
                self._fm_dict[k_x1] = x1_fm

            #check the k2 and get feature map
            x2_fm = None
            if k_x2 in self._fm_dict:
                x2_fm = self._fm_dict[k_x2]
            else:
                x2_qc = self._qEncoding(x2)
                x2_fm = self.measure_fn(x2_qc, observables=self.obs)        
                self._fm_dict[k_x2] = x2_fm    

            #compute kernel            
            k_computed = self.c_kernel(x1_fm, x2_fm)
            return k_computed      
       

        #hook methods
        def evaluate(self,x_vec: np.ndarray,y_vec: np.ndarray | None = None) -> np.ndarray:

            #clear the cache
            self._fm_dict.clear()

            #the gram matrix
            gram_matrix = None
                         
            #x_vec, y_vec = self._validate_input(x_vec, y_vec)
            new_x_vec = self._parameter_array(x_vec)
            if y_vec is not None:
                new_y_vec = self._parameter_array(y_vec)
                gram_matrix = self._kernel_matrix(new_x_vec, new_y_vec)
            else:
                gram_matrix = self._kernel_matrix(new_x_vec, new_x_vec)
            
            #if needed, save feature map
            if self.save_feature_map:                                  
                 self.save_fm(prefix='TKFM-')
            
            return gram_matrix
        

        def _kernel_matrix(self, A, B):             
            """
            compute the kernel matrix (Gram if A==B)    
            """        
            return np.array([[self._qfKernel(a, b) for b in B] for a in A])           
             

                    
        def save_fm(self, prefix = ''):
            '''
            #create a csv file with feature maps
            '''
            current_timestamp = time.time()
            datetime_object = datetime.datetime.fromtimestamp(current_timestamp)
            formatted_datetime = datetime_object.strftime("%Y%m%d%H%M%S%f")
            csv_file = '../qfm/tkfm/' + prefix + str(formatted_datetime) + '.csv'        

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

            


        
