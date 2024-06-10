from numpy import ndarray
from qiskit_machine_learning.kernels import TrainableKernel
import qke.qproc as qk

#define my trainable quantum kernel using the outer quantum kernel.
# qke.Circuit should be defined before running the training process
class TrainableOuterQuantumKernel(TrainableKernel):

    #define qquantum feature kernel
    def _kernel(x1, x2):
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
            x1_fm = circuit_container.measure_fn(x1_qc, observables=obs)
            
            circuit_container.fm_dict[k_x1] = x1_fm

        #check the k2 and get feature map
        x2_fm = None
        if k_x2 in circuit_container.fm_dict:
            x2_fm = circuit_container.fm_dict[k_x2]
        else:
            x2_qc = qEncoding(qc_template, x2)
            x2_fm = circuit_container.measure_fn(x2_qc, observables=obs)        
            circuit_container.fm_dict[k_x2] = x2_fm    

        #compute kernel
        k_computed = np.dot(x1_fm, x2_fm)
        return k_computed




    #evaluate using my kernel
    def evaluate(self, x_vec: ndarray, y_vec: ndarray | None = None) -> ndarray:
        self._kernel(x_vec, y_vec)

