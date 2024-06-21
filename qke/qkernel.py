import numpy as np
from qke.CircuitContainer import CircuitContainer

# In this file the procedures to compute kernel entry (qfkernel) and Gram matrix (kernel_matrix) are provided. 
# The procedures will be used in scikit-learn SVC in order to comnpute quantum kernel using the outer approach.
# The provided procedure use CircuitContainer in order to process data using quantum feature map 


#define qquantum feature kernel using CircuitContainer
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
 

#compute the kernel matrix (Gram if A==B)
def kernel_matrix(A, B):
    """Compute the matrix whose entries are the kernel
       evaluated on pairwise data from sets A and B."""
    return np.array([[qfKernel(a, b) for b in B] for a in A])   

#encode data in parameter
def qEncoding(qc, data):               
    qc_assigned = qc.assign_parameters(data, inplace = False)
    return qc_assigned;    


if __name__ == "__main__":
    pass
    



    
    
