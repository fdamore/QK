import numpy as np
from numpy.linalg import norm


class CKernels:    

    #compute a kernel function
    @staticmethod
    def linear(x: np.ndarray, y: np.ndarray) -> float:
            return x.dot(y)            
    
    @staticmethod
    def rbf(x: np.ndarray, y: np.ndarray) -> float:
          return np.exp(0.5 * -(norm(x - y))**2)
            
    

