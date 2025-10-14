import numpy as np
import pandas as pd
from qiskit import QuantumCircuit

from pqk.QMeasures import QMeasures

class QEncodingMC:

    qcs: list[(QuantumCircuit,np.ndarray)] = {}


    
    def __init__(self, *,qcs: list[(QuantumCircuit,np.ndarray)], obs = 'Z',use_pe = 'sv'):
        """
        Initializes the QEncodingMC class.
        Args:
            qcs (dict[QuantumCircuit,np.ndarray ]):
                A dictionary mapping a Parametrized QuantumCircuits with N parameter sets each of M params,
            obs (str, optional): 
                The observable to be used in computations. Defaults to 'Z'.
            use_pe (str, optional): 
                Specifies the method or backend to use for phase estimation or simulation. Defaults to 'sv'.
        Attributes:
            qcs (dict[QuantumCircuit,np.ndarray]):
                Stores the provided Parametrized Quantum Circuit Data.
            obs (str): 
                Stores the observable.
            use_pe (str): 
                Stores the phase estimation method or backend.
            encoded_data (list): 
                Stores the encoded data.
        """

       
        self.qcs = qcs
        self.obs = obs  # Store the observable.
        
        self.use_pe = use_pe
        # Dictionary to store the encoded data
        self.encoded_data = []    


    
    def encode(self, nshots = 100, shots_seed = 123):

        self.encoded_data = []
        print(self.qcs)
        for qcircuit,data in self.qcs:

            print(f'Encoding Circuit {len(data)} data point')
            for i, data_point in enumerate(data):
                try:
                    bound_circuit = qcircuit.assign_parameters(data_point, inplace=False)
                    res = QMeasures.StateVectorEstimator(bound_circuit, self.obs)
                    if self.use_pe == 'pe':
                        res = QMeasures.PrimitiveEstimator(bound_circuit, self.obs, nshots=nshots, seed=shots_seed)
                    elif self.use_pe == 'sv':
                        res = QMeasures.StateVectorEstimator(bound_circuit, self.obs)
                    elif self.use_pe == 'gpu_sv':
                        res = QMeasures.GPUAerStateVectorEstimator(qc=bound_circuit, observables=self.obs)
                    elif self.use_pe == 'gpu_aer_sv':
                        res = QMeasures.GPUAerVigoNoiseStateVectorEstimator(qc=bound_circuit, observables=self.obs)
                    self.encoded_data.append(res)
                except Exception as e:
                    print(f"Error encoding data point number {i}: {e}")
        print('End data encoding')

        return self.encoded_data
    

    def save_encoding(self, y_labels: list[np.ndarray], file_name: str = 'enc.csv', save_on_disk: bool = True) -> pd.DataFrame:
        #recreate data frame from encoding 
        df = pd.DataFrame(self.encoded_data)
        labels=[]
        for y_label in y_labels:
            labels=labels+y_label.tolist()
        print(labels)
        df['label'] = labels

        if save_on_disk:
            df.to_csv(file_name, index=False)

        return df
    
    def get_encoding(self, y_labels: list[np.ndarray]):
        return self.save_encoding(y_labels=y_labels, save_on_disk=False)



