import numpy as np
import pandas as pd
from qiskit import QuantumCircuit

from pqk.QMeasures import QMeasures

class QEncoding:


    
    def __init__(self, *,qcircuit: QuantumCircuit, obs = 'Z', data: np.ndarray, use_pe = 'sv'):
        """
        Initializes the QEncoding object.

        Args:
            qcircuit (QuantumCircuit): The quantum circuit used for encoding.
            obs (str, optional): The observable to measure. Defaults to 'Z'.
            data (np.ndarray): The data to be encoded. 
            use_pe (str, optional): Type of estimator to use. Options are 'pe' for PrimitiveEstimator,
                                    'sv' for StateVectorEstimator, 'gpu_sv' for GPUAerStateVectorEstimator,
                                    and 'gpu_aer_sv' for GPUAerVigoNoiseStateVectorEstimator. Defaults to 'sv'. 

        """
        self.qcircuit = qcircuit
        self.obs = obs  # Store the observable.
        self.data = data
        self.use_pe = use_pe
        # Dictionary to store the encoded data
        self.encoded_data = []    


    
    def encode(self, nshots = 100, shots_seed = 123):
        """
        Encodes the data using the quantum circuit.

        Iterates over each data point in self.data, assigns it as parameters to the 
        quantum circuit, and stores the bound circuit and the associated key in a dictionary.
        
        nshots: Number of shots for PrivateEstimator. ignored if self.use_pe is False. 
        shots_seed: Seed for PrivateEstimator. ignored if self.use_pe is False        

        Returns:
            None
        
        """
        if self.data is None or len(self.data) == 0:
            print("Warning: No data provided for encoding.")
            return

        print(f'Encoding {len(self.data)} data point')
        for i, data_point in enumerate(self.data):
            try:               
               
                # Assign parameters to the quantum circuit.
                # 'inplace=False' means we get a new circuit with parameters bound.                
                bound_circuit = self.qcircuit.assign_parameters(data_point, inplace=False)

                # use StateVectorEstimator as default
                res =  QMeasures.StateVectorEstimator(bound_circuit, self.obs)

                #measure
                if self.use_pe == 'pe':
                    res =  QMeasures.PrimitiveEstimator(bound_circuit, self.obs, nshots = nshots, seed= shots_seed)
                elif self.use_pe == 'sv':
                    res =  QMeasures.StateVectorEstimator(bound_circuit, self.obs)
                elif self.use_pe == 'gpu_sv':
                    res = QMeasures.GPUAerStateVectorEstimator(qc=bound_circuit, observables=self.obs)
                elif self.use_pe == 'gpu_aer_sv':
                    res = QMeasures.GPUAerVigoNoiseStateVectorEstimator(qc=bound_circuit, observables=self.obs)

                # Store the bound circuit in the dictionary.
                self.encoded_data.append(res)

            except Exception as e:
                print(f"Error encoding data point number {i}: {e}")
                
        print('End data encoding')

        return self.encoded_data
    

    def save_encoding(self, y_label: np.ndarray, file_name: str = 'enc.csv', save_on_disk: bool = True) -> pd.DataFrame:
        #recreate data frame from encoding 
        df = pd.DataFrame(self.encoded_data)
        df['label'] = y_label.tolist()

        if save_on_disk:
            df.to_csv(file_name, index=False)

        return df
    
    def get_encoding(self, y_label: np.ndarray):
        return self.save_encoding(y_label=y_label, save_on_disk=False)



