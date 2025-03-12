import numpy as np
import pandas as pd
from qiskit import QuantumCircuit

from pqk.QMeasures import QMeasures

class QEncoding:


    
    def __init__(self, *,qcircuit: QuantumCircuit, obs = 'Z', data: np.ndarray):
        """
        Initializes the QEncoding object.

        Args:
            qcircuit (QuantumCircuit): The quantum circuit used for encoding.
            obs (str, optional): The observable to measure. Defaults to 'Z'.
            data (np.ndarray): The data to be encoded.
        """
        self.qcircuit = qcircuit
        self.obs = obs  # Store the observable.
        self.data = data

        # Dictionary to store the encoded data
        self.encoded_data = []    


    
    def encode(self):
        """
        Encodes the data using the quantum circuit.

        Iterates over each data point in self.data, assigns it as parameters to the 
        quantum circuit, and stores the bound circuit and the associated key in a dictionary.
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

                #measure
                res =  QMeasures.StateVectorEstimator(bound_circuit, self.obs)

                # Store the bound circuit in the dictionary.
                self.encoded_data.append(res)

            except Exception as e:
                print(f"Error encoding data point number {i}: {e}")
                
        print('End data encoding')

        return self.encoded_data
    

    def save_encoding(self, y_label: np.ndarray, file_name: str) -> pd.DataFrame:
        #recreate data frame from encoding 
        df = pd.DataFrame(self.encoded_data)
        df['label'] = y_label.tolist()
        df.to_csv(file_name, index=False)

        return df


