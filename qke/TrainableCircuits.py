
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit.circuit import ParameterVector


class TrainableCircuits:

    #quantum circuit
    qc = None

    #trainable parameters
    training_parameters = None

    def __init__(self, *, qc, training_parameters) -> None:
        self.qc = qc
        self.training_parameters = training_parameters
        

    @staticmethod
    def zzfm(n_wire):
        trainable_fm = QuantumCircuit(n_wire)
        training_params = ParameterVector("θ_par", n_wire)

        # Create an initial rotation layer of trainable parameters
        for i, param in enumerate(training_params):
            trainable_fm.ry(param, trainable_fm.qubits[i])
            zzfm = ZZFeatureMap(feature_dimension=n_wire)
        
        fm = trainable_fm.compose(zzfm)
        
        #return the fm
        return TrainableCircuits(qc=fm, training_parameters=training_params)
    

    @staticmethod
    def d_stack(n_wire):
        fm = QuantumCircuit(n_wire)
        input_params = ParameterVector("x_par", n_wire)
        training_params = ParameterVector("θ_par", n_wire)

        # Create an initial rotation layer of trainable parameters
        for i, param in enumerate(training_params):
            fm.ry(param, fm.qubits[i])

        # Create a rotation layer of input parameters
        for i, param in enumerate(input_params):
            fm.rz(param, fm.qubits[i])
        
        #return the fm
        return TrainableCircuits(qc=fm, training_parameters=training_params)
    

    @staticmethod
    def twl_zzfm(n_wire):
        zzfm = ZZFeatureMap(feature_dimension=n_wire)
        #circuit = TwoLocal(n_qubits, 'ry', 'cx', 'linear', reps=n_layers)
        twl = TwoLocal(num_qubits=n_wire, 
                   rotation_blocks='ry', 
                   entanglement_blocks='cx', 
                   entanglement='full', 
                   reps=1, 
                   insert_barriers=True, skip_final_rotation_layer=True)
        fm = twl.compose(zzfm)

        #use the traing paramenters in TwoLoca
        training_params = twl.parameters

        #return the fm
        return TrainableCircuits(qc=fm, training_parameters=training_params)
    
    @staticmethod
    def trainable_twl(n_wire):
        trainable_fm = QuantumCircuit(n_wire)
        #circuit = TwoLocal(n_qubits, 'ry', 'cx', 'linear', reps=n_layers)
        fm = TwoLocal(num_qubits=n_wire, 
                   rotation_blocks='ry', 
                   entanglement_blocks='cx', 
                   entanglement='full', 
                   reps=1, 
                   insert_barriers=True, skip_final_rotation_layer=False)
        
        #define the traning paramenter
        training_params = fm.parameters[n_wire:]

        #return the fm
        return TrainableCircuits(qc=fm, training_parameters=training_params)

        
