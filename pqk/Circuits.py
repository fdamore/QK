from qiskit.circuit import Parameter
from qiskit.circuit.library import ZZFeatureMap
from qiskit.circuit.library import IQP
from qiskit.quantum_info import random_hermitian
from qiskit import QuantumCircuit
import numpy as np


class Circuits:

    @staticmethod
    def iqpfeaturemap(n_wire, full_ent = True):
        mat_param = np.real(random_hermitian(n_wire, seed=1234))
        circuit = IQP(mat_param)
        return circuit   


    @staticmethod
    def zzfeaturemap(n_wire, full_ent = True):
        zfm = ZZFeatureMap(feature_dimension=n_wire)
        return zfm   


    #cascade embedding
    @staticmethod
    def cascade(n_wire, full_ent = True):

        # Create a new circuit with two qubits
        qc = QuantumCircuit(n_wire)
        
        for i in range(n_wire):

            #add hadamrd
            qc.h(i)

            phi_name = 'phi_'+str(i)
            phi = Parameter(phi_name)
            qc.rx(phi, i)

            if(full_ent):
                qc.cx(i%n_wire, (i+1)%n_wire)

            #add hadamrd
            qc.h(i)
        
        return qc

    #embedding - encoded used in paper Hubregsten et all.
    @staticmethod
    def ansatz_encoded(n_wire, full_ent = True):

        # Create a new circuit with two qubits
        qc = QuantumCircuit(n_wire)
        
        for i in range(n_wire):
                    
            #add hadamrd
            qc.h(i)

        for i in range(n_wire):
            
            phi_name = 'phi_'+str(i)
            phi = Parameter(phi_name)
            qc.rz(phi, i)

        if(full_ent):
            for i in range(n_wire):            
                qc.cx(i%n_wire, (i+1)%n_wire)

        
        qc.barrier()    

        
        for i in range(n_wire):
            #add hadamrd
            qc.h(i)
        
        return qc
    
    @staticmethod
    def xyz_encoded(n_wire, full_ent = True):

        # Create a new circuit with two qubits
        qc = QuantumCircuit(n_wire)        
        

        for i in range(n_wire):            
            phi_name = 'phi_'+str(i)
            phi = Parameter(phi_name)
            qc.rx(phi, i) 
            qc.ry(phi, i)
            qc.rz(phi, i)     

        if(full_ent):
            for i in range(n_wire):
                qc.cx(i%n_wire, (i+1)%n_wire)    
      
        
        return qc 

    @staticmethod
    def x_encoded(n_wire, full_ent = True):

        # Create a new circuit with two qubits
        qc = QuantumCircuit(n_wire)        
        

        for i in range(n_wire):            
            phi_name = 'phi_'+str(i)
            phi = Parameter(phi_name)
            qc.rx(phi, i)      

        if(full_ent):
            for i in range(n_wire):
                qc.cx(i%n_wire, (i+1)%n_wire)    
      
        
        return qc 


    @staticmethod
    def y_encoded(n_wire, full_ent = True):

        # Create a new circuit with two qubits
        qc = QuantumCircuit(n_wire)        
        

        for i in range(n_wire):            
            phi_name = 'phi_'+str(i)
            phi = Parameter(phi_name)
            qc.ry(phi, i)

        if(full_ent):
            for i in range(n_wire):            
                qc.cx(i%n_wire, (i+1)%n_wire)        
      
        
        return qc


    @staticmethod
    def y_encoded_scaled(n_wire, full_ent = True):

        # Create a new circuit with two qubits
        qc = QuantumCircuit(n_wire)               

        for i in range(n_wire):
            
            phi_name = 'phi_'+str(i)
            phi = Parameter(phi_name)
            qc.ry(((phi + 1)/2) * np.pi, i)

        if(full_ent):
            for i in range(n_wire):
                qc.cx(i%n_wire, (i+1)%n_wire)        
      
        
        return qc




    @staticmethod
    def z_encoded(n_wire, full_ent = True):

        # Create a new circuit with two qubits
        qc = QuantumCircuit(n_wire)   
        

        for i in range(n_wire):            
            phi_name = 'phi_'+str(i)
            phi = Parameter(phi_name)
            qc.rz(phi, i) 

        if(full_ent):
            for i in range(n_wire):            
                qc.cx(i%n_wire, (i+1)%n_wire)       
        
        return qc