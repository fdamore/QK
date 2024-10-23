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
    def spiral_encoding(n_wire, n_windings, full_ent = True):   # im considering n_wire as number of qubits (not features) - Luca

        # Create a new circuit with two qubits
        qc = QuantumCircuit(n_wire)        
        
        # assuming phis normalized bw 0 and 1
        for i in range(n_wire):            
            phi_name = 'phi_'+str(i)
            phi = Parameter(phi_name)
            qc.ry(np.pi * phi, i)
            qc.rz(2 * np.pi * n_windings * phi, i)    # n_windings is the number of times the spiral makes a complete turn - Luca

        if(full_ent):
            for i in range(n_wire):
                qc.cx(i%n_wire, (i+1)%n_wire)    
    
        
        return qc 

    @staticmethod
    def uniform_bloch_encoding(n_wire, full_ent = True):   # requires half as much qubits like dense encoding! - L

        # Create a new circuit with two qubits
        qc = QuantumCircuit(n_wire)        
        
        # assuming phis, thetas normalized bw 0 and 1
        for i in range(n_wire):            
            theta_name = 'phi_'+str(2*i)
            theta = Parameter(theta_name)
            phi_name = 'phi_'+str(2*i+1)
            phi = Parameter(phi_name)
            qc.ry(2*np.arccos(np.sqrt(theta)), i)
            qc.rz(phi*np.pi, i)

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