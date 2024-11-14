from textwrap import wrap
from qiskit.circuit import Parameter
from qiskit.circuit.library import ZZFeatureMap
from qiskit.circuit.library import IQP
from qiskit.quantum_info import random_hermitian
from qiskit import QuantumCircuit
import numpy as np


class Circuits:

    
    @staticmethod
    def zzfeaturemap(n_wire, full_ent = True, param_prefix = 'phi'):
        zfm = ZZFeatureMap(feature_dimension=n_wire, parameter_prefix=param_prefix)

        return zfm   


    #cascade embedding
    @staticmethod
    def cascade(n_wire, full_ent = True, param_prefix = 'phi'):

        # Create a new circuit with two qubits
        qc = QuantumCircuit(n_wire)
        qc.name = 'Cascade'
        
        for i in range(n_wire):

            #add hadamrd
            qc.h(i)

            phi_name = param_prefix + '_'+str(i)
            phi = Parameter(phi_name)
            qc.rx(phi, i)

            if(full_ent):
                qc.cx(i%n_wire, (i+1)%n_wire)

            #add hadamrd
            qc.h(i)
        
        return qc

    #embedding - encoded used in paper Hubregsten et all.
    @staticmethod
    def ansatz_encoded(n_wire, full_ent = True, param_prefix = 'phi'):

        # Create a new circuit with two qubits
        qc = QuantumCircuit(n_wire)
        qc.name = 'AnsatzEncoded'
        
        for i in range(n_wire):
                    
            #add hadamrd
            qc.h(i)

        for i in range(n_wire):
            
            phi_name = param_prefix + '_' + str(i)
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
    def xyz_encoded(n_wire, full_ent = True, param_prefix = 'phi'):

        # Create a new circuit with two qubits
        qc = QuantumCircuit(n_wire) 
        qc.name = 'XYZ'       
        

        for i in range(n_wire):            
            phi_name = param_prefix + '_' + str(i)
            phi = Parameter(phi_name)
            qc.rx(phi, i) 
            qc.ry(phi, i)
            qc.rz(phi, i)     

        if(full_ent):
            for i in range(n_wire):
                qc.cx(i%n_wire, (i+1)%n_wire)    
      
        
        return qc 
    
    @staticmethod
    def spiral_encoding(n_wire, n_windings, full_ent = True, param_prefix = 'phi'):   # im considering n_wire as number of qubits (not features) - Luca

        # Create a new circuit with two qubits
        qc = QuantumCircuit(n_wire)
        qc.name = 'Spiral'        
        
        # assuming phis normalized bw 0 and 1
        for i in range(n_wire):            
            phi_name = param_prefix + '_' + str(i)
            phi = Parameter(phi_name)
            qc.ry(np.pi * phi, i)
            qc.rz(2 * np.pi * n_windings * phi, i)    # n_windings is the number of times the spiral makes a complete turn - Luca

        if(full_ent):
            for i in range(n_wire):
                qc.cx(i%n_wire, (i+1)%n_wire)    
    
        
        return qc 

    @staticmethod
    def uniform_bloch_encoding(n_wire, full_ent = True, param_prefix = 'phi'):   
        
        '''
        requires half as much qubits like dense encoding! - L

        '''

        # Create a new circuit with two qubits
        qc = QuantumCircuit(n_wire)
        qc.name = 'UniformBloch'        
        
        # assuming phis, thetas normalized bw 0 and 1
        for i in range(n_wire):            
            theta_name = param_prefix + '_'+str(2*i)
            theta = Parameter(theta_name)            
            phi_name = param_prefix + '_'+str(2*i+1)
            phi = Parameter(phi_name)
            #qc.ry(2*np.arccos(np.sqrt(theta)), i)             
            qc.ry(2*np.arccos(theta**(1/2)), i)            
            qc.rz(phi*np.pi, i)

        if(full_ent):
            for i in range(n_wire):
                qc.cx(i%n_wire, (i+1)%n_wire)
        
        return qc 
    
    @staticmethod
    def x_encoded(n_wire, full_ent = True, param_prefix = 'phi'):

        # Create a new circuit with two qubits
        qc = QuantumCircuit(n_wire) 
        qc.name = 'XEncoded'       
        

        for i in range(n_wire):            
            phi_name = param_prefix + '_'+str(i)
            phi = Parameter(phi_name)
            qc.rx(phi, i)      

        if(full_ent):
            for i in range(n_wire):
                qc.cx(i%n_wire, (i+1)%n_wire)    
      
        
        return qc 



  



