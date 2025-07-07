from qiskit.circuit import Parameter
from qiskit.circuit.library import ZZFeatureMap, iqp
from qiskit.quantum_info import random_hermitian, random_statevector
from qiskit import QuantumCircuit
import numpy as np


class Circuits:

    
    @staticmethod
    def zzfeaturemap(n_wire, full_ent = True, param_prefix = 'phi'):
        zfm = ZZFeatureMap(feature_dimension=n_wire, parameter_prefix=param_prefix)

        return zfm

    @staticmethod
    def IQPfeaturemap(W):
        iqp_qc = iqp(interactions=W)     

        return iqp_qc      
    
    @staticmethod
    def IQPfeaturemapRH(n_wire, seed_ = 1234):
        mat_param = np.real(random_hermitian(n_wire,seed=seed_))
        circuit = iqp(mat_param)
        circuit.name = 'IQP'
        return circuit 


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
    def xyz_encoded(n_wire, full_ent = True, param_prefix = 'phi', dr_layers = 1, dr_sep = 'cnot'):
        
        '''
        XYZ encoded
        '''
        
        qc = QuantumCircuit(n_wire) 
        qc.name = 'XYZ'       
        
        param_dict = {}
        for i in range(n_wire):            
            phi_name = param_prefix + '_' + str(i)
            phi = Parameter(phi_name)
            qc.rx(phi, i) 
            qc.ry(phi, i)
            qc.rz(phi, i)
            if dr_layers > 1:
                param_dict[phi_name] = phi


        if dr_layers <= 1 and full_ent:
            for i in range(n_wire):
                qc.cx(i % n_wire, (i + 1) % n_wire)

        if dr_layers > 1:
            #qc.barrier()

            for layer in range(dr_layers-1):
                qc.barrier()  


                if dr_sep == 'cnot':
                    for i in range(n_wire):
                        qc.cx(i%n_wire, (i+1)%n_wire)
                elif dr_sep == 'h':
                    for i in range(n_wire):
                        qc.h(i)
                else:
                    raise ValueError("dr_sep must be either 'cnot' or 'swap'")

                qc.barrier()               


                for i in range(n_wire):
                    phi_name = param_prefix + '_' + str(i)
                    phi = param_dict[phi_name]
                    qc.rx(phi, i) 
                    qc.ry(phi, i)
                    qc.rz(phi, i)


      
        
        return qc
    
    @staticmethod
    def corr_encoded(n_wire, full_ent = True, param_prefix = 'phi'):
        
        '''
        CORR_XYZ encoded (work with COGITO database only)
        '''
        
        qc = QuantumCircuit(n_wire) 
        qc.name = 'CORR XYZ'       
        

        for i in range(n_wire):            
            phi_name = param_prefix + '_' + str(i)
            phi = Parameter(phi_name)
            qc.rx(phi, i) 
            qc.ry(phi, i)
            qc.rz(phi, i)
        
        #entanlgled using correlation
        qc.cx(0,2) #illuminance->lamps
        qc.cx(1,5) #blinds ->temps
        qc.cx(3,5) #rh -> temps
        qc.cx(4,3) #co2 -> rh


    @staticmethod
    def corr3_encoded(n_wire, full_ent = True, param_prefix = 'phi'):
        
        '''
        CORR_XYZ encoded (work with COGITO database only)
        '''
        
        qc = QuantumCircuit(n_wire) 
        qc.name = 'CORR XYZ'       
        

        for i in range(n_wire):            
            phi_name = param_prefix + '_' + str(i)
            phi = Parameter(phi_name)
            qc.rx(phi, i) 
            qc.ry(phi, i)
            qc.rz(phi, i)
        
        #entanlgled using correlation
        qc.cx(0,2) #illuminance->lamps
        #qc.cx(1,5) #blinds ->temps
        qc.cx(3,5) #rh -> temps
        qc.cx(4,3) #co2 -> rh

        return qc

    @staticmethod
    def anticorr3_encoded(n_wire, full_ent = True, param_prefix = 'phi'):
        
        '''
        CORR_XYZ encoded (work with COGITO database only)
        '''
        
        qc = QuantumCircuit(n_wire) 
        qc.name = 'ANTICORR XYZ'       
        

        for i in range(n_wire):            
            phi_name = param_prefix + '_' + str(i)
            phi = Parameter(phi_name)
            qc.rx(phi, i) 
            qc.ry(phi, i)
            qc.rz(phi, i)
        
        #entanlgled using correlation
        qc.cx(1,4) #co2->blinds        
        qc.cx(0,4) #illuminance->blinds
        qc.cx(2,3) #lamps->rh
        
        return qc

    @staticmethod
    def zy_decomposition(n_wire, full_ent = True, param_prefix = 'phi'):
        
        '''
        Z-Y decomposition (see Nielsen pag. 175. Therem 4.1)
        '''
        
        qc = QuantumCircuit(n_wire) 
        qc.name = 'Z-Y decomposition'       
        

        for i in range(n_wire):            
            phi_beta_name = param_prefix + 'beta' + '_' + str(i)
            phi_gamma_name = param_prefix + 'gamma' + '_' + str(i)
            phi_delta_name = param_prefix + 'delta' + '_' + str(i)

            phi_beta = Parameter(phi_beta_name)
            phi_gamma = Parameter(phi_gamma_name)
            phi_delta = Parameter(phi_delta_name)           
            
            qc.rz(phi_beta, i) 
            qc.ry(phi_gamma, i)
            qc.rz(phi_delta, i)     

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
    def uniform_bloch_encoding(n_wire, full_ent = False, param_prefix = 'phi', margin=.001):

        # Create a new circuit with two qubits
        qc = QuantumCircuit(n_wire)
        qc.name = 'UniformBloch'  

        phis = [Parameter(f"{param_prefix}_{i}") for i in range(2*n_wire)]
        # normalization: even index: 0<phi<1 ; odd index --> 0<phi<pi. original normalization is bw 0 and 2pi
        for i, phi in enumerate(phis):
            phi /= 2*np.pi     # normalize bw 0 and 1
            phi = margin + phi * (1 - 2*margin)    # slightly shrinks the interval to avoid 0 and 1
            if i%2 == 0:
                phi *= np.pi
            phis[i] = phi      # updating the phis

        for i in range(n_wire):            
            qc.ry(2 * (phis[2*i+1] ** (1/2.)).arccos(), i)
            qc.rz(phis[2*i] * np.pi, i)

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


    def IQP_HuangE2(n_wire, full_ent=True, param_prefix='phi'):
        """
        E2 encoding in Huang's 'Power of Data in QML', Appendix 12.A
        """
        qc = QuantumCircuit(n_wire)
        qc.name = 'IQP_Huang'


        phi_params = [Parameter(f"{param_prefix}_{i}") for i in range(n_wire)]

        for _ in range(2):
            qc.h(range(n_wire))
            for i in range(n_wire):            
                qc.rz(2 * phi_params[i], i)   

                for j in range(i):
                    qc.rzz(2 * phi_params[i] * phi_params[j], i, j)

        return qc


    def Trotter_HuangE3(n_wire, full_ent=True, T = 20, param_prefix='phi'):
        qc = QuantumCircuit(n_wire)
        qc.name = 'Trotter_Huang'

        t = n_wire/3.
        # initialization to Haar random state
        for i in range(n_wire):
            state = random_statevector(2)  # random qubit state, assigned to qubit i
            qc.initialize(state, i)

        phi_params = [Parameter(f"{param_prefix}_{i}") for i in range(n_wire)]

        for _ in range(T):
            for i in range(n_wire):
                qc.rzz(2 * t / T * phi_params[i], i, (i+1)%n_wire)
                qc.ryy(2 * t / T * phi_params[i], i, (i+1)%n_wire)
                qc.rxx(2 * t / T * phi_params[i], i, (i+1)%n_wire)

        return qc
