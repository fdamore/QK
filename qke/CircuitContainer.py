import numpy as np

import os
import time
import datetime
from qke.QMeasures import QMeasures
from qke.Circuits import Circuits

from functools import wraps

def singleton(cls):
    @wraps(cls)
    def _wrap(*args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = cls(*args, **kwargs)
        return cls._instance
    return _wrap


#singleton container for circuit
@singleton
class CircuitContainer:   

    #quantum circuit used to define feature map
    circuit = None

    #quantum template for the circuit
    template = None    

    #number of qubit
    nwire = None

    #list of observables
    obs = None

    #store computed feature map
    fm_dict = {}

    #measure function
    measure_fn = None    

    def __init__(self, nwire = 1, obs = ['Z'], full_ent = True, qtemplate = Circuits.ansatz_encoded, measure_fn = QMeasures.Aer):
        print('*** Create a Container ***')
        self.build(nwire=nwire, obs=obs,full_ent=full_ent, qtemplate=qtemplate, measure_fn= measure_fn)
    
    def build(self, nwire = 1, obs = ['Z'], full_ent = True, qtemplate = Circuits.ansatz_encoded, measure_fn = QMeasures.Aer):
        #define parameters
        self.obs = obs
        self.nwire = nwire 
        self.template = qtemplate
        self.full_ent =full_ent
        self.measure_fn = measure_fn

        self.circuit = self.template(self.nwire,  self.full_ent)

        #print metadata
        self.metadata()

        if len(self.obs) == 0:
            print('WARNING: provide observables')
        
        #clear the cache
        self.fm_dict.clear()


    def metadata(self):
        print(f'*** Quantum template for feature map using {str(self.nwire)} qubit ***')                
        print(self.circuit.draw())
        print(f'*** Required observables: {self.obs}')
        print(f'*** Measure procedure: {self.measure_fn.__name__}')
        return ""

    
    #save my feature map
    def save_feature_map(self, prefix = ''):
        #create a csv file with feature maps
        current_timestamp = time.time()
        datetime_object = datetime.datetime.fromtimestamp(current_timestamp)
        formatted_datetime = datetime_object.strftime("%Y%m%d%H%M%S")
        csv_file = '../qfm/' + prefix + str(formatted_datetime) + '.csv'        
        
        main_path = os.path.dirname(__file__)
        file_path = os.path.join(main_path, csv_file)

        #store the features map
        with open(file_path, 'w') as f:            
            f.write(','.join(['key', 'value']))
            f.write('\n') # Add a new line
            for i, (k,v) in enumerate(self.fm_dict.items()):
                l_item = [k,str(v)]
                f.write(','.join(l_item))
                f.write('\n') # Add a new line

        #print the related time stamp. 
        print(f'Timestamp of the file storing data: {formatted_datetime}')