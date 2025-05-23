import numpy as np

from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit.primitives import Estimator as PrimitiveEstimator 
from qiskit.primitives import StatevectorEstimator



class QMeasures:
    
    #measure using Aer
    @staticmethod
    def Aer(qc, observables,**kargs):    
        obs = [SparsePauliOp(label) for label in observables]
    
        estimator = AerEstimator() 
        estimator.options.default_precision = 0  

        obs = [
            observable.apply_layout(qc.layout) for observable in obs
        ]        
        
        # One pub, with one circuit to run against observables.
        job = estimator.run([(qc, obs)])
        
        # This is the result of the entire submission.  We submitted one Pub,
        # so this contains one inner result (and some metadata of its own).
        job_result = job.result()     

        return job_result[0].data.evs
    
    #measure using primitive estimator
    @staticmethod
    def PrimitiveEstimator(qc, observables, **kargs):

        #option for primitive
        my_options=None

        #get the number of shots
        nshots = kargs.get('nshots')
        if nshots is None or type(nshots) is not int:            
            nshots = 100
        
        #get the number of shots
        seed = kargs.get('seed')
        if seed is None or type(seed) is not int:
            my_options={'shots':nshots}
        else:             
            my_options={'shots':nshots, 'seed':seed}                    

        
        estimator = PrimitiveEstimator(options=my_options)         

        l = []         

        for itm in observables:
            job = estimator.run(qc, itm)
            job_result = job.result()
            l.append(job_result.values[0])   

        #return job_result[0].data.evs
        return np.array(l)

    #measure using state vector (evs is the expectation values of the measure)
    def StateVectorEstimator(qc, observables,**kargs):         
        
        estimator = StatevectorEstimator(default_precision=0)     

        obs = [SparsePauliOp(label) for label in observables]

        pub = (qc, obs)
        job = estimator.run([pub])
        result = job.result()[0]
        return result.data.evs

    def GPUAerStateVectorEstimator(qc, observables, **kargs):

        default_precision=0.0
        backend_options={
            "method":"statevector",
            "device":"GPU"
        }
        estimator=AerEstimator(
            options={
                "backend_options":backend_options,
                "default_precision":default_precision
            }
        )

        obs = [SparsePauliOp(label) for label in observables]

        pub = (qc, obs)
        job = estimator.run([pub])
        result = job.result()[0]
        return result.data.evs

    def CPUAerStateVectorEstimator(qc, observables, **kargs):

        default_precision=0.0
        backend_options={
            "method":"statevector",
            "device":"CPU"
        }
        estimator=AerEstimator(
            options={

                "backend_options":backend_options,
                "default_precision":default_precision
            }
        )

        obs = [SparsePauliOp(label) for label in observables]

        pub = (qc, obs)
        job = estimator.run([pub])
        result = job.result()[0]
        return result.data.evs

    def GPUAerVigoNoiseStateVectorEstimator(qc, observables, **kargs):
        from qiskit_ibm_runtime.fake_provider import FakeVigoV2
        from qiskit_aer.noise import NoiseModel
        fake_backend = FakeVigoV2()
        noise_model = NoiseModel.from_backend(fake_backend)

        default_precision = 0.0
        backend_options = {
            "method": "statevector",
            "device": "GPU",
            "noise_model": noise_model
        }

        run_options={};
        if kargs.get("seed_simulator") is not None:
            run_options["seed_simulator"] =kargs["seed_simulator"]

        estimator = AerEstimator(
            options={
                "run_options": run_options,
                "backend_options": backend_options,
                "default_precision": default_precision
            }
        )

        obs = [SparsePauliOp(label) for label in observables]

        pub = (qc, obs)
        job = estimator.run([pub])
        result = job.result()[0]
        return result.data.evs

    def CPUAerVigoNoiseStateVectorEstimator(qc, observables, **kargs):
        from qiskit_ibm_runtime.fake_provider import FakeVigoV2
        from qiskit_aer.noise import NoiseModel
        fake_backend = FakeVigoV2()
        noise_model = NoiseModel.from_backend(fake_backend)

        default_precision = 0.0
        backend_options = {
            "method": "statevector",
            "device": "CPU",
            "noise_model": noise_model
        }
        run_options = {};
        if kargs.get("seed_simulator") is not None:
            run_options["seed_simulator"] = kargs["seed_simulator"]

        estimator = AerEstimator(
            options={
                "run_options": run_options,
                "backend_options": backend_options,
                "default_precision": default_precision
            }
        )

        obs = [SparsePauliOp(label) for label in observables]

        pub = (qc, obs)
        job = estimator.run([pub])
        result = job.result()[0]
        return result.data.evs

    def CPUAerBrisbaneNoiseStateVectorEstimator(qc, observables, **kargs):
        from qiskit_ibm_runtime.fake_provider import FakeBrisbane
        from qiskit_aer.noise import NoiseModel
        fake_backend = FakeBrisbane()
        noise_model = NoiseModel.from_backend(fake_backend)

        default_precision = 0.0
        backend_options = {
            "method": "statevector",
            "device": "CPU",
            "noise_model": noise_model
        }
        run_options = {};
        if kargs.get("seed_simulator") is not None:
            run_options["seed_simulator"] = kargs["seed_simulator"]

        estimator = AerEstimator(
            options={
                "run_options": run_options,
                "backend_options": backend_options,
                "default_precision": default_precision
            }
        )

        obs = [SparsePauliOp(label) for label in observables]

        pub = (qc, obs)
        job = estimator.run([pub])
        result = job.result()[0]
        return result.data.evs

    def GPUAerBrisbaneNoiseStateVectorEstimator(qc, observables, **kargs):
        from qiskit_ibm_runtime.fake_provider import FakeBrisbane
        from qiskit_aer.noise import NoiseModel
        fake_backend = FakeBrisbane()
        noise_model = NoiseModel.from_backend(fake_backend)

        default_precision = 0.0
        backend_options = {
            "method": "statevector",
            "device": "GPU",
            "noise_model": noise_model
        }

        run_options={};
        if kargs.get("seed_simulator") is not None:
            run_options["seed_simulator"] =kargs["seed_simulator"]

        estimator = AerEstimator(
            options={
                "run_options": run_options,
                "backend_options": backend_options,
                "default_precision": default_precision
            }
        )

        obs = [SparsePauliOp(label) for label in observables]

        pub = (qc, obs)
        job = estimator.run([pub])
        result = job.result()[0]
        return result.data.evs