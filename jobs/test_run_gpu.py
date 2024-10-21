from qiskit.quantum_info import SparsePauliOp
from qiskit import QuantumCircuit
from pqk.QMeasures import QMeasures

if __name__ == '__main__':

    qc=QuantumCircuit(2)
    qc.h(0)
    qc.cx(0,1)
    print(qc)

    obs=SparsePauliOp(["ZZ"])
    ##without noise
    res=QMeasures.GPUAerStateVectorEstimator(qc,obs)
    print(res)

    ##with simulated noise
    res = QMeasures.GPUAerVigoNoiseStateVectorEstimator(qc, obs)
    print(res)