import QCSimulatorII as qc
import numpy as np
import DiracNotation as dn

# Not Gate
def NOT(wire, inputState):
    outputState = qc.H(wire, inputState)
    outputState = qc.Phase(wire, np.pi, outputState)
    outputState = qc.H(wire, outputState)
    return outputState

# RZ Gate
def RZ(wire, theta, inputState):
    outputState = qc.Phase(wire, theta/2, inputState)
    outputState = qc.H(wire, outputState)
    outputState = qc.Phase(wire, np.pi, outputState)
    outputState = qc.H(wire, outputState)
    outputState = qc.Phase(wire, -theta/2, outputState)
    outputState = qc.H(wire, outputState)
    outputState = qc.Phase(wire, np.pi, outputState)
    outputState = qc.H(wire, outputState)
    return outputState

# Control RZ Gate
def CRZ(controlWire, rzWire, theta, inputState):
    outputState = qc.Phase(rzWire, theta/2, inputState)
    outputState = qc.CNOT(controlWire, rzWire, outputState)
    outputState = qc.Phase(rzWire, -theta/2, outputState)
    outputState = qc.CNOT(controlWire, rzWire, outputState)
    return outputState

# Control Phase Gate
def CPHASE(controlWire, phaseWire, theta, inputState):
    outputState = CRZ(controlWire, phaseWire, theta, inputState)
    outputState = qc.Phase(controlWire, theta/2, outputState)
    return outputState

# Swap Gate
def SWAP(firstWire, secondWire, inputState):
    outputState = qc.CNOT(firstWire, secondWire, inputState)
    outputState = qc.CNOT(secondWire, firstWire, outputState)
    outputState = qc.CNOT(firstWire, secondWire, outputState)
    return outputState

if __name__ == "__main__":
    #Enter Input File
    numberOfWires, myInput = qc.ReadInput("Input Files/nonAtomicGates.circuit")
    inputBasis = ""
    for i in range(numberOfWires):
        inputBasis += '0'
    circuitState = [((1+0j), inputBasis)]

    #Input
    if myInput[0][0] == "INITSTATE":
        if myInput[0][1] == "FILE":
            circuitState = qc.ReadState("Input Files/" + myInput[0][2], numberOfWires)
        elif myInput[0][1] == "BASIS":
            basis = myInput[0][2][1:-1]
            circuitState = [((1+0j), basis)]
            

    #Gates
    for gate in myInput:
        if gate[0] == 'H':
            circuitState = qc.H(int(gate[1]), circuitState)
        elif gate[0] == 'P':
            circuitState = qc.Phase(int(gate[1]), float(gate[2]), circuitState)
        elif gate[0] == 'CNOT':
            circuitState =qc. CNOT(int(gate[1]), int(gate[2]), circuitState)
        elif gate[0] == 'NOT':
            circuitState = NOT(int(gate[1]), circuitState)
        elif gate[0] == 'RZ':
            circuitState = RZ(int(gate[1]), float(gate[2]),circuitState)
        elif gate[0] == 'CRZ':
            circuitState = CRZ(int(gate[1]), int(gate[2]), float(gate[3]), circuitState)
        elif gate[0] == 'CPHASE':
            circuitState = CPHASE(int(gate[1]), int(gate[2]), float(gate[3]), circuitState)
        elif gate[0] == 'SWAP':
            circuitState = SWAP(int(gate[1]), int(gate[2]), circuitState)
    print(dn.PrettyPrintInteger(circuitState))