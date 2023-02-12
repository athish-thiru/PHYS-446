import DiracNotation as dn
import numpy as np
import matplotlib.pyplot as plt
import QCSimulatorII as qcII

# Helper function to calculate tensor product
def tensorMe(listOfMatrices):
    output = np.kron(listOfMatrices[0], listOfMatrices[1])
    for matrix in listOfMatrices[2:]:
        output = np.kron(output, matrix)
    return output

# Hadamard Array
def HadamardArray(gate, numberOfWires):
    hadamard = 1/np.sqrt(2)*np.array([[1, 1], [1, -1]])
    identity = np.eye(2)
    matrixList = []
    for i in range(numberOfWires):
        if i == gate:
            matrixList.append(hadamard)
        else:
            matrixList.append(identity)
    output = tensorMe(matrixList)
    return output

# Phase Array
def PhaseArray(gate, theta, numberOfWires):
    phase = np.array([[1, 0], [0, np.cos(theta) + (1j)*np.sin(theta)]])
    identity = np.eye(2)
    matrixList = []
    for i in range(numberOfWires):
        if i == gate:
            matrixList.append(phase)
        else:
            matrixList.append(identity)
    output = tensorMe(matrixList)
    return output

# CNOT Array
def CNOTArray(controWire, notWire, numberOfWires):
    controlAbove = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    controlBelow = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
    identity = np.eye(2)
    matrixList = []
    gate = np.min([controWire, notWire])
    i = 0
    while i < numberOfWires:
        if (i == gate) and (controWire - notWire == -1):
            i += 2
            matrixList.append(controlAbove)
        elif (i == gate) and (controWire - notWire == 1):
            i += 2
            matrixList.append(controlBelow)
        else:
            matrixList.append(identity)
            i += 1
    output = tensorMe(matrixList)
    return output

# Quantum Computer Simulator 1a
def qcIa(circuitVec, myInput):
    circuitMatrix = np.eye(len(circuitVec))
    for gate in myInput:
        if gate[0] == 'H':
            gateArray = HadamardArray(int(gate[1]), numberOfWires)
            circuitMatrix = circuitMatrix @ gateArray
        elif gate[0] == 'P':
            gateArray = PhaseArray(int(gate[1]), float(gate[2]), numberOfWires)
            circuitMatrix = circuitMatrix @ gateArray
        elif gate[0] == 'CNOT':
            gateArray = CNOTArray(int(gate[1]), int(gate[2]), numberOfWires)
            circuitMatrix = circuitMatrix @ gateArray
    output = circuitVec @ circuitMatrix
    return output

# Quantum Computer Simulator 1b
def qcIb(circuitVec, myInput):
    for gate in myInput:
        if gate[0] == 'H':
            gateArray = HadamardArray(int(gate[1]), numberOfWires)
            circuitVec = circuitVec @ gateArray
        elif gate[0] == 'P':
            gateArray = PhaseArray(int(gate[1]), float(gate[2]), numberOfWires)
            circuitVec = circuitVec @ gateArray
        elif gate[0] == 'CNOT':
            gateArray = CNOTArray(int(gate[1]), int(gate[2]), numberOfWires)
            circuitVec = circuitVec @ gateArray
    return circuitVec

if __name__ == "__main__":
    #Enter Input File
    numberOfWires, myInput = qcII.ReadInput("Input Files/rand.circuit")
    circuitVec = np.zeros(2**numberOfWires)
    circuitVec[0] = 1

    #Input
    if myInput[0][0] == "INITSTATE":
        if myInput[0][1] == "FILE":
            circuitState = qcII.ReadState("Input Files/" + myInput[0][2], numberOfWires)
            circuitVec = dn.StateToVec(circuitState)
        elif myInput[0][1] == "BASIS":
            basis = myInput[0][2][1:-1]
            circuitVec[0] = 0
            index = 0 
            for i in range(len(basis)):
                index += int(basis[i])*(2**(len(basis) - i - 1))
            circuitVec[index] = 1
    
    circuitVec = qcIb(circuitVec, myInput)
    circuitVec = circuitVec*np.conj(circuitVec)
    circuitState = dn.VecToState(circuitVec)
    print(dn.PrettyPrintInteger(circuitState))

    #Measurement
    if myInput[-1][0] == 'MEASURE':
        probList = []
        basisList = []
        for element in circuitState:
            (prob, basis) = element
            basisList.append(basis)
            probList.append(prob)
        measurements = np.random.choice(basisList, size=10000, p=probList)
        #Plotting
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.hist(measurements, histtype='barstacked')
        ax.set_xlabel("Output")
        ax.set_ylabel("Count")

        for label in ax.get_xticklabels():
            label.set_rotation("vertical")

        fig.savefig("measure circuit result")
        plt.show()
