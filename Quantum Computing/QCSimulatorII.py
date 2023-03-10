import DiracNotation as dn
import numpy as np
import matplotlib.pyplot as plt

#Control-NOT Gate
def CNOT(controlWire, notWire, inputState):
    outputState = []
    for element in inputState:
        (prob, state) = element
        #Converting string to list to modify individual elements
        stateList = list(state)
        if stateList[controlWire] == '1':
            if stateList[notWire] == '1':
                stateList[notWire] = '0'
            else:
                stateList[notWire] = '1'
        state = "".join(stateList)
        outputState.append((prob, state))
    return outputState

#Phase Gate
def Phase(wire, theta, inputState):
    outputState = []
    for element in inputState:
        (prob, state) = element
        if state[wire] == '1':
            prob *= (np.cos(theta) + np.sin(theta)*(1j))
        outputState.append((prob, state))
    return outputState

#Hadamard Gate
def H(wire, inputState):
    outputState = []
    for element in inputState:
        (prob, state) = element
        firstState, secondState = list(state), list(state)
        if firstState[wire] == '0':
            secondState[wire] = '1'
            outputState.append((prob/np.sqrt(2), "".join(firstState)))
            outputState.append((prob/np.sqrt(2), "".join(secondState)))
        else:
            secondState[wire] = '0'
            outputState.append((-prob/np.sqrt(2), "".join(firstState)))
            outputState.append((prob/np.sqrt(2), "".join(secondState)))
    #Removing duplicates
    #print("HADAMARD OUTPUT", outputState)
    trueOutputState = AddDuplicates(outputState)
    return trueOutputState

#Helper function for Hadamard Gate
def AddDuplicates(myState):
    outputState = []
    stateList = []
    for element in myState:
        (prob, state) = element
        # Checks if the state has been dealt with before
        if state in stateList:
            # Goees through output to find state and add probability
            for i in range(len(outputState)):
                newElementList = list(outputState[i])
                if (newElementList[1] == state):
                    #print("BEFORE: ", newElementList[0])
                    newElementList[0] += prob
                    #print("AFTER: ", newElementList[0])
                outputState[i] = tuple(newElementList)
                #print("NEW ELEMENT: ", newElement)
        else:
            stateList.append(state)
            outputState.append((prob, state))
    # Removes zero probability
    trueOutputState = []
    for element in outputState:
        if element[0] != 0:
            trueOutputState.append(element)
        
    return trueOutputState
    
# Deals with parsing input files
def ReadInput(fileName):
    myInput_lines=open(fileName).readlines()
    myInput=[]
    numberOfWires=int(myInput_lines[0])
    for line in myInput_lines[1:]:
        myInput.append(line.split())
    return (numberOfWires,myInput)

# Deals with parsing through state files
def ReadState(fileName, numberOfWires):
    lines=open(fileName).readlines()
    state = []
    for i in range(len(lines)):
        complexNumber = lines[i].split()
        prob = float(complexNumber[0]) + float(complexNumber[1])*(1j)
        basis = str(bin(i))[2:]
        while (len(basis) != numberOfWires):
            basis = '0' + basis
        state.append((prob, basis))
    return state

#Subsequent portion of code only runs when file is run as a script
if __name__ == "__main__":
    #Enter Input File
    numberOfWires, myInput = ReadInput("Input Files/measure.circuit")
    inputBasis = ""
    for i in range(numberOfWires):
        inputBasis += '0'
    circuitState = [((1+0j), inputBasis)]

    #Input
    if myInput[0][0] == "INITSTATE":
        if myInput[0][1] == "FILE":
            circuitState = ReadState("Input Files/" + myInput[0][2], numberOfWires)
        elif myInput[0][1] == "BASIS":
            basis = myInput[0][2][1:-1]
            circuitState = [((1+0j), basis)]
            

    #Gates
    for gate in myInput:
        if gate[0] == 'H':
            circuitState = H(int(gate[1]), circuitState)
        elif gate[0] == 'P':
            circuitState = Phase(int(gate[1]), float(gate[2]), circuitState)
        elif gate[0] == 'CNOT':
            circuitState = CNOT(int(gate[1]), int(gate[2]), circuitState)
    circuitVec = dn.StateToVec(circuitState)
    circuitProbVec = circuitVec * np.conj(circuitVec)
    circuitProbState = dn.VecToState(circuitProbVec)
    print(dn.PrettyPrintInteger(circuitProbState))

    #Measurement
    if myInput[-1][0] == 'MEASURE':
        probList = []
        basisList = []
        for element in circuitProbState:
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