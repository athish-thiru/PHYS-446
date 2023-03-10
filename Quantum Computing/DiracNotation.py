import numpy as np

def PrettyPrintBinary(myState):
    #Sorting myState using bubblesort
    for i in range(len(myState)):
        for j in range(i, len(myState)):
            if (myState[i][1] > myState[j][1]):
                myState[i], myState[j] = myState[j], myState[i]
            
    output = "("
    for currentState in myState:
        (probability, state) = currentState
        #Outputing into string form
        output += " " + str(probability) + " |" + state + "> + "
    #Removing last + sign
    output = output[0:-3] + ")"
    return output

def PrettyPrintInteger(myState):
    #Sorting myState using bubblesort
    for i in range(len(myState)):
        for j in range(i, len(myState)):
            if (myState[i][1] > myState[j][1]):
                myState[i], myState[j] = myState[j], myState[i]

    output = "("
    for currentState in myState:
        (probability, state) = currentState
        #Changing binary to integer
        intstate = 0
        for i in range(len(state)):
            intstate += int(state[i])*(2**(len(state) - i - 1))
        #Outputing into string form
        output += " " + str(probability) + " |" + str(intstate) + "> + "
    #Removing last + sign
    output = output[0:-3] + ")"
    return output

def StateToVec(myState):
    #Creating output vector
    myVec = np.zeros(np.power(2, len(myState[0][1])), dtype=complex)

    #Going through each state to modify myVec
    for currentState in myState:
        (probability, state) = currentState
        #Changing binary to integer to access index
        index = 0 
        for i in range(len(state)):
            index += int(state[i])*(2**(len(state) - i - 1))
        myVec[index] = probability

    return myVec

def VecToState(myVec):
    size = np.log2(len(myVec))
    myState = []
    for idx in range(len(myVec)):
        if myVec[idx] != (0+0.j):
            state = str(bin(idx))[2:]
            #Adding zeros so all states have equal size
            while (len(state) != size): 
                state = '0' + state
            myState.append((myVec[idx], state))
    return myState

if __name__ == "__main__":
    myState2 = [(np.sqrt(0.1)*1.j, '101'), (np.sqrt(0.5), '000') , (-np.sqrt(0.4), '010')]

    print(PrettyPrintBinary(myState2))
    print(PrettyPrintInteger(myState2))
    print(StateToVec(myState2))
    print(VecToState(StateToVec(myState2)))
# Add code to avoid neglibible values