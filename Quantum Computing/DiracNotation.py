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
        # Converting complex amplitudes into real number
        if probability < 0:
            probability = -np.sqrt(probability*np.conj(probability))
        else:
            probability = np.sqrt(probability*np.conj(probability))
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
        # Converting complex amplitudes into real number
        if probability < 0:
            probability = -np.sqrt(probability*np.conj(probability))
        else:
            probability = np.sqrt(probability*np.conj(probability))
        #Chaning binary to integer
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
    myState = []
    for idx in range(len(myVec)):
        if myVec[idx] != (0+0.j):
            state = str(bin(idx))[2:]
            #Adding zeros so all states have equal size
            while (len(state) != 3): 
                state = '0' + state
            myState.append((myVec[idx], state))
    return myState


myState = [(np.sqrt(0.1), '00'), (np.sqrt(0.4), '01'), (-np.sqrt(0.5), '11')]
myState2 = [(np.sqrt(0.1)*1.j, '101'), (np.sqrt(0.5), '000') , (-np.sqrt(0.4), '010')]

# print(PrettyPrintBinary(myState))
# print(PrettyPrintInteger(myState))
# print(PrettyPrintBinary(myState2))
# print(PrettyPrintInteger(myState2))
print(StateToVec(myState2))
print(VecToState(StateToVec(myState2)))