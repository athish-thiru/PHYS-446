import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import makeImage

# Hopfield Network Class
class HopfieldNetwork:
    # Initializes class
    def __init__(self, size):
        #Stores size for uodate function
        self.size = size
        # The state of each "neuron" is either 1 or -1
        self.states = np.random.choice([-1, 1], size)
        # Creates a 1d array of biases for each "neuron"
        self.biases = np.random.default_rng().uniform(-1, 1, size)
        # Creates a 2d array of weights for each neuron pair
        # Array is symmetric since Hopfield network edges are undirected
        assymWeigths = np.random.default_rng().uniform(-1, 1, (size, size))
        self.weights = np.maximum(assymWeigths, assymWeigths.T)
    
    # Takes in a series of images as inputs to update the Weigth matrix to store memories
    def setWeightMatrix(self, listOfMemories):
        weights = np.outer(listOfMemories[0], listOfMemories[0])
        for i in range(1, len(listOfMemories)):
            weights += np.outer(listOfMemories[i], listOfMemories[i])
        weights = weights/len(listOfMemories)
        self.weights = weights
    
    # Gets state of network at idx
    def getState(self, idx):
        return self.states[idx]
    
    # Checks if state is valid and changes it
    def changeState(self, idx):
        self.states[idx] = -self.states[idx]

    # Get the entire the state vector
    def getStates(self):
        return self.states
    
    # Sets the entire the state vector
    def setStates(self, states):
        self.states = states
    
    # Calculates weight for a given index
    def calculateWeight(self, index):
        incomingWeight = 0
        for i in range(self.size):
            if i != index:
                incomingWeight += self.states[i] * self.weights[i, index]
        return incomingWeight
    
    #Updates a random state in the network
    def update(self):
        #index to update
        idx = np.random.choice(self.size)
        #Calculates incoming Weight
        incomingWeight = self.calculateWeight(idx)
        # Rule for updating weight
        if incomingWeight > self.biases[idx]:
            self.states[idx] = 1
        else:
            self.states[idx] = -1
    
    # Checks whether the Hopfield network has converged
    # Does so by seeing if any state wants to be updated
    def checkConvergence(self):
        # Goes through every state; if states wants to be updated return False
        # If no state wants to be updated returns True
        for i in range(self.size):
            incomingWeight = self.calculateWeight(i)
            if ((incomingWeight > self.biases[i]) and (self.states[i] == -1))\
                or ((incomingWeight <= self.biases[i]) and (self.states[i] == 1)):
                return False
        return True
    
    # Caluclates the total energy associated with the Hopfield Network
    def calculateEnergy(self):
        return (-0.5*self.states@self.weights@self.states.T) + np.sum(self.states*self.biases)


# Creates 10 random hopfield networks of the same size and
# sees how long they take to converge
def PlotConvergence():
    fig, ax = plt.subplots(1, 1)
    for i in range(10):
        energyList = []
        hn = HopfieldNetwork(100)
        initialEnergy = hn.calculateEnergy()
        energyList.append(initialEnergy)
        flag = True
        while (flag):
            for j in range(100):
                hn.update()
            energyList.append(hn.calculateEnergy())
            if hn.checkConvergence():
                break
        energyList.append(hn.calculateEnergy())
        ax.plot(energyList)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Energy")
    ax.set_title("Convergence of 10 random Hopfield Networks")
    fig.savefig("Plots/Convergence of Hopfield Networks")

# Make a face and a tree using makeImage.py and saves the memories as weights
# Then checks if the Hopfield Network can recover the face
def TestRecovery():
    # Creating initial memories vector
    face = makeImage.MakeFace()
    face = np.where(face == 0, -1, 1)
    tree = makeImage.MakeTree()
    tree = np.where(tree == 0, -1, 1)
    memories = [face.flatten(), tree.flatten()]

    # Creating pertubed face image for initial state of Hopfield Network
    flatFace = face.flatten()
    fig, ax = plt.subplots(nrows=1, ncols=1)
    for i in range(flatFace.size//4):
        ch = np.random.choice(flatFace.size)
        flatFace[ch] = -flatFace[ch]
    ax.matshow(flatFace.reshape((10, 10)))
    ax.set_title("Initial Perturbed Image")
    fig.savefig("Plots/Initial perturbed Image")

    # Creating Hopfield Network
    hn = HopfieldNetwork(100)
    hn.setWeightMatrix(memories)
    hn.setStates(flatFace)
    currentEnergy = hn.calculateEnergy()
    prevEnergy = 0
    k = 0
    while (currentEnergy != prevEnergy):
        # Update Hopfield network
        for j in range(flatFace.size):
            hn.update()
        # Just to get an intermediate image
        k += 1
        if (k == 1):
            fig2, ax2 = plt.subplots(nrows=1, ncols=1)
            ax2.matshow(hn.getStates().reshape(10, 10))
            ax2.set_title("Intermediate Image")
            fig2.savefig("Plots/Intermediate Image")

        # Updating exit conditions
        prevEnergy = currentEnergy
        currentEnergy = hn.calculateEnergy()
    
    fig3, ax3 = plt.subplots(nrows=1, ncols=1)
    ax3.matshow(hn.getStates().reshape(10, 10))
    ax3.set_title("Final Image upon Convergence")
    fig3.savefig("Plots/Final Image upon Convergence")

# Measures the Hamming distance for different combinations of memories and pertubations
# Size is the size of each memory
# NumMemories is the number of memories used to create the weigth for the Hopfield Network
# NumPertubations is the number of pertubations performed on the randomly picked memory
def measureHammingDistance(size, NumMemories, NumPertubations):
    L = size
    # Create Hamming Distance Arrray
    hammingDistanceArray = np.zeros((NumPertubations, NumMemories))
    for k in range(1, NumPertubations+1):
        for p in range(1, NumMemories+1):
            print("K: ", k, " P: ", p)
            # Create Hopfield Network
            hn = HopfieldNetwork(L*L)

            # Add p memories to Hopfield Network
            listOfMemories = []
            for i in range(p):
                listOfMemories.append(np.random.choice([-1, 1], size=(L, L)))
            hn.setWeightMatrix(listOfMemories)

            # Pick one random memory and perturb it k times
            ch = np.random.choice(p)
            perturbedMemory = listOfMemories[ch].flatten()
            indices = np.random.choice(L*L, k+1)
            perturbedMemory[indices] = -perturbedMemory[indices]
            hn.setStates(perturbedMemory)

            # Run Hopfield Network until convergence and calculate Hamming Distance
            currentEnergy = hn.calculateEnergy()
            prevEnergy = 0
            while (currentEnergy != prevEnergy):
                # Update Hopfield network
                for j in range(L*L):
                    hn.update()

                # Updating exit conditions
                prevEnergy = currentEnergy
                currentEnergy = hn.calculateEnergy()
            convergedMemory = hn.getStates()
            hammingDistance = np.sum(np.abs(convergedMemory - listOfMemories[ch].flatten())/2)
            print("Hamming Distance:", hammingDistance)
            hammingDistanceArray[k-1, p-1] = hammingDistance

    # Plot Hamming Distance Array
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.matshow(hammingDistanceArray)
    ax.set_xlabel("Number of Memories")
    ax.set_ylabel("Number of Perturbed Bits")
    cmap = mpl.cm.viridis
    norm = mpl.colors.Normalize(vmin=hammingDistanceArray.min(), vmax=hammingDistanceArray.max())
    fig.colorbar(mpl.cm.ScalarMappable(norm, cmap) , ax=ax)
    fig.suptitle("Hamming Distance for each combination of \n Pertubations and Memories")
    fig.savefig("Plots/Hamming Distance Plot")

# Calculates the Energy Landscape for a Hopfield Network
def createEnergyLandscape(numNeurons, numMemories):
    # Creating array for powers of 2 for binary-int conversion
    powersOfTwo = []
    for i in range(numNeurons):
        powersOfTwo.append(2**(numNeurons-i-1))
    powersOfTwo = np.array(powersOfTwo)
    # Initialize Hopfield Network and energies list and configurations dictionary
    hn = HopfieldNetwork(numNeurons)
    energies = []
    configurations = {}
    # Create 2 random memories to save as weights
    listofMemories = []
    for i in range(numMemories):
        listofMemories.append(np.random.choice([1, -1], size=numNeurons))
    print(listofMemories)
    hn.setWeightMatrix(listofMemories)
    # Start with the state corresponding to the binary number 0 and calculate energy
    # Going through every possible state
    for state in range(2**numNeurons):
        binRep = str(bin(state)[2:])
        # Makes every string length numNeurons with added 0s
        while (len(binRep) != numNeurons):
            binRep = "0" + binRep
        
        # Creating initial state
        initState = np.where(np.array([int(elem) for elem in binRep]) == 0, -1, 1)
        hn.setStates(initState)
        initEnergy = hn.calculateEnergy()
        energies.append(initEnergy)
        configurations[state] = []

        # Going to all possible connections to find lower energies states and adding those states to configuration
        for j in range(numNeurons):
            hn.changeState(numNeurons-j-1)
            nextState = hn.getStates()
            nextStateNum = np.sum(np.where(nextState == -1, 0, 1)*powersOfTwo)
            nextEnergy = hn.calculateEnergy()
            if (nextEnergy < initEnergy):
                configurations[state].append(nextStateNum)
            # Change state back
            hn.changeState(numNeurons-j-1)
        # Remove key if list is empty
        if not configurations[state]:
            configurations.pop(state)

    # Plotting Energy Landscape
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(np.arange(2**numNeurons), energies, ".")
    ax.set_xlabel("States")
    ax.set_ylabel("Energy")
    for state, connections in configurations.items():
        for nextState in connections:
            ax.plot([state, nextState], [energies[state], energies[nextState]], "--")
    fig.suptitle("Energy Landscape")
    fig.savefig("Plots/Energy Landscape")
    plt.show()

    # Writing file for graphviz representation
    file = open("landscape.digraph", "w")
    file.write("digraph G { \n")
    # Color the lowest states red
    for memory in listofMemories:
        stateNum = np.sum(np.where(memory == -1, 0, 1)*powersOfTwo)
        file.write(str(stateNum) + " [shape=circle, style=filled, fillcolor=red] \n")
    # Create connections for every state
    for state, connections in configurations.items():
        connectionsString = " ".join([str(elem) for elem in connections])
        file.write(str(state) + " -> {" + connectionsString + "}; \n")
    file.write("}")
    file.close()


if __name__ == "__main__":
    #createEnergyLandscape(7, 2)
    pass