#Run script from inside Cellular Automata folder or update paths
import matplotlib.pyplot as plt
import numpy as np
import gzip
import sys

#Creating Initial state
#Here 1 is white/empty and 0 is black/gas
state = np.zeros((100, 100))
state[0:100, 0:100] = 1
state[0:100, 0:50] = 0

#Rule
def swap(state):
    x1 = np.random.randint(0, 100)
    y1 = np.random.randint(0, 100)
    ch = np.random.randint(0, 4)

    if (ch == 0) and (x1 != 0): #Swap up
        state[x1][y1], state[x1-1][y1] = state[x1-1][y1], state[x1][y1]
        return True
    if (ch == 1) and (x1 != 99): #Swap down
        state[x1][y1], state[x1+1][y1] = state[x1+1][y1], state[x1][y1]
        return True
    if (ch == 2) and (y1 != 0): #Swap left
        state[x1][y1], state[x1][y1-1] = state[x1][y1-1], state[x1][y1]
        return True
    if (ch == 3) and (y1 != 99): #Swap right
        state[x1][y1], state[x1][y1+1] = state[x1][y1+1], state[x1][y1]
        return True
    return False

#bits_array stores size of state
bits_array = []

for i in range(10001):
    #Each sweep
    j = 0
    while(j < 10000):
        if (swap(state)):
            j += 1
    
    #updating state and saving size of state
    data = gzip.compress(state.tobytes())
    bits_array.append(sys.getsizeof(data))

    #snapshots
    if i%500 == 0:
        plt.imshow(state, cmap='gray')
        plt.title("Snapshot at sweep {}".format(i))
        plt.savefig("snapshots/sweep{}".format(i))

#Final plot
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(1, 1, 1)
ax.plot(bits_array)
ax.set_xlabel("Sweeps")
ax.set_ylabel("Number of compressed bits")
ax.set_title("Change in entropy for \n Lattice Gas Automata")
ax.set_ylim(0, 3000)
fig.savefig("EntropyPlot")