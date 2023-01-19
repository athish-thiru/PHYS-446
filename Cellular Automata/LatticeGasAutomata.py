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
    x2 = np.random.randint(0, 100)
    y2 = np.random.randint(0, 100)

    state[x1][y1], state[x2][y2] = state[x2][y2], state[x1][y1]

#bits_array stores size of state
bits_array = []

for i in range(10001):
    #snapshots
    if i%500 == 0:
        plt.imshow(state, cmap='gray')
        plt.title("Snapshot at sweep {}".format(i))
        plt.savefig("snapshots/sweep{}".format(i))
    
    #updating state and saving size of state
    swap(state)
    data = gzip.compress(state.tobytes())
    bits_array.append(sys.getsizeof(data))

#Final plot
plt.plot(bits_array)
plt.xlabel("Sweeps")
plt.ylabel("Number of compressed bits")
plt.title("Change in entropy for \n Lattice Gas Automata")
plt.ylim(0, 3000)
plt.savefig("EntropyPlot")

plt.show()
