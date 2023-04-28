import numpy as np
import matplotlib.pyplot as plt

# Make Labels for the Distorted Hydrogen Chain
# n is the number of unit cells
def DHCLabels1D(n):
    output = []
    for i in range(n):
        output.append((i, "a"))
        output.append((i, "b"))
    return output

# Make Labels for the Hydrogen Chain
# n is the number of unit cells
def HCLabels1D(n):
    output = []
    for i in range(n):
        output.append((i, "a"))
    return output

# Change from the Label notation to location from 0
def LabelToR1D(label):
    unitcell, atom = label
    location = unitcell if atom == "a" else unitcell + 0.8
    return location

# Change from the Label notation to index in list
def LabelToIndex1D(label):
    unitcell, atom = label
    index = 2*(unitcell) if atom == "a" else 2*(unitcell)+1
    return index

# Make momentum labels for Hydrogen chain
# N is the number of unit cells
def momentumLabelsHC1D(N):
    output = []
    for i in range(N):
        output.append((i, "a"))
    return output

# Make momentum labels for Distorted Hydrogen chain
# N is the number of unit cells
def momentumLabelsDHC1D(N):
    output = []
    for i in range(N):
        output.append((i, "a"))
        output.append((i, "b"))
    return output

# Converts the label into its position in momentum space
# N is the number of unit cells
def momentumLabelToK1D(label, N):
    # Defining the unit vectors
    b1 = (2*np.pi)/N
    # Calculating the vector distance
    unitcell, _ = label
    location = unitcell*b1
    return location

# Make Labels for the Distorted Hydrogen Chain
# M is the number of horizontal unit cells
# N is the number of vertical unit cells
def DHCLabels2D(M, N):
    output = []
    for i in range(M):
        for j in range(N):
            output.append(((i, j), "a"))
            output.append(((i, j), "b"))
    return output

# Make Labels for the Hydrogen Chain
# M is the number of horizontal unit cells
# N is the number of vertical unit cells
def HCLabels2D(M, N):
    output = []
    for i in range(N):
        for j in range(M):
            output.append(((i, j), "a"))
    return output

# Change from the Label notation to location from 0
def LabelToR2D(label):
    a1, a2 = (1, 0), (0.5, np.sqrt(3)/2)
    unitcell, atom = label
    location1 = unitcell[0]*a1[0] + unitcell[1]*a2[0]
    location2 = unitcell[1]*a2[1] if atom == "a" else unitcell[1]*a2[1] + (1/np.sqrt(3))
    location = [location1, location2]
    return location

# Change from the Label notation to index in list
# N is the number of vertical unit cells
def LabelToIndex2D(label, N):
    unitcell, atom = label
    index = 2*N*unitcell[0] + 2*unitcell[1] if atom == "a" else 2*N*unitcell[0] + 2*unitcell[1] + 1
    return index

# Make momentum labels for Hydrogen chain
# M is the number of horizontal unit cells
# N is the number of vertical unit cells
def momentumLabelsHC2D(M, N):
    output = []
    for i in range(M):
        for j in range(N):
            output.append(((i, j), "a"))
    return output

# Make momentum labels for Distorted Hydrogen chain
# M is the number of horizontal unit cells
# N is the number of vertical unit cells
def momentumLabelsDHC2D(M, N):
    output = []
    for i in range(M):
        for j in range(N):
            output.append(((i, j), "a"))
            output.append(((i, j), "b"))
    return output

# Converts the label into its position in momentum space
# M is the number of horizontal unit cells
# N is the number of vertical unit cells
def momentumLabelToK2D(label, M, N):
    # Defining the unit vectors
    b1, b2 = ((1)*2*np.pi/M, (-1/np.sqrt(3))*2*np.pi/N), (0, (2/np.sqrt(3))*2*np.pi/N)
    # Calculating the vector distance
    unitcell, _ = label
    location1 = unitcell[0]*b1[0] 
    location2 = unitcell[0]*b1[1] + unitcell[1]*b2[1]
    location = [location1, location2]
    return location

if __name__ == "__main__":
    labels = DHCLabels2D(3, 4)
    r = np.array([LabelToR2D(label) for label in labels])
    klabels = momentumLabelsDHC2D(3, 4)
    k = np.array([momentumLabelToK2D(label, 3, 4) for label in klabels])
    plt.plot(k[:, 0], k[:, 1], ".")
    plt.show()