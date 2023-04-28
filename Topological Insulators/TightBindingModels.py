import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import Lattice as lt

# Activate LaTeX
plt.rcParams['text.usetex'] = True

# Generates the real space hamiltonian for the Hydrogen Chain
# N is number of rows and columns
def HCHamiltonianReal(N):
    hamiltonian = np.zeros((N, N))
    # Nearest neghbours are have some interaction energy
    for i in range(N):
        hamiltonian[i, (i+1)%N] = -1
        hamiltonian[(i+1)%N, i] = -1
    
    return hamiltonian

# Generates the fourier tranform that takes your real space hamiltonian to momentum space
# Works on the Hydrogen Chain Hamiltonian and matrix is of size NxN
# N is number of rows and columns
def HCFourierMatrix(N):
    matrix = np.zeros((N, N), dtype=complex)
    realSpaceLabels = lt.HCLabels1D(N)
    momentumSpaceLabels = lt.momentumLabelsHC1D(N)
    for a in range(N):
        for b in range(N):
            r = lt.LabelToR1D(realSpaceLabels[a])
            k = lt.momentumLabelToK1D(momentumSpaceLabels[b], N)
            matrix[a, b] = (1/(N**0.5))*np.exp(k*r*1j)
    return matrix

# Generates the real space hamiltonian for the Distored Hydrogen Chain
# N is number of rows and columns
def DHCHamiltonianReal(N):
    hamiltonian = np.zeros((N, N))
    # Nearest neghbours are have some interaction energy
    for i in range(N//2):
        #Rule 1
        hamiltonian[2*i, (2*i+1)%N] = -1
        hamiltonian[(2*i+1)%N, 2*i] = -1
        # Rule 2
        hamiltonian[(2*i+1)%N, (2*i+2)%N] = -0.1
        hamiltonian[(2*i+2)%N, (2*i+1)%N] = -0.1
    
    return hamiltonian

# Generates the fourier tranform that takes your real space hamiltonian to momentum space
# Works on the Distorted Hydrogen Chain Hamiltonian and matrix is of size NxN
# N is number of rows and columns
def DHCFourierMatrix(N):
    matrix = np.zeros((N, N), dtype=complex)
    realSpaceLabels = lt.DHCLabels1D(N//2)
    momentumSpaceLabels = lt.momentumLabelsDHC1D(N//2)
    for i in range(N):
        for j in range(N):
            if realSpaceLabels[i][1] == momentumSpaceLabels[j][1]:
                r = lt.LabelToR1D(realSpaceLabels[i])
                k = lt.momentumLabelToK1D(momentumSpaceLabels[j], N//2)
                matrix[i, j] = (1/((N//2)**0.5))*np.exp(k*r*1j)
    return matrix

# Makes the Hamiltonian for Graphene
# M is the number of horizontal unit cells
# N is the number of vertical unit cells
def grapheneHamitonian(M, N):
    # Loop through labels and take nearest 3 neighbours
    labels = lt.DHCLabels2D(M, N)
    # Rule - nearest neighbours are -1 else 0
    hami = np.zeros((2*M*N,2*M*N), dtype=complex)
    for label in labels:
        unitcell, atom = label
        if atom == "a":
            i, j = lt.LabelToIndex2D(label, N), lt.LabelToIndex2D(((unitcell[0], (unitcell[1]-1)%N), "b"), N)
            hami[i, j], hami[j, i] = -1, -1
            i, j = lt.LabelToIndex2D(label, N), lt.LabelToIndex2D((unitcell, "b"), N)
            hami[i, j], hami[j, i] = -1, -1
            i, j = lt.LabelToIndex2D(label, N), lt.LabelToIndex2D((((unitcell[0]+1)%M, (unitcell[1]-1)%N), "b"), N)
            hami[i, j], hami[j, i] = -1, -1
    return hami

# Generates the fourier tranform that takes your real space hamiltonian to momentum space
# Works on the Graphene Hamiltonian and matrix is of size (2*M*N)x(2*M*N)
# M is the number of horizontal unit cells
# N is the number of vertical unit cells
def FourierTransform2D(M, N):
    matrix = np.zeros((2*M*N, 2*M*N), dtype=complex)
    realSpaceLabels = lt.DHCLabels2D(M, N)
    momentumSpaceLabels = lt.momentumLabelsDHC2D(M, N)
    for i in range(2*M*N):
        for j in range(2*M*N):
            if realSpaceLabels[i][1] == momentumSpaceLabels[j][1]:
                r = lt.LabelToR2D(realSpaceLabels[i])
                k = lt.momentumLabelToK2D(momentumSpaceLabels[j], M, N)
                matrix[i, j] = (1/((M*N)**0.5))*np.exp(np.dot(k,r)*1j)
    return matrix

# Makes the Hamiltonian for Boron Nitride
# M is the number of horizontal unit cells
# N is the number of vertical unit cells
# L is the Energy of the lattice point
def BoronNitrideHamitonian(M, N, L):
    # Loop through labels and take nearest 3 neighbours
    # Rule - nearest neighbours are -1 else 0
    # Rule - if atom is a energy is L else -L
    hami = grapheneHamitonian(M, N)
    for i in range(2*M*N):
        if i%2 == 0:
            hami[i, i] = L
        else:
            hami[i, i] = -L
    return hami

# Makes the Hamiltonian for Graphene
# M is the number of horizontal unit cells
# N is the number of vertical unit cells
# L is the energy of the lattice point
# t is the energy of the same-atom interactions
# phi is the phase of the same-atom interactions
def haldaneModelHamitonian(M, N, L, t, phi):
    # Loop through labels and take nearest 3 neighbours
    labels = lt.DHCLabels2D(M, N)
    hami = BoronNitrideHamitonian(M, N, L)
    # Rule - nearest neighbours are -1 else 0
    # Rule - if atom is a energy is L else -L
    # Rule - nearest neighbour same atom interactions is te^(iphi)
    for label in labels:
        unitcell, atom = label
        n1label, n2label = label, label
        if atom == "a":
            n1label, n2label = ((unitcell[0], (unitcell[1]+1)%N), "a"), (((unitcell[0]+1)%M, unitcell[1]), "a")
        else:
            n1label, n2label = (((unitcell[0]-1)%M, (unitcell[1]+1)%N), "b"), ((unitcell[0], (unitcell[1]+1)%N), "b")

        i, j, k = lt.LabelToIndex2D(label, N), lt.LabelToIndex2D(n1label, N), lt.LabelToIndex2D(n2label, N)
        hami[i, j], hami[j, i] = -t*np.exp(phi*1j), -t*np.exp(-phi*1j)
        hami[j, k], hami[k, j] = -t*np.exp(phi*1j), -t*np.exp(-phi*1j)
        hami[k, i], hami[i, k] = -t*np.exp(phi*1j), -t*np.exp(-phi*1j)
    return hami


def HydrogenChainPlot(N):
    # Generate the real and momentum space hamiltonian
    hami = HCHamiltonianReal(N)
    F = HCFourierMatrix(N)
    klabels = lt.momentumLabelsHC1D(N)
    FH = np.conjugate(F).T
    Hk = (FH@hami@F).round(decimals=5)
    Ek = np.diagonal(Hk)
    k = [lt.momentumLabelToK1D(label, N) for label in klabels]
    # Fermi Energy line
    sortedEk = np.sort(Ek)
    fermiEnergy = (sortedEk[(N//2) - 1].real + sortedEk[N//2].real)/2
    # Plotting
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(k, Ek, ".", label="Band Structure")
    ax.plot(k, -2*np.cos(k), "--", color = "black", label="Theoretical", alpha=0.5)
    ax.hlines(fermiEnergy, 0, 2*np.pi, label="Fermi Energy")
    ax.set_xlabel("Momentum Location (k)")
    ax.set_ylabel("Energy E(k)")
    ax.legend()
    fig.suptitle("Energy curve for Hydrogen Chain")
    #fig.savefig("Plots/HydrogenChain")
    plt.show()

def DistortedHydrogenChainPlot(N):
    # Generate the real and momentum space hamiltonian
    hami = DHCHamiltonianReal(N)
    F = DHCFourierMatrix(N)
    klabels = lt.momentumLabelsDHC1D(N//2)
    FH = np.conjugate(F).T
    Hk = (FH@hami@F).round(decimals=5)
    Ek1, Ek2 = [], []
    # Find the Eigenvalues of the block diagonals
    for i in range(N//2):
        E, _ = np.linalg.eigh(Hk[2*i:2*i+2, 2*i:2*i+2])
        Ek1.append(E[0])
        Ek2.append(E[1])
    Ek1, Ek2 = np.array(Ek1), np.array(Ek2)
    ks = [lt.momentumLabelToK1D(label, N//2) for label in klabels]
    # Fermi Energy line
    fermiEnergy = (Ek1.min() + Ek2.max())/2
    # Plotting
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(ks[0:N+1:2], Ek1, ".", color="blue", label="Band Structure")
    ax.plot(ks[1:N+2:2], Ek2, ".", color="blue")
    ax.hlines(fermiEnergy, 0, 2*np.pi, label="Fermi Energy")
    ax.set_xlabel("Momentum Location (k)")
    ax.set_ylabel("Energy E(k)")
    ax.legend()
    fig.suptitle("Energy curve for Distorted Hydrogen Chain")
    #fig.savefig("Plots/DistortedHydrogenChain")
    plt.show()

def Model2DPlot(model, hamiltonian, M, N):
    # Model 2 is text without space
    model2 = "".join(model.split(" "))
    #hami = grapheneHamitonian(M, N)
    F = FourierTransform2D(M, N)
    klabels = lt.momentumLabelsDHC2D(M, N)
    Ek1, Ek2 = np.zeros((M, N)), np.zeros((M, N))
    kxs, kys = np.zeros((M, N)), np.zeros((M, N))

    Hk = (np.conjugate(F.T)@hamiltonian@F)
    for i in range(M*N):
        E, _ = np.linalg.eigh(Hk[2*i:2*i+2, 2*i:2*i+2])
        unitcell, _ = klabels[2*i]
        kx, ky = lt.momentumLabelToK2D(klabels[2*i], M, N)
        Ek1[unitcell[0], unitcell[1]], Ek2[unitcell[0], unitcell[1]] = E[0], E[1]
        kxs[unitcell[0], unitcell[1]], kys[unitcell[0], unitcell[1]] = kx, ky
    
    # Fermi Energy
    fermiEnergy = (Ek2.min() + Ek1.max())/2
    fermiEnergyArray = np.ones_like(Ek1)*fermiEnergy

    # 3D Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(kxs, kys, Ek1, color="red")
    ax.plot_surface(kxs, kys, Ek2, color="red")
    ax.plot_surface(kxs, kys, fermiEnergyArray)
    ax.set_xlabel("$K_x$")
    ax.set_ylabel("$K_y$")
    ax.set_zlabel("$E(k)$")
    fig.suptitle(model+" Band Structure")
    # fig.savefig("Plots/"+model2+"3D")
    # Countour plot
    fig2, ax2 = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    cs1 = ax2[0].pcolormesh(kxs, kys, Ek1)
    fig2.colorbar(cs1, ax=ax2[0], label="$E(k)$")
    ax2[0].set_xlabel("$k_x$")
    ax2[0].set_ylabel("$k_y$")
    ax2[0].set_title("$E(k) > 0$")
    cs2 = ax2[1].pcolormesh(kxs, kys, Ek2)
    fig2.colorbar(cs2, ax=ax2[1], label="$E(k)$")
    ax2[1].set_xlabel("$k_x$")
    ax2[1].set_ylabel("$k_y$")
    ax2[1].set_title("$E(k) < 0$")
    fig2.suptitle("Energy in Momentum Space for " + model)
    # fig2.savefig("Plots/"+model2+"ContourPlot")

    #kx Slice Plot
    # If MxN = 9x9 you can do a 3x3 plot
    idxs = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
    fig3, ax3 = plt.subplots(nrows=3, ncols=3, figsize=(10, 5))
    for i in range(M):
        ax3[idxs[i][0], idxs[i][1]].plot(kys[i, :], Ek1[i, :])
        ax3[idxs[i][0], idxs[i][1]].plot(kys[i, :], Ek2[i, :])
        ax3[idxs[i][0], idxs[i][1]].set_title("$k_x=${:.3f}".format(kxs[i, 0]))
        ax3[idxs[i][0], idxs[i][1]].set_xlabel("$k_y$")
        ax3[idxs[i][0], idxs[i][1]].set_ylabel("$E(k)$")
    fig3.suptitle("Energy of "+model+" $k_y$ along slices of $k_x$")
    fig3.tight_layout()
    # fig3.savefig("Plots/"+model2+"SlicePlots")
    # Hermitian Matshow
    fig4, ax4 = plt.subplots(nrows=1, ncols=1)
    ax4.matshow(np.abs(Hk))
    fig4.suptitle("Momentum Space Hamiltonian")
    # fig4.savefig("Plots/"+model2+"Hk")
    plt.show()


if __name__ == "__main__":
    # HydrogenChainPlot(100)
    # DistortedHydrogenChainPlot(100)
    # hami = grapheneHamitonian(9, 9)
    # Model2DPlot("Graphene", hami, 9, 9)
    # hami = BoronNitrideHamitonian(9, 9, 0.2)
    # Model2DPlot("Boron Nitride", hami, 9, 9)
    hami = haldaneModelHamitonian(9, 9, 1.2, 0.3, 0.7)
    Model2DPlot("Haldane Model", hami, 9, 9)
    

