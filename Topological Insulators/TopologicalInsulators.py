import numpy as np
import matplotlib.pyplot as plt
import Lattice as lt
import TightBindingModels as models

# Activate LaTeX for matplotlib
plt.rcParams['text.usetex'] = True

# Measures the Direct Energy Gap of the Haldane Model for different M/L values
# M is the number of horizontal unit cells
# N is the number of vertical unit cells
def gapPlot(M, N):
    Ls = np.arange(0, 2.1, 0.1)
    F = models.FourierTransform2D(M, N)
    klabels = lt.momentumLabelsDHC2D(M, N)
    Ek1, Ek2 = np.zeros((M, N)), np.zeros((M, N))
    #kxs, kys = np.zeros((M, N)), np.zeros((M, N))
    Energies = np.zeros_like(Ls)

    for j in range(Ls.size):
        L = Ls[j]
        hami = models.haldaneModelHamitonian(M, N, L, 0.3, 0.7)
        Hk = np.conjugate(F.T)@hami@F
        for i in range(M*N):
            E, _ = np.linalg.eigh(Hk[2*i:2*i+2, 2*i:2*i+2])
            unitcell, _ = klabels[2*i]
            #kx, ky = lt.momentumLabelToK2D(klabels[2*i], M, N)
            Ek1[unitcell[0], unitcell[1]], Ek2[unitcell[0], unitcell[1]] = E[0], E[1]
            #kxs[unitcell[0], unitcell[1]], kys[unitcell[0], unitcell[1]] = kx, ky
        Energies[j] = (Ek2 - Ek1).min()

    # Plotting
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(Ls, Energies, ".")
    ax.set_xlabel("M")
    ax.set_ylabel("Direct Gap [$E_2(k) - E_1(k)$]")
    fig.suptitle("Measuring the Direct Gap for the Haldane Model")
    #fig.savefig("Plots/MeasuringTheGap")
    plt.show()

# Measure the clockwise berry Plaqutte for each point in the momentum space
# L is the atom's self energy
# M is the number of horizontal unit cells
# N is the number of vertical unit cells
def berryPlaqutte(L, M, N):
    F = models.FourierTransform2D(M, N)
    klabels = lt.momentumLabelsDHC2D(M, N)
    Ev1, Ev2 = np.zeros((M, N, 2), dtype=complex), np.zeros((M, N, 2), dtype=complex)
    berryPlaquttes = np.zeros((M, N))

    hami = models.haldaneModelHamitonian(M, N, L, 0.3, 0.7)
    Hk = np.conjugate(F.T)@hami@F

    # Creating a grid of eigenvectors
    for i in range(M*N):
        _, Ev = np.linalg.eigh(Hk[2*i:2*i+2, 2*i:2*i+2])
        unitcell, _ = klabels[2*i]
        Ev1[unitcell[0], unitcell[1], :], Ev2[unitcell[0], unitcell[1], :] = Ev[:, 0], Ev[:, 1]
    
    # Calculating Berry Plaqutte for each position:
    for i in range(M*N):
        unitcell, _ = klabels[2*i]
        z1 = Ev2[unitcell[0], unitcell[1]]@np.conjugate(Ev2[(unitcell[0]+1)%M, unitcell[1]].T)
        z2 = Ev2[(unitcell[0]+1)%M, unitcell[1]]@np.conjugate(Ev2[(unitcell[0]+1)%M, (unitcell[1]-1)%N].T)
        z3 = Ev2[(unitcell[0]+1)%M, (unitcell[1]-1)%N]@np.conjugate(Ev2[unitcell[0], (unitcell[1]-1)%N].T)
        z4 = Ev2[unitcell[0], (unitcell[1]-1)%N]@np.conjugate(Ev2[unitcell[0], unitcell[1]].T)
        zTotal = z1*z2*z3*z4
        berryPlaquttes[unitcell[0], unitcell[1]] = np.arctan2(zTotal.imag, zTotal.real)

    return berryPlaquttes

def berryPhasePlots(M, N):
    # For Different Ls
    berryCurvature1 = berryPlaqutte(0.8, M, N)
    berryCurvature2 = berryPlaqutte(1.2, M, N)

    # Create kx, ky grid
    klabels = lt.momentumLabelsDHC2D(M, N)
    kxs, kys = np.zeros((M, N)), np.zeros((M, N))
    for i in range(M*N):
        unitcell, _ = klabels[2*i]
        kx, ky = lt.momentumLabelToK2D(klabels[2*i], M, N)
        kxs[unitcell[0], unitcell[1]], kys[unitcell[0], unitcell[1]] = kx, ky

    # Calculate chern number for a bunch of atoms
    Ls = np.arange(0, 2.1, 0.1)
    chernNumbers = []
    for L in Ls:
        berryCurvature = berryPlaqutte(L, M, N)
        chernNumbers.append(np.sum(berryCurvature)/(2*np.pi))

    # Plotting
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    cs1 = ax[0].pcolormesh(kxs, kys, berryCurvature1)
    ax[0].set_xlabel("$k_x$")
    ax[0].set_ylabel("$k_y$")
    ax[0].set_title("$M = 0.8$")
    fig.colorbar(cs1, ax=ax[0])
    cs2 = ax[1].pcolormesh(kxs, kys, berryCurvature2)
    ax[1].set_xlabel("$k_x$")
    ax[1].set_ylabel("$k_y$")
    ax[1].set_title("$M = 1.2$")
    fig.colorbar(cs2, ax=ax[1])
    fig.suptitle("Countour Plots of Berry Curvature in Haldane Model")
    #fig.savefig("Plots/BerryCurvaturePlot")
    

    fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    ax2.plot(Ls, chernNumbers, ".")
    ax2.set_xlabel("$M$")
    ax2.set_ylabel("Chern Number")
    fig2.suptitle("Chern Number for various $M$ \n in the Haldane Model")
    fig2.savefig("Plots/ChernNumberPlot")
    plt.show()
    

if __name__ == "__main__":
    # gapPlot(9, 9)
    berryPhasePlots(9, 9)