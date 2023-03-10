import IsingModel as im
import numpy as np
import matplotlib.pyplot as plt
import stats as st
import scipy as sc

#Renders LaTeX
plt.rcParams["text.usetex"] = True

# Function performs a convolution for the average magnitization
# Reduces size of system by 3
def coarseGraining(system):
    newL = system.shape[0]//3
    newSystem = np.zeros((newL, newL), dtype=int)

    for i in range(1, system.shape[0], 3):
        for j in range(1, system.shape[1], 3):
            newVal = np.sum(system[i-1:i+2, j-1:j+2])
            if newVal >= 1:
                newVal = 1
            else:
                newVal = -1
            newSystem[(i-1)//3, (j-1)//3] = newVal
    
    return newSystem

# L: Size of system
# betas: A list of temperatures
# masks: A list of masks to perform mutiple spins calculations at once
# Function produces the result of two 3x3 coarse graining
def coarseGrainingSnapshots(L, betasList, masks):
    for beta in betasList:
        # Defining a new system for each beta
        system = np.random.choice([1, -1], size=(L, L))
        #Stabilising markov chain monte carlo
        for sweep in range(10000):
            #Creating mask for spins which do not depend on each other
            system = im.sweep(system, beta, masks)
        
        # Coarse graining the system
        newSystem = coarseGraining(system)
        newNewSystem = coarseGraining(newSystem)

        #Plotting
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(6,3))
        ax[0].imshow(system)
        ax[0].set_title("${} \\times {}$ Configuration".format(system.shape[0], system.shape[1]))
        ax[1].imshow(newSystem)
        ax[1].set_title("${} \\times {}$ Configuration".format(newSystem.shape[0], newSystem.shape[1]))
        ax[2].imshow(newNewSystem)
        ax[2].set_title("${} \\times {}$ Configuration".format(newNewSystem.shape[0], newNewSystem.shape[1]))
        fig.suptitle("Coarse Graining for $\\beta = {}$".format(beta))
        fig.savefig("Coarse graining for beta = {}.png".format(beta))
    
    #Theory
    theorySystem = np.random.choice([1, -1], size=(L, L))
    newTheorySystem = coarseGraining(theorySystem)
    newNewTheorySystem = coarseGraining(newTheorySystem)
    fig2, ax2 = plt.subplots(nrows=1, ncols=3, figsize=(6,3))
    ax2[0].imshow(theorySystem)
    ax2[0].set_title("${} \\times {}$ Configuration".format(theorySystem.shape[0], theorySystem.shape[1]))
    ax2[1].imshow(newTheorySystem)
    ax2[1].set_title("${} \\times {}$ Configuration".format(newTheorySystem.shape[0], newTheorySystem.shape[1]))
    ax2[2].imshow(newNewTheorySystem)
    ax2[2].set_title("${} \\times {}$ Configuration".format(newNewTheorySystem.shape[0], newNewTheorySystem.shape[1]))
    fig2.suptitle("Coarse Graining for $\\beta = 0$ Theoretical")
    fig2.savefig("Coarse graining for beta = 0 theoretical")

    theorySystemInf = np.ones_like(theorySystem)
    newTheorySystemInf = coarseGraining(theorySystemInf)
    newNewTheorySystemInf = coarseGraining(newTheorySystemInf)
    fig3, ax3 = plt.subplots(nrows=1, ncols=3, figsize=(6,3))
    ax3[0].imshow(theorySystemInf)
    ax3[0].set_title("${} \\times {}$ Configuration".format(theorySystemInf.shape[0], theorySystemInf.shape[1]))
    ax3[1].imshow(newTheorySystemInf)
    ax3[1].set_title("${} \\times {}$ Configuration".format(newTheorySystemInf.shape[0], newTheorySystemInf.shape[1]))
    ax3[2].imshow(newNewTheorySystemInf)
    ax3[2].set_title("${} \\times {}$ Configuration".format(newNewTheorySystemInf.shape[0], newNewTheorySystemInf.shape[1]))
    fig3.suptitle("Coarse Graining for $\\beta = 100$ Theoretical")
    fig3.savefig("Coarse graining for beta = 100 theoretical")



if __name__ == "__main__":
    L = 81
    #system = np.random.choice([1, -1], size=(L, L))
    betas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    #beta = 0.0
    #betas = [0.0]
    masks = im.create_masks(L)
    smallMasks = im.create_masks(L//3)

    mean_mags_array = []
    mean_mags_error_array = []
    mean_small_mags_array = []
    mean_small_mags_error_array = []

    system = np.random.choice([1, -1], size=(L, L))
    smallSystem = np.random.choice([1, -1], size=(L//3, L//3))

    for beta in betas:
        # Defining a new system for each beta
        #Stabilising markov chain monte carlo
        for sweep in range(100):
            system = im.sweep(system, beta, masks)
            smallSystem = im.sweep(smallSystem, beta, smallMasks)
            
        #Snapshots
        mags = []
        smallMags = []
        for sweep in range(10000):
            system = im.sweep(system, beta, masks)
            smallSystem = im.sweep(smallSystem, beta, smallMasks)
            
            #Magnetization
            coarseGrainSystem = coarseGraining(system)
            mags.append(np.mean(coarseGrainSystem)**2)
            smallMags.append(np.mean(smallSystem)**2)
        
        # End of run calculations
        magMean, magVar, magErr, magCor = st.Stats(np.array(mags))
        mean_mags_array.append(magMean)
        mean_mags_error_array.append(magErr)

        smallMagMean, smallMagVar, smallMagErr, smallMagCor = st.Stats(np.array(smallMags))
        mean_small_mags_array.append(smallMagMean)
        mean_small_mags_error_array.append(smallMagErr)
    
    #Theory values
    # theorySystem = np.random.choice([1, -1], size=(L, L))
    # coarseGrainTheorySystem = coarseGraining(theorySystem)

    # smallTheorySystem = np.random.choice([1, -1], size=(L//3, L//3))
    
    #Plotting
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.errorbar(betas, mean_mags_array, yerr=mean_mags_error_array, ecolor="red", barsabove=True, capsize=1.0, label="Coarse Grained")
    ax.errorbar(betas, mean_small_mags_array, yerr=mean_small_mags_error_array, ecolor="black", barsabove=True, capsize=1.0, label="Native")
    # ax.plot(0, np.mean(coarseGrainTheorySystem)**2, 'g*', label="Theory Coarse Grained")
    # ax.plot(0, np.mean(smallTheorySystem)**2, 'cD', alpha=0.5, label="Theory Native")
    ax.set_xlabel("$\\beta$")
    ax.set_ylabel("$\\langle M^2 \\rangle$")
    ax.set_title("Effect of Coarse graining vs. Native Configuration")
    ax.legend()
    # fig.savefig("CoarseGrainingvsNative")
    plt.show()

    # Interpolation
    R = sc.interpolate.interp1d(mean_small_mags_array, betas, bounds_error=False, fill_value=(0, 1))
    #plt.plot(betas, R(mean_small_mags_array))
    fig2, ax2 = plt.subplots(1, 1)
    ax2.plot(betas, R(mean_mags_array), label="Interpolation")
    ax2.plot(betas, betas, 'b--', label="x=y")
    ax2.set_xlabel("J")
    ax2.set_ylabel("R(J)")
    ax2.legend()
    # fig2.savefig("RJvsJ2")
    # np.save("RJ", R(mean_mags_array))
    # np.save("betas", betas)

    #coarseGrainingSnapshots(81, betas, masks)
    plt.show()
