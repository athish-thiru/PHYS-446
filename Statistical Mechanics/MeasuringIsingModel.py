import numpy as np
import matplotlib.pyplot as plt
import IsingModel as im
import stats as st
import scipy as sc

#Renders LaTeX
plt.rcParams["text.usetex"] = True

if __name__ == "__main__":
    L = 27
    system = np.random.choice([1, -1], size=(L, L))
    #betas = np.arange(0, 1.1, 0.1)
    #betas = np.append(betas, [100])
    #betas = [0, 100]
    betas = np.arange(0, 1.05, 0.05)
    masks = im.create_masks(L)
    mean_mags_array = []
    mean_mags_error_array = []
    mean_energies_array = []
    mean_energies_error_array = []

    #Do betas[::-1] for genrating Cvs plot
    for beta in betas:
        #Stabilising markov chain monte carlo
        for sweep in range(20):
            #Creating mask for spins which do not depend on each other
            for mask in masks:
                #Heat Bath rule
                flip_prob = (1/(1 + np.exp(-beta*im.deltaE2(system, mask))))
                rands = np.random.rand(*flip_prob.shape)
                switch = rands < flip_prob
                finalmask = np.logical_and(mask, switch)
                system[finalmask] = np.negative(system[finalmask])
        
        #Snapshots
        mags = []
        energies = []
        for sweep in range(10000):
            for mask in masks:
                #Heat Bath rule
                flip_prob = (1/(1 + np.exp(-beta*im.deltaE2(system, mask))))
                rands = np.random.rand(*flip_prob.shape)
                switch = rands < flip_prob
                switch = np.where(np.random.rand(*flip_prob.shape) < flip_prob, True, False)
                finalmask = np.logical_and(mask, switch)
                system[finalmask] = np.negative(system[finalmask])
        
            #Magnetization
            mags.append(np.mean(system)**2)
            #Energy
            energies.append(im.Energy(system))
        
        magMean, magVar, magErr, magCor = st.Stats(np.array(mags))
        energyMean, energyVar, energyErr, energyCor = st.Stats(np.array(energies))
        mean_mags_array.append(magMean)
        mean_mags_error_array.append(magErr)
        mean_energies_array.append(energyMean)
        mean_energies_error_array.append(energyErr)


        # magHist, magBins = np.histogram(mags, bins=100)

        # # Theorectical calculation
        # theoryEnergies = []
        # theoryMags = []
        # if beta == 0:
        #     for i in range(10000):
        #         theorySystem = np.random.choice([1, -1], p =[0.5, 0.5], size=(L,L))
        #         theoryEnergies.append(im.Energy(theorySystem))
        #         theoryMags.append(np.mean(theorySystem)**2)
        # elif beta == 100:
        #     for i in range(10000):
        #         theorySystem = np.ones_like(system)
        #         theoryEnergies.append(im.Energy(theorySystem))
        #         theoryMags.append(np.mean(theorySystem)**2)

        # #Plotting
        # fig, ax = plt.subplots(1, 2, figsize=(10, 8))
        # ax[0].hist(energies, density=True, bins=100, label="MCMC")
        # ax[0].axvline(energyMean, ls="solid", color="red", label="Expectation Value")
        # ax[0].axvline(energyMean - energyErr, ls="dashed",  color="orange", label="Error Range")
        # ax[0].axvline(energyMean + energyErr, ls="dashed",  color="orange")
        # ax[0].set_xlabel("Energies")
        # ax[0].set_ylabel("Probability")
        # ax[1].step(magBins[:-1], magHist/np.sum(magHist), where="post", label="MCMC")
        # ax[1].axvline(magMean, ls="solid", color="red", label="Expectation Value")
        # ax[1].axvline(magMean - magErr, ls="dashed",  color="orange", label="Error Range")
        # ax[1].axvline(magMean + magErr, ls="dashed",  color="orange")
        # ax[1].set_xlabel("Magnetization")
        # ax[1].set_ylabel("Probability")
        
        # if len(theoryEnergies) != 0:
        #     tMagMean, tMagVar, tMagErr, tMagCor = st.Stats(np.array(theoryMags))
        #     tEnergyMean, tEnergyVar, tEnergyErr, tEnergyCor = st.Stats(np.array(theoryEnergies))
        #     tMagHist, tMagBins = np.histogram(theoryMags, bins=100)
        #     tEnergyHist, tEnergyBins = np.histogram(theoryEnergies, bins=100)

        #     ax[0].hist(theoryEnergies, density=True, bins=100, label="Theoretical", zorder=100, alpha=0.4, color="black")
        #     ax[0].set_title("Measurement $\langle E \\rangle = {:.3f} \pm {:.3f}$ \n Theoretical $\langle E \\rangle = {:.3f} \pm {:.3f}$".format(energyMean, energyErr, tEnergyMean, tEnergyErr))
        #     ax[1].step(tMagBins[:-1], tMagHist/np.sum(tMagHist), where="post", label="Theoretical", color="black", alpha=0.7)
        #     ax[1].set_title("Measurement $\langle M^2 \\rangle = {:.3f} \pm {:.3f}$ \n Theoretical $\langle M^2 \\rangle = {:.3f} \pm {:.3f}$".format(magMean, magErr, tMagMean, tMagErr))
        # else:
        #     ax[0].set_title(r"$\langle E \rangle = {:.3f} \pm {:.3f}$".format(energyMean, energyErr))
        #     ax[1].set_title(r"$\langle M^2 \rangle = {:.3f} \pm {:.3f}$".format(magMean, magErr))

        # ax[0].legend(loc = 'upper right')
        # ax[1].legend(loc = 'upper right')
        # fig.suptitle(r"Ising Model for $\beta = {:.1f}$".format(beta))
        # fig.savefig("Plots/Ising Model for beta = {:.1f}.png".format(beta))
            

        # fig2, ax2 = plt.subplots(1, 1)
        # ax2.matshow(system)
        # fig2.suptitle(r"Final Snapshot of Ising Model for $\beta = {:.1f}$".format(beta))
        # fig2.savefig("Plots/Final Snapshot of Ising Model for beta = {:.1f}.png".format(beta))
        # plt.close(fig)
        # plt.close(fig2)
    fig3, ax3 = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    ax3.errorbar(betas, mean_mags_array, yerr=mean_mags_error_array, ecolor="red", barsabove=True, capsize=1.0)
    ax3.set_xlabel("$\\beta$")
    ax3.set_ylabel("$\langle M^2 \\rangle$")
    ax3.set_title("Change in Magnetization with $\\beta$")
    fig3.savefig("MvsBeta")

    # fig4, ax4 = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    # # temps needs to be in increasing order for interpolation
    # temps = 1/betas[::-1]
    # Cvs = sc.interpolate.UnivariateSpline(temps, mean_energies_array).derivative()
    # new_temps = np.arange(1, 10, 0.1)
    # ax4.errorbar(new_temps, Cvs(new_temps), ecolor="red", barsabove=True, capsize=1.0)
    # ax4.set_xlabel("Temperature")
    # ax4.set_ylabel("Specific Heat")
    # ax4.set_title("Change in Specific Heat with Temperature")
    #fig4.savefig("CVvsTemp")
    plt.show()