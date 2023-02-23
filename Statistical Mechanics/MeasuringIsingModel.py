import numpy as np
import matplotlib.pyplot as plt
import IsingModel as im
import stats as st

#Renders LaTeX
plt.rcParams["text.usetex"] = True

if __name__ == "__main__":
    L = 27
    system = np.random.choice([1, -1], size=(L, L))
    betas = np.arange(0, 1.1, 0.1)
    betas = np.append(betas, [100])
    #beta = 100
    masks = im.create_masks(L)

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
        
        mean1, var1, err1, cor1 = st.Stats(np.array(mags))
        mean2, var2, err2, cor2 = st.Stats(np.array(energies))

        hist, bins = np.histogram(mags, bins=100)


        #Plotting
        fig, ax = plt.subplots(1, 2, figsize=(10, 8))
        ax[0].hist(energies, density=True, bins=100, label="MCMC")
        ax[0].axvline(mean2, ls="solid", color="black", label="Expectation Value")
        ax[0].axvline(mean2 - err2, ls="dashed",  color="red", label="Error Range")
        ax[0].axvline(mean2 + err2, ls="dashed",  color="red")
        #ax.plot(x_points, theory_configs, "r.", label="Theoretical value")
        ax[0].set_xlabel("Energies")
        ax[0].set_ylabel("Probability")
        ax[0].set_title(r"$\langle E \rangle = {:.3f} \pm {:.3f}$".format(mean2, err2))
        ax[0].legend(loc = 'upper right')
        #fig.savefig("Plots/Simulating Ising Model")
        ax[1].step(bins[:-1], hist/np.sum(hist), where="post", label="MCMC")
        #ax[1].hist(mags, weights=hist/np.sum(hist), bins=100, label="MCMC")
        ax[1].axvline(mean1, ls="solid", color="black", label="Expectation Value")
        ax[1].axvline(mean1 - err1, ls="dashed",  color="red", label="Error Range")
        ax[1].axvline(mean1 + err1, ls="dashed",  color="red")
        ax[1].set_xlabel("Magnetization")
        ax[1].set_ylabel("Probability")
        ax[1].set_title(r"$\langle M^2 \rangle = {:.3f} \pm {:.3f}$".format(mean1, err1))
        ax[1].legend(loc = 'upper right')
        #fig.savefig("Plots/Simulating Ising Model")
        fig.suptitle(r"Ising Model for $\beta = {:.1f}$".format(beta))
        fig.savefig("Plots/Ising Model for beta = {:.1f}.png".format(beta))

        fig2, ax2 = plt.subplots(1, 1)
        ax2.matshow(system)
        fig2.suptitle(r"Final Snapshot of Ising Model for $\beta = {:.1f}$".format(beta))
        fig2.savefig("Plots/Final Snapshot of Ising Model for beta = {:.1f}.png".format(beta))
        #plt.show()
        plt.close(fig)
        plt.close(fig2)
        