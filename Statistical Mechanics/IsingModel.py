import numpy as np
import matplotlib.pyplot as plt

#Calculates energies of a system
def Energy(spins):
    x, y = spins.shape
    energy = 0
    for i in range(x):
        for j in range(y):
            #Adding top and bottom neighbours
            energy += (system[i][j]*system[i-1][j]) + (system[i][j]*system[(i+1)%x][j])
            #Adding left and right neighbours
            energy += (system[i][j]*system[i][j-1]) + (system[i][j]*system[i][(j+1)%y])
    #Dealing with double counting
    energy /= 2
    return energy

# Calculates energy difference 
# spin_to_flip is a tuple of coordinates
def deltaE(system, spin_to_flip):
    L1, L2 = system.shape
    x, y = spin_to_flip
    before_energy = (system[x][y]*system[x-1][y]) + (system[x][y]*system[(x+1)%L1][y])
    before_energy += (system[x][y]*system[x][y-1]) + (system[x][y]*system[x][(y+1)%L2])
    system[x][y] = -system[x][y]
    after_energy = (system[x][y]*system[x-1][y]) + (system[x][y]*system[(x+1)%L1][y])
    after_energy += (system[x][y]*system[x][y-1]) + (system[x][y]*system[x][(y+1)%L2])
    system[x][y] = -system[x][y]
    energy_diff = after_energy - before_energy
    return energy_diff

if __name__ == "__main__":
    L = 3
    system = np.random.choice([1, -1], size=(L, L))
    beta = 0.3

    #Stabilising markov chain monte carlo
    for sweep in range(20):
        for step in range(L*L):
            x = np.random.choice(L)
            y = np.random.choice(L)
            #Markov Chain Rule
            alpha = np.exp(-beta*deltaE(system, (x, y)))
            if alpha > 1:
                system[x][y] = -system[x][y]
            else:
                switch = np.random.choice([1, 0], p=[alpha, 1-alpha])
                if switch == 1:
                    system[x][y] = -system[x][y]
    
    #Snapshots
    configs = []
    for sweep in range(100000):
        for step in range(L*L):
            x = np.random.choice(L)
            y = np.random.choice(L)
            #Markov Chain Rule
            alpha = np.exp(-beta*deltaE(system, (x, y)))
            if alpha > 1:
                system[x][y] = -system[x][y]
            else:
                switch = np.random.choice([1, 0], p=[alpha, 1-alpha])
                if switch == 1:
                    system[x][y] = -system[x][y]
        #Converting system into binary number
        system_flat = system.flatten()
        system_flat = [0 if i == -1 else 1 for i in system_flat]
        bin_num = 0
        for i in range(len(system_flat)):
            bin_num += system_flat[i]*(2**((L*L)-i-1))
        configs.append(bin_num)

    #Theorectical value
    total_energy = 0
    theory_configs = []
    #Converting each inter to its associated binary number and hence state
    #Then calculating the energy of that state
    for i in range(2**(L*L)):
        bin_num = str(bin(i))[2:]
        while len(bin_num) < (L*L): bin_num = "0" + bin_num
        system = np.array(["-1" if x == "0" else "1" for x in bin_num]).astype(int).reshape((L, L))
        config_energy = np.exp(-beta*Energy(system))
        total_energy += config_energy
        theory_configs.append(config_energy)
    theory_configs = np.array(theory_configs)/total_energy
    x_points = [x for x in range(2**(L*L))]
    
    #Plotting
    fig, ax = plt.subplots(1, 1)
    ax.hist(configs, density=True, bins=2**(L*L), label="Monte Carlo")
    ax.plot(x_points, theory_configs, "r.", label="Theoretical value")
    ax.set_xlabel("System")
    ax.set_ylabel("Probability")
    ax.set_title("Simulating an Ising Model using\nMarkov Chain Monte Carlo")
    ax.legend(loc = 'upper right')
    fig.savefig("Plots/Simulating Ising Model")
    plt.show()