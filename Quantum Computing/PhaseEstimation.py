import QCSimulatorII as qcII
import NonAtomicGates as nag
import DiracNotation as dn
import numpy as np
import matplotlib.pyplot as plt

#Allows matplotlib to render latex
plt.rcParams.update({"text.usetex": True})

#Quantum Fourier Transform
def QFT(inputState, numWires):
    #Seqeucne of hadamards and control phase
    for i in range(numWires-1, -1, -1):
        inputState = qcII.H(i, inputState)
        for j in range(i-1, -1, -1):
            phase = np.pi/(2**(i-j))
            inputState = nag.CPHASE(i, j, phase, inputState)
    
    #Swap gates to reverse output
    for k in range(numWires//2):
        inputState = nag.SWAP(k, numWires-1-k, inputState)
    return inputState

#Inverse Quantum Fourier Transform
def InverseQFT(inputState, numWires):
    #Seqeucne of hadamards and control phase
    for i in range(numWires):
        for j in range(i):
            phase = -np.pi/(2**(i-j))
            inputState = nag.CPHASE(j, i, phase, inputState)
        inputState = qcII.H(i, inputState)
        
    
    #Swap gates to reverse output
    for k in range(numWires//2):
        inputState = nag.SWAP(k, numWires-1-k, inputState)
    return inputState

#Phase Estimation 1 Wire
def phaseEstimator1(phase, inputState):
    outputState = qcII.H(0, inputState)
    outputState = nag.CPHASE(0, 1, phase*2*np.pi, outputState)
    outputState = qcII.H(0, outputState)
    return outputState

#Phase Estimation 2 Wire
def phaseEstimator2(phase, inputState):
    outputState = qcII.H(0, inputState)
    outputState = qcII.H(1, outputState)
    outputState = nag.CPHASE(1, 2, phase*2*np.pi, outputState)
    outputState = nag.CPHASE(0, 2, phase*2*np.pi, outputState)
    outputState = nag.CPHASE(0, 2, phase*2*np.pi, outputState)
    outputState = qcII.H(0, outputState)
    outputState = nag.CPHASE(0, 1, -np.pi/2, outputState)
    outputState = qcII.H(1, outputState)
    outputState = nag.SWAP(0, 1, outputState)
    return outputState

# Phase Estimator n Wires
def phaseEstimatorN(numWires, phase, inputState):
    for i in range(numWires):
        inputState = qcII.H(i, inputState)

    for i in range(numWires):
        for j in range(2**i):
            inputState = nag.CPHASE(numWires-i-1, numWires, phase*2*np.pi, inputState)

    outputState = InverseQFT(inputState, numWires)
    return outputState

# Phase Estimator n Wires but faster
def phaseEstimatorNFaster(numWires, phase, inputState):
    for i in range(numWires):
        inputState = qcII.H(i, inputState)

    for i in range(numWires):
        inputState = nag.CPHASE(numWires-i-1, numWires, (2**i)*phase*2*np.pi, inputState)

    outputState = InverseQFT(inputState, numWires)
    return outputState

if __name__ == "__main__":
    inputState = [((0.7+0j), '0000001'), ((0.3+0j), '0000000')]
    numWires = len(inputState[0][1]) - 1

    phases = np.linspace(0, 1, 100)
    thetas = []

    #For First Plot
    for phase in phases:
        outputState = phaseEstimatorNFaster(numWires, phase, inputState)
        outputVec = dn.StateToVec(outputState)
        outputVecProb = np.argmax(outputVec * np.conj(outputVec))
        thetas.append((outputVecProb - 1)/(2**numWires))
    
    #For second plot
    outputState2 = phaseEstimatorNFaster(6, 0.5, inputState)
    outputVec2 = dn.StateToVec(outputState2)
    print(outputVec2)
    print(outputState2)
    outputVec2Prob = outputVec2*np.conj(outputVec2)
    x = [(i)/(2**(numWires)) for i in range(2**numWires)]
    y = [outputVec2Prob[(2*i)+1] for i in range(2**numWires)]
        
    #Plotting
    fig, ax = plt.subplots(1, 1)

    ax.plot(phases, thetas, ".")
    ax.set_title("7 Wire Phase Estimation With Input")
    ax.set_xlabel("$\phi$/2$\pi$")
    ax.set_ylabel("Predicted $\phi$/2$\pi$")
    #fig.savefig("Plots/7 Wire Phase Estimation Simulator Plot")

    fig2, ax2 = plt.subplots(1,1)
    ax2.plot(x, y, ".")
    ax2.axvline(0.5, ls="--")
    ax2.set_xlabel("$\phi$/2$\pi$ Predicted")
    ax2.set_ylabel("Probability of prediction")
    ax2.set_title("7 Wire Phase Estimation With Input")
    fig2.savefig("Plots/7 Wire Phase Estimation Simulator Plot With Input")
    plt.show()
