import numpy as np
import sympy
import DiracNotation as dn
from fractions import Fraction

def ClassicalShor(n):
    #Checking that factor isn't easy or impossible
    if (n%2 == 0):
        return (0,)
    if (sympy.isprime(n)):
        return (0,)
    for i in range(2, int(np.log2(n)) + 1):
        root = n**(1./i)
        if (root == int(root)):
            return (0,)
    #Picking a random number
    factor = (0,)
    flag = True
    while (flag):
        num = np.random.randint(1, n)
        if np.gcd(num, n) == 1:
            for j in range(n):
                if (pow(num, j, n) == 1) and (j%2==0):
                    #factor = (np.power(num, j/2)-1)%n
                    factor = (num, j)
        #Break case
        if (len(factor) != 1):
            factor1 = np.gcd((pow(factor[0], factor[1]//2, n)-1)%n, n)
            factor2 = np.gcd((pow(factor[0], factor[1]//2, n)+1)%n, n)
            if (factor1 == 1) or (factor1 == n) or (factor2 == 1) or (factor2 == n):
                continue
            else:
                flag = False
    return factor

def QuantumMatrix(x, N):
    outputMatrix = np.zeros((2**(int(np.ceil(np.log2(N)))), 2**(int(np.ceil(np.log2(N))))))
    for i in range(N):
        outputMatrix[i, (x*i)%N] = 1
    return outputMatrix

def ShorwithMatrix(n):
    #Checking that factor isn't easy or impossible
    if (n%2 == 0):
        return (0,)
    if (sympy.isprime(n)):
        return (0,)
    for i in range(2, int(np.log2(n)) + 1):
        root = n**(1./i)
        if (root == int(root)):
            return (0,)
    #Picking a random number
    factor = (0,)
    flag = True
    while (flag):
        num = np.random.randint(1, n)
        if np.gcd(num, n) == 1:

            output = QuantumMatrix(num, n)
            w, v = np.linalg.eig(output)
            w = np.angle(w)
            for i in range(len(w)):
                if w[i] < 0:
                    w[i] += 2*np.pi
            phase = w/(2*np.pi)
            
            for i in range(len(phase)):
                r = Fraction(phase[i]).limit_denominator(10).denominator
                factor1 = np.gcd((pow(num, r//2, n)-1)%n, n)
                factor2 = np.gcd((pow(num, r//2, n)+1)%n, n)
                #Break case
                if (factor1 == 1) or (factor1 == n) or (factor2 == 1) or (factor2 == n):
                    continue
                else:
                    factor = (num, r)
                    flag = False
                
    return factor

if __name__ == "__main__":
    #n=235631
    #n=33
    #factor = ClassicalShor(n)
    #print(factor == sympy.isprime(n))
    #print(factor)
    #if (len(factor) != 1):
    #    print(np.gcd((pow(factor[0], factor[1]//2, n)-1)%n, n))
    #    print(np.gcd((pow(factor[0], factor[1]//2, n)+1)%n, n))
    
    #[[3245331, (753202, 3245328), (3, 1081777)], [1273657, (789454, 1255800), (74921, 17)], 
    # [4537253, (2427042, 4533900), (156457, 29)], [3563821, (43814, 126282), (42119, 3)], 
    # [356381, (282205, 354816), (29, 12289)], [3849275, (3182459, 3842430), (79, 48725)],
    # [354731, (261000, 354562), (2099, 169)], [2545387, (1296701, 2528064), (299, 8513)],
    # [25647381, (), ()], [1263511, (), ()]]
    n = 791
    factor = ShorwithMatrix(n)
    print(factor)
    if (len(factor) != 1):
       print(np.gcd((pow(factor[0], factor[1]//2, n)-1)%n, n))
       print(np.gcd((pow(factor[0], factor[1]//2, n)+1)%n, n))