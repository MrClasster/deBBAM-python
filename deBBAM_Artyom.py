## importing libraries
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import random as rng
from pathlib import Path


## function used later
def RtoQ(Rvector):
    width = np.size(Rvector)
    Qvector = np.zeros_like(Rvector)
    Qvector[0] = -2.0 + Rvector[1] / Rvector[0]
    Qvector[width - 1] = -2.0 + Rvector[width - 2] / Rvector[width - 1]

    for i in range(1, width - 1):
        Qvector[i] = (Rvector[i-1] + Rvector[i+1])/Rvector[i] - 2.0

    return(Qvector)

def WriteDataC(prefix, surname, data, index): 
    filename = f"{prefix}{surname}.raw"
    with open(filename, 'a') as file:
        for j, z in enumerate(data, start=1):
            re, im = z.real, z.imag
            if abs(re) <= 1e-99:
                re = 0.0
            if abs(im) <= 1e-99:
                im = 0.0
            file.write(f"{index:5d}{j:5d}{re:25.15g}{im:25.15g}\n")
        file.write("\n")

def WriteDataR(prefix, surname, data, index):
    filename = f"{prefix}{surname}.raw"
    with open(filename, 'a') as file:
        for j, x in enumerate(data, start=1):
            if abs(x) <= 1e-99:
                x = 0
            file.write(f"{index:5d}{j:5d}{x:25.15g}\n")
        file.write("\n")

## read parameters from the input file
with open('deBBAM.inp') as file:
    inputParameters = file.read().split()

## inputParameters is a list of strings, so we need to explicitly define each variable with its type
ISeed = int(inputParameters[0])
IWidth = int(inputParameters[1])
IDuration = int(inputParameters[2])
IVpotFlag = int(inputParameters[3])
RVpotWidth = float(inputParameters[4])
EngVpotRatio = float(inputParameters[5])
RPulsPosition = float(inputParameters[6])
RPulsWidth = float(inputParameters[7])
RTrajPosition = float(inputParameters[8])
RTrajWidth = float(inputParameters[9])
ITrajNumber = int(inputParameters[10])


## defining variables
physwidth = 1.0
lambda_ = 1.0 # note that we have to name this lambda_ as opposed to lambda
hbar = 1.0 # normalising 
mass = 0.5 # helps later with kinetic energy calculation

eps = physwidth/IWidth
delta = 2.0 * (eps**2) /lambda_
sigma0 = RPulsWidth / 4.0
Vpotmax = 2.0 * (50.0 * np.pi)**2
energy = Vpotmax * EngVpotRatio
k0mom = np.sqrt(2.0 * mass * energy / (hbar**2))
fullwidth = 2 * IWidth


## defining the potential
rng.seed(ISeed)
points = np.arange(1, fullwidth + 1) ## make an array of points/positions between 1 and fullwidth
potentials = np.zeros_like(points, dtype=complex) ## match every point to a complex 0 potential
well = np.abs(points - fullwidth // 2) < int(RVpotWidth * IWidth) // 2 ## create an array with true and false values, checking if every point is within the well
truewidth = IWidth

if IVpotFlag == -1: ## the potential well case
    potname = "WW"
    potentials[well] = -Vpotmax ## for every true point, set the potential to be -Vpotmax
elif IVpotFlag == 0: ## 0 potential case
    potname = "00"
elif IVpotFlag == 1: ## the potential barrier case
    potname = "BB"
    potentials[well] = Vpotmax ## for every true point, set the potential to be Vpotmax
elif IVpotFlag == 2: ## Anderson disorder
    potname = "AM"
    randomarray = np.zeros(well.sum()) 
    for i in range(well.sum()):
        randomarray[i] = rng.random()
    potentials[well] = Vpotmax * randomarray
elif IVpotFlag == 3: ## Aubry-Andre quasiperiodic potential
    potname = "AA"
    fiboA, fiboB = 1, 1 ## start the Fibonacci sequence
    while fiboA <= RVpotWidth * IWidth:
        fiboA, fiboB = fiboA + fiboB, fiboA
    alpha = fiboA / fiboB
    well = np.abs(points - fullwidth // 2) < fiboA // 2 ## we have to recreate the well array as we change the well size to be fiboA
    phi = rng.random()
    potentials[well] = Vpotmax * (np.cos(2 * np.pi * alpha * points[well] + 2 * np.pi * phi) + 1 ) / 2
    truewidth = fiboA


## absorbing potential
Uconst = complex(0, -50)
Aconst = 0.05
index = np.arange(1, 202)

absorber = Uconst * Vpotmax / (np.cosh(Aconst * index))**2 ## setup an array with imaginary potential values for points 1-201
potentials[:201] = absorber ## set the first 201 points to have that imaginary potential
potentials[-201:] = absorber[::-1] ## set the last 201 points to have that imaginary potential

## generate surname
def nint(x):
    return int(np.floor(x + 0.5))

surname = (f"{potname}-W{truewidth:05d}-D{IDuration:05d}-V{abs(IVpotFlag)}"
           f"-Vw{nint(100 * RVpotWidth):02d}"
           f"-ER{nint(10 * EngVpotRatio):02d}"
           f"-PP{nint(100 * RPulsPosition):02d}"
           f"-PW{nint(100 * RPulsWidth):02d}")


## setting up the initial puls
pulsvector = np.exp(1j * k0mom * points / IWidth) * np.exp(-(points / fullwidth - RPulsPosition/2 - 0.25)**2 / (2 * sigma0**2)) ## setting up the Gaussian wavepacket centered at RPulsPosition/2 + 0.25 of the full width
norm = np.sum(np.abs(pulsvector)**2) ## find the normalisation constant
pulsvector /= norm ## not truly normalised - consult later.

## setting the initial Bohm particles onto trajectories
tndex = np.arange(1, ITrajNumber + 1) ## setup an array for ITrajNumber particles
Trajectory = np.floor(IWidth * (RTrajPosition - RTrajWidth/2) + IWidth/2 + IWidth * RTrajWidth * tndex / ITrajNumber + 0.5) ## to recreate Fortran's NINT we use np.floor(... + 0.5)

## psivector and evector definition
psivector = pulsvector.copy()
evector = np.zeros(fullwidth, dtype = complex) ## setup a new array with fullwidth number of complex entries
evector[0] = 2.0 + eps**2 * potentials[0] - 1j * lambda_
for i in range(1, fullwidth):
    evector[i] = 2 + eps**2 * potentials[i] - 1j * lambda_ - 1.0 / evector[i-1]



## writing data
for f in Path(".").glob(f"*{surname}.raw"): ## delete previous data
    f.unlink()

WriteDataC("V-", surname, potentials, 0) ## Starting potentials
WriteDataC("A-", surname, pulsvector, 0) ## Starting pulse
WriteDataR("B-", surname, np.abs(pulsvector), 0) ## modulus of starting pulse
WriteDataR("T-", surname, Trajectory, 0) ## starting particle locations


## main time loop 
for itime in range(1, IDuration + 2):
    omegavector = np.zeros(fullwidth, dtype = complex)
    fvector = omegavector.copy()
    omegavector[0] = -psivector[1] + (1j * lambda_ + eps**2 * potentials[0] + 2) * psivector[0]
    fvector[0] = omegavector[0]


    for i in range(1, fullwidth - 1):
        omegavector[i] = -psivector[i+1] + (1j * lambda_ + eps**2 * potentials[i] + 2) * psivector[i] - psivector[i-1]
        fvector[i] = omegavector[i] + fvector[i-1] / evector[i-1]

    omegavector[fullwidth - 1] = (1j * lambda_ + eps**2 * potentials[fullwidth - 1] + 2) * psivector[fullwidth - 1] - psivector[fullwidth - 2]
    fvector[fullwidth - 1] = omegavector[fullwidth - 1] + fvector[fullwidth - 2] / evector[fullwidth - 2]

    psinewvector = np.zeros_like(psivector, dtype = complex)
    psinewvector[fullwidth - 1] = psivector[fullwidth - 1]

    for i in range(fullwidth - 2, -1, -1):
        psinewvector[i] = (psinewvector[i+1] - fvector[i]) / evector[i]

    ## Bohm quantities
    Rvector = np.abs(psivector)
    Qvector = RtoQ(Rvector)
    Svector = np.angle(psivector)

    ## Bohm trajectories
    for i in range(0, ITrajNumber):
        ipos = int(Trajectory[i])

        if ipos < fullwidth:
            PSright = np.conjugate(psivector[ipos])
            PSleft = np.conjugate(psivector[ipos-1])
            Pright = psivector[ipos]
            Pleft = psivector[ipos-1]
            trajnorm = np.abs(Pleft)**2

        else:
            PSright = 0.0
            PSleft = 0.0
            Pright = 0.0
            Pleft = 0.0
            trajnorm = 1.0

        deltaP = np.imag((np.conjugate(psivector[ipos-1]) * (Pright - Pleft) - psivector[ipos-1] * (PSright - PSleft)) / eps / (2.0 * mass) * delta / 2.0 * fullwidth / trajnorm)

        Trajectory[i] += deltaP

        if Trajectory[i] > fullwidth:
            Trajectory[i] = fullwidth
        if Trajectory[i] < 1.0:
            Trajectory[i] = 1.0

    WriteDataR('T-', surname, Trajectory, itime) # positions after this step
    WriteDataC('P-', surname, psivector, itime) # wavefunction
    WriteDataR('R-', surname, Rvector, itime) # amplitude
    WriteDataR('Q-', surname, Qvector, itime) # qvector
    WriteDataR("S-", surname, Svector, itime) # phase

    psivector = psinewvector.copy()