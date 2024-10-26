import numpy as np


# Function for dB to power conversion
def db2pow(x):
    """
    :param x: input variable in dB scale
    :return: output in linear scale
    """
    return 10**(0.1*x)


# function to compute path loss
def plCalc(d, alphaHat):
    """
    :param d: distance between transmitter and receiver nodes
    :param alphaHat: path loss exponent
    :return:
    """
    c0 = -30  # path loss in dB at the reference distance
    PL = db2pow(c0) * d**(-alphaHat)  # path loss at a distance of d meters
    return PL


# function to generate channel coefficients
def generate_channels(arg):

    # parameters unpacking
    Nb, xNs, zNs, Ns, K = arg.Nb, arg.Mx, arg.Mz, arg.Ns, arg.K

    rf = db2pow(3)  # Rician factor

    # node locations
    locB = np.array([0, 0, 2.5])  # location of the BS
    locS = np.array([20, 0, 2.5])  # location of the IRS
    locU = np.array([20, 70, 0])  # center location of users in reflection region

    # generate communication users at random locations
    userRadius = 2  # radius of the user circle
    randRadius = 2 * userRadius * np.random.rand(K, )
    randAngle = np.random.rand(K)
    xU = locU[0] + randRadius * np.cos(2 * np.pi * randAngle)
    yU = locU[1] + randRadius * np.sin(2 * np.pi * randAngle)
    zU = np.ones(K) * locU[2]
    locU = np.vstack([xU, yU, zU]).T  # communication user locations

    # distance between nodes
    dBS = np.linalg.norm(locB - locS)  # BS-STARS distance
    dBU = np.linalg.norm(locB - locU, axis=1)  # BS-users distance
    dSU = np.linalg.norm(locS - locU, axis=1)  # IRS-users distance

    # BS-IRS channel
    alphaHatBS = 2.2  # path loss exponent
    plBS = plCalc(dBS, alphaHatBS)  # path loss
    AoD_BS = np.random.rand()
    AoA_IRS = np.random.rand()
    losBS = (np.exp(-1j * np.pi * np.arange(Ns) * np.sin(2 * np.pi * AoA_IRS))[:, None]
              @ np.exp(-1j * np.pi * np.arange(Nb) * np.sin(2 * np.pi * AoD_BS))[None, :])  # LoS component
    nlosBS = np.sqrt(0.5) * (np.random.randn(Ns, Nb) + 1j * np.random.randn(Ns, Nb))  # NLoS component
    G = np.sqrt(plBS) * (1 / np.sqrt(rf + 1)) * (np.sqrt(rf) * losBS + nlosBS)  # Rician faded link

    # BS-users channel
    alphaHatBU = 3.6  # path loss exponent
    hD = np.zeros((K, Nb), dtype=complex)  # memory allocation
    for k in range(K):
        plBU = plCalc(dBU[k], alphaHatBU)  # path loss
        hD[k, :] = np.sqrt(plBU) * np.sqrt(0.5) * (np.random.randn(1, Nb)
                                                    + 1j * np.random.randn(1, Nb))  # Rayleigh faded link

    # IRS-users channel
    alphaHatSU = 2.2  # path loss exponent
    hR = np.zeros((K, Ns), dtype=complex)  # memory allocation
    for k in range(K):
        plSU = plCalc(dSU[k], alphaHatSU)  # path loss
        AoD_IRS = np.random.rand()
        AoA_U = np.random.rand()
        losR = (np.exp(-1j * np.pi * np.arange(1) * np.sin(2 * np.pi * AoA_U))[:, None]
                 @ np.exp(-1j * np.pi * np.arange(Ns) * np.sin(2 * np.pi * AoD_IRS))[None, :])  # LoS component
        nlosR = np.sqrt(0.5) * (np.random.randn(1, Ns) + 1j * np.random.randn(1, Ns))  # NLoS component
        hR[k, :] = np.sqrt(plSU) * (1 / np.sqrt(rf + 1)) * (np.sqrt(rf) * losR + nlosR)  # Rician faded link

    # IRS-target channel
    alphaHatST = 2  # path loss exponent
    dST = 10  # distance between IRS and target
    plST = plCalc(dST, alphaHatST)  # path loss
    thetaLocal = np.radians(-30)  # azimuth angle of the target
    phiLocal = np.radians(40)  # elevation angle of the target
    gR = (np.sqrt(plST) *
          np.kron(np.exp(-1j * 2 * np.pi * 0.5 * np.sin(thetaLocal) * np.cos(phiLocal) * np.arange(xNs)),
                  np.exp(-1j * 2 * np.pi * 0.5 * np.sin(thetaLocal) * np.cos(phiLocal) * np.arange(zNs))))

    # ============= Scaling the channels
    sf = 20 / np.amax([abs(np.amax(hD)), abs(np.amax(hR)), abs(np.amax(gR))])
    hD = sf * hD
    hR = sf * hR
    gR = sf * gR
    scaledNoisePower = (sf ** 2) * arg.sigmaSquare

    return G, hD, hR, gR, scaledNoisePower, sf
