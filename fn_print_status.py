import numpy as np


# function for pow2db conversion
def pow2db(x):
    """
    :param x: input variable in linear scale
    :return: output in dB scale 
    """
    # returns the power value
    return 10*np.log10(x)


# function to print the constraint status
def print_status(arg, channels, xTilde, thetaVec):
    # unpack channels
    G, hD, hR, gR, scaledNoisePower, sf = channels
    h = hD + hR @ np.diag(thetaVec) @ G
    g = gR @ np.diag(thetaVec) @ G

    # compute parameters of interest
    scaledBPG = np.sum(abs(g[None, :] @ xTilde) ** 2)
    sinrNumC = abs(np.diag(h @ xTilde[:, 0:arg.K])) ** 2
    sinrDenC = scaledNoisePower + np.sum(abs(np.einsum('ij,jk->ik', h, xTilde)) ** 2, 1) - sinrNumC
    sinrNumR = abs(g @ (xTilde[:, 0:arg.K])) ** 2
    sinrDenR = scaledNoisePower + np.sum(abs(np.einsum('ij,jk->ik', g[None, :], xTilde)) ** 2, 1) - sinrNumR
    sinrC = sinrNumC / sinrDenC
    sinrR = sinrNumR / sinrDenR
    print("\n========================================")
    print(f"Scaled beampattern gain = {scaledBPG:.3f} W")
    print("----------------------------------------")
    print(f"SINR at users should be > = {pow2db(arg.gamma):.3f} dB")
    for k in range(arg.K):
        print(f"User {k} SINR = {pow2db(sinrC[k]):.3f} dB")
    print("----------------------------------------")
    print(f"SINR at the target should be < =  {pow2db(arg.gammaR):.3f} dB")
    for k in range(arg.K):
        print(f"SINR at target to decode user {k} = {pow2db(sinrR[k]):.3f} dB")
    print("----------------------------------------")
    print(f"Transmit power = {np.linalg.norm(xTilde, 'fro') ** 2:.3f} W")
    print(f"Transmit power budget = {arg.Pmax:.3f} W")
    return []
