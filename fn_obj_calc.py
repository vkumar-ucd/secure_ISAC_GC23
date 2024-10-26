import numpy as np


def obj_calc(channels, xTilde, thetaVec):

    # unpack the channels
    G, hD, hR, gR, scaledNoisePower, sf = channels
    gCurrent = gR @ np.diag(thetaVec) @ G

    return np.real(np.sum(abs(gCurrent[None, :] @ xTilde) ** 2) / (sf ** 2))
