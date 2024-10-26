import numpy as np
from mosek.fusion import *
import sys
import itertools


# Define the cursor symbols
cursor_symbols = itertools.cycle(['-', '\\', '|', '/'])


# function for Hermitian operator
def Herm(x):
    """
    :param x: input complex-valued matrix
    :return: conjugate-transpose of the input matrix
    """
    return x.conj().T


# function to calculate the first set of constants
def computeParameterSet1(Nb, K, Q, gCurrent, xTildeCurrent):
    aCurrent = np.squeeze(gCurrent[None, :] @ xTildeCurrent)
    bCurrent = np.tile(aCurrent, (Nb, 1)) * np.tile(gCurrent.conj()[:, None], (1, K + Q)) + xTildeCurrent
    aR = aCurrent.real
    aI = aCurrent.imag
    aAbsSQ = abs(aCurrent) ** 2
    bCurrentR = bCurrent.real
    bCurrentI = bCurrent.imag
    bCurrentNormSQ = np.linalg.norm(bCurrent, axis=0) ** 2  # columnwise 2-norm
    delta1 = bCurrentR.T * aR[:, None] + bCurrentI.T * aI[:, None]
    delta2 = bCurrentR.T * aI[:, None] - bCurrentI.T * aR[:, None]
    return aCurrent, aAbsSQ, bCurrentR, bCurrentI, bCurrentNormSQ, delta1, delta2


# function to calculate the second set of constants
def computeParameterSet2(Nb, K, hCurrent, xTildeCurrent):
    cCurrent = np.diag(hCurrent @ xTildeCurrent[:, 0:K])
    dCurrent = np.tile(cCurrent, (Nb, 1)) * Herm(hCurrent) + xTildeCurrent[:, 0:K]
    cR = cCurrent.real
    cI = cCurrent.imag
    cAbsSQ = abs(cCurrent) ** 2
    dCurrentR = dCurrent.real
    dCurrentI = dCurrent.imag
    dCurrentNormSQ = np.linalg.norm(dCurrent, axis=0) ** 2  # columnwise 2-norm
    delta3 = dCurrentR.T * cR[:, None] + dCurrentI.T * cI[:, None]
    delta4 = dCurrentR.T * cI[:, None] - dCurrentI.T * cR[:, None]
    return cCurrent, cAbsSQ, dCurrentR, dCurrentI, dCurrentNormSQ, delta3, delta4


# function to calculate the third set of constants
def ComputeParametersSet3(k, K, Q, hCurrent, xTildeCurrent):
    # ------------ constant values for (10) ------------
    Lambda = np.real(np.tile(hCurrent[[k], :], (K + Q - 1, 1))) - np.real(xTildeCurrent[:, np.arange(K + Q) != k].T)
    LambdaTilde = np.imag(np.tile(hCurrent[[k], :], (K + Q - 1, 1))) + np.imag(
        xTildeCurrent[:, np.arange(K + Q) != k].T)
    normCSQ = np.reshape(np.linalg.norm(np.tile(Herm(hCurrent[[k], :]), (1, K + Q - 1)) 
                                        - xTildeCurrent[:, np.arange(K + Q) != k], axis=0) ** 2, (-1, 1))

    # ------------ constant values for (11) ------------
    eta = np.real(np.tile(hCurrent[[k], :], (K + Q - 1, 1))) + np.real(xTildeCurrent[:, np.arange(K + Q) != k].T)
    etaTilde = np.imag(np.tile(hCurrent[[k], :], (K + Q - 1, 1))) - np.imag(xTildeCurrent[:, np.arange(K + Q) != k].T)
    normDSQ = np.reshape(np.linalg.norm(np.tile(Herm(hCurrent[[k], :]), (1, K + Q - 1)) 
                                        + xTildeCurrent[:, np.arange(K + Q) != k], axis=0) ** 2, (-1, 1))

    # ------------ constant values for (13) ------------
    psi = np.real(np.tile(hCurrent[[k], :], (K + Q - 1, 1))) - np.imag(xTildeCurrent[:, np.arange(K + Q) != k].T)
    psiTilde = np.imag(np.tile(hCurrent[[k], :], (K + Q - 1, 1))) - np.real(xTildeCurrent[:, np.arange(K + Q) != k].T)
    normESQ = np.reshape(np.linalg.norm(np.tile(Herm(hCurrent[[k], :]), (1, K + Q - 1)) 
                                        + 1j * xTildeCurrent[:, np.arange(K + Q) != k], axis=0) ** 2, (-1, 1))

    # ------------ constant values for (14) ------------
    phi = np.real(np.tile(hCurrent[[k], :], (K + Q - 1, 1))) + np.imag(xTildeCurrent[:, np.arange(K + Q) != k].T)
    phiTilde = np.imag(np.tile(hCurrent[[k], :], (K + Q - 1, 1))) + np.real(xTildeCurrent[:, np.arange(K + Q) != k].T)
    normFSQ = np.reshape(np.linalg.norm(np.tile(Herm(hCurrent[[k], :]), (1, K + Q - 1)) 
                                        - 1j * xTildeCurrent[:, np.arange(K + Q) != k], axis=0) ** 2, (-1, 1))

    return Lambda, LambdaTilde, normCSQ, eta, etaTilde, normDSQ, psi, psiTilde, normESQ, phi, phiTilde, normFSQ


# function to calculate the fourth set of constants
def computeParameterSet4(k, gCurrent, xTildeCurrent):
    eta1 = np.real(gCurrent) - np.real(xTildeCurrent[:, k])
    eta2 = np.imag(gCurrent) + np.imag(xTildeCurrent[:, k])
    etaNormSQ = np.linalg.norm(Herm(gCurrent) - xTildeCurrent[:, k]) ** 2

    etaBar1 = np.real(gCurrent) + np.real(xTildeCurrent[:, k])
    etaBar2 = np.imag(gCurrent) - np.imag(xTildeCurrent[:, k])
    etaBarNormSQ = np.linalg.norm(Herm(gCurrent) + xTildeCurrent[:, k]) ** 2

    chi1 = np.real(gCurrent) - np.imag(xTildeCurrent[:, k])
    chi2 = np.imag(gCurrent) - np.real(xTildeCurrent[:, k])
    chiNormSQ = np.linalg.norm(Herm(gCurrent) + 1j * xTildeCurrent[:, k]) ** 2

    chiBar1 = np.real(gCurrent) + np.imag(xTildeCurrent[:, k])
    chiBar2 = np.imag(gCurrent) + np.real(xTildeCurrent[:, k])
    chiBarNormSQ = np.linalg.norm(Herm(gCurrent) - 1j * xTildeCurrent[:, k]) ** 2

    return eta1, eta2, etaBar1, etaBar2, chi1, chi2, chiBar1, chiBar2, etaNormSQ, etaBarNormSQ, chiNormSQ, chiBarNormSQ


# function to maximize BPG
def maximize_BPG(arg, channels, xTildeCurrent, thetaVecCurrent):

    # unpack parameter
    Nb, K, Q, Ns, Pmax, gamma, gammaR, zeta = arg.Nb, arg.K, arg.Q, arg.Ns, arg.Pmax, arg.gamma, arg.gammaR, arg.zeta

    # unpack channels
    G, hD, hRelayed, gRelayed, scaledNoisePower, sf = channels
    hCurrent = hD + hRelayed @ np.diag(thetaVecCurrent) @ G
    gCurrent = gRelayed @ np.diag(thetaVecCurrent) @ G
    thetaVecNormSqCurrent = np.linalg.norm(thetaVecCurrent) ** 2

    # ------ MOSEK model
    myModel = Model()

    # ------------- variables --------------------
    xtildeR = myModel.variable('xtildeR', [Nb, K + Q], Domain.unbounded())  # real component of variable xtilde
    xtildeI = myModel.variable('xtildeI', [Nb, K + Q], Domain.unbounded())  # imaginary component of variable xtilde
    thetaR = myModel.variable('thetaR', [Ns, 1], Domain.unbounded())  # real component of variable theta
    thetaI = myModel.variable('thetaI', [Ns, 1], Domain.unbounded())  # imaginary component of variable theta
    t = myModel.variable('t', [K, K + Q], Domain.unbounded())  # variable t
    tBar = myModel.variable('tBar', [K, K + Q], Domain.unbounded())  # variable tBar
    tau = myModel.variable('tau', K, Domain.unbounded())  # variable tau
    tauBar = myModel.variable('tauBar', K, Domain.unbounded())  # variable tauBar
    xi = myModel.variable('xi', 1, Domain.unbounded())  # slack variable

    xtildeRTranspose = xtildeR.transpose()
    xtildeITranspose = xtildeI.transpose()

    # ------ BS-user channels
    hR = Expr.sub(Expr.sub(Expr.sub(Expr.add(hD.real,
                                             Expr.mul(
                                                 Expr.mulElm(hRelayed.real, Expr.transpose(Expr.repeat(thetaR, K, 1))),
                                                 G.real)),
                                    Expr.mul(Expr.mulElm(hRelayed.real, Expr.transpose(Expr.repeat(thetaI, K, 1))),
                                             G.imag)),
                           Expr.mul(Expr.mulElm(hRelayed.imag, Expr.transpose(Expr.repeat(thetaR, K, 1))), G.imag)),
                  Expr.mul(Expr.mulElm(hRelayed.imag, Expr.transpose(Expr.repeat(thetaI, K, 1))), G.real))
    # real component of the effective BS-user channel
    hRTranspose = Expr.transpose(hR)  # transpose of hR

    hI = Expr.sub(Expr.add(Expr.add(Expr.add(hD.imag,
                                             Expr.mul(
                                                 Expr.mulElm(hRelayed.real, Expr.transpose(Expr.repeat(thetaR, K, 1))),
                                                 G.imag)),
                                    Expr.mul(Expr.mulElm(hRelayed.real, Expr.transpose(Expr.repeat(thetaI, K, 1))),
                                             G.real)),
                           Expr.mul(Expr.mulElm(hRelayed.imag, Expr.transpose(Expr.repeat(thetaR, K, 1))), G.real)),
                  Expr.mul(Expr.mulElm(hRelayed.imag, Expr.transpose(Expr.repeat(thetaI, K, 1))), G.imag))
    # imaginary component of the effective BS-user channel
    hITranspose = Expr.transpose(hI)  # transpose of hI

    # ------ BS-target channel
    gR = Expr.sub(Expr.sub(Expr.sub(Expr.mul(Expr.mulElm(gRelayed[None, :].real, Expr.transpose(thetaR)), G.real),
                                    Expr.mul(Expr.mulElm(gRelayed[None, :].real, Expr.transpose(thetaI)), G.imag)),
                           Expr.mul(Expr.mulElm(gRelayed[None, :].imag, Expr.transpose(thetaR)), G.imag)),
                  Expr.mul(Expr.mulElm(gRelayed[None, :].imag, Expr.transpose(thetaI)), G.real))
    # real component of the effective BS-target channel
    gRTranspose = Expr.transpose(gR)  # transpose of gR

    gI = Expr.sub(Expr.add(Expr.add(Expr.mul(Expr.mulElm(gRelayed[None, :].real, Expr.transpose(thetaR)), G.imag),
                                    Expr.mul(Expr.mulElm(gRelayed[None, :].real, Expr.transpose(thetaI)), G.real)),
                           Expr.mul(Expr.mulElm(gRelayed[None, :].imag, Expr.transpose(thetaR)), G.real)),
                  Expr.mul(Expr.mulElm(gRelayed[None, :].imag, Expr.transpose(thetaI)), G.imag))
    # imaginary component of the effective BS-user channel
    gITranspose = Expr.transpose(gI)  # transpose of gI

    obj = Expr.add(xi, Expr.mul(zeta, Expr.sub(Expr.add(Expr.dot(2 * thetaVecCurrent.real, thetaR),
                                                        Expr.dot(2 * thetaVecCurrent.imag, thetaI)),
                                               thetaVecNormSqCurrent)))
    myModel.objective("obj", ObjectiveSense.Maximize, obj)  # objective function

    # ------ calculating constants in objective function
    (aCurrent, aAbsSQ, bCurrentR, bCurrentI,
     bCurrentNormSQ, delta1, delta2) = computeParameterSet1(Nb, K, Q, gCurrent, xTildeCurrent)
    aR = aCurrent.real
    aI = aCurrent.imag

    # ======================== constraints =============================

    # ------------- constraint for slack variable xi in objective (see 18b)
    slackLHS = Expr.sub(Expr.sub(
        Expr.add(Expr.add(Expr.add(Expr.sum(Expr.mul(delta1, gRTranspose)), Expr.sum(Expr.mul(delta2, gITranspose))),
                          Expr.sum(Expr.mulDiag(bCurrentR.T, xtildeR))),
                 Expr.sum(Expr.mulDiag(bCurrentI.T, xtildeI))),
        np.sum(0.5 * bCurrentNormSQ + aAbsSQ)), xi)
    slackRHS = Expr.flatten(Expr.sub(Expr.add(Expr.mulElm(np.tile(aR, (Nb, 1)), Expr.repeat(gRTranspose, K + Q, 1)),
                                              Expr.mulElm(np.tile(aI, (Nb, 1)), Expr.repeat(gITranspose, K + Q, 1))),
                                     xtildeR))
    slackRHS = Expr.hstack(slackRHS, Expr.flatten(
        Expr.sub(Expr.sub(Expr.mulElm(np.tile(aI, (Nb, 1)), Expr.repeat(gRTranspose, K + Q, 1)),
                          Expr.mulElm(np.tile(aR, (Nb, 1)), Expr.repeat(gITranspose, K + Q, 1))), xtildeI)))
    slackRHS = Expr.vstack(Expr.mul(Expr.flatten(slackRHS), np.sqrt(0.5)), Expr.mul(Expr.sub(slackLHS, 1), 0.5))
    slackLHS = Expr.mul(Expr.add(slackLHS, 1), 0.5)
    myModel.constraint(Expr.vstack(slackLHS, slackRHS), Domain.inQCone())

    # ------------- transmit power constraint (see 7d)
    tpCone = Expr.flatten(Expr.hstack(xtildeR, xtildeI))
    myModel.constraint(Expr.vstack(np.sqrt(Pmax), tpCone), Domain.inQCone())

    # ------------- relaxed unit-modulus constraints (see 18i)
    myModel.constraint(Expr.hstack(Expr.constTerm(Matrix.ones(Ns, 1)), Expr.hstack(thetaR, thetaI)), Domain.inQCone())

    # ------------- computing the second set of constants
    (cCurrent, cAbsSQ, dCurrentR, dCurrentI,
     dCurrentNormSQ, delta3, delta4) = computeParameterSet2(Nb, K, hCurrent, xTildeCurrent)
    cR = cCurrent.real
    cI = cCurrent.imag

    for k in range(K):
        # --------------- constraint (18c)
        lhsB = Expr.sub(
            Expr.mul(Expr.sub(Expr.add(Expr.add(Expr.add(Expr.dot(delta3[k, :], hR.slice([k, 0], [k + 1, Nb])),
                                                         Expr.dot(delta4[k, :], hI.slice([k, 0], [k + 1, Nb]))),
                                                Expr.dot(dCurrentR[:, k].T, xtildeR.slice([0, k], [Nb, k + 1]))),
                                       Expr.dot(dCurrentI[:, k].T, xtildeI.slice([0, k], [Nb, k + 1]))),
                              0.5 * dCurrentNormSQ[k] + cAbsSQ[k]), 1 / (gamma)), scaledNoisePower)

        rhsB = Expr.hstack(Expr.hstack(
            Expr.hstack(Expr.hstack(Expr.reshape(t.pick([[k, j] for j in range(K + Q) if j != k]), 1, K + Q - 1),
                                    Expr.reshape(tBar.pick([[k, j] for j in range(K + Q) if j != k]), 1, K + Q - 1)),
                        Expr.mul(Expr.sub(Expr.add(Expr.mul(hR.slice([k, 0], [k + 1, Nb]), cR[k]),
                                                   Expr.mul(hI.slice([k, 0], [k + 1, Nb]), cI[k])),
                                          xtildeRTranspose.slice([k, 0], [k + 1, Nb])),
                                 np.sqrt(1 / (2 * gamma)))),
            Expr.mul(Expr.sub(Expr.sub(Expr.mul(hR.slice([k, 0], [k + 1, Nb]), cI[k]),
                                       Expr.mul(hI.slice([k, 0], [k + 1, Nb]), cR[k])),
                              xtildeITranspose.slice([k, 0], [k + 1, Nb])),
                     np.sqrt(1 / (2 * gamma)))),
                           Expr.mul(Expr.sub(lhsB, 1), 1 / 2))

        lhsB = Expr.mul(Expr.add(lhsB, 1), 1 / 2)

        myModel.constraint(Expr.hstack(lhsB, rhsB), Domain.inQCone())

        # --------------- computing the third set of constants

        (Lambda, LambdaTilde, normCSQ, eta, etaTilde, normDSQ, psi, psiTilde,
         normESQ, phi, phiTilde, normFSQ) = ComputeParametersSet3(k, K, Q, hCurrent, xTildeCurrent)

        # --------------- constraint (18d)

        lhsC = Expr.add(Expr.add(Expr.reshape(t.pick([[k, l] for l in range(K + Q) if l != k]), K + Q - 1, 1),
                                 Expr.reshape(Expr.mulDiag(0.5 * Lambda,
                                                           Expr.sub(Expr.repeat(hRTranspose.slice([0, k], [Nb, k + 1]),
                                                                                K + Q - 1, 1),
                                                                    Expr.transpose(Expr.reshape(
                                                                            Expr.flatten(xtildeRTranspose).pick(
                                                                                    [[l] for l in range((K + Q) * Nb) if
                                                                                     ((l < k * Nb) or (
                                                                                                 l >= k * Nb + Nb))]),
                                                                            (K + Q) - 1, Nb)))), K + Q - 1, 1)),
                        Expr.reshape(Expr.mulDiag(0.5 * LambdaTilde,
                                                  Expr.add(
                                                      Expr.repeat(hITranspose.slice([0, k], [Nb, k + 1]), K + Q - 1, 1),
                                                      Expr.transpose(Expr.reshape(
                                                              Expr.flatten(xtildeITranspose).pick(
                                                                      [[l] for l in range((K + Q) * Nb) if
                                                                       ((l < k * Nb) or (l >= k * Nb + Nb))]),
                                                              (K + Q) - 1, Nb)))), K + Q - 1, 1))

        rhsC = Expr.mul(Expr.sub(lhsC, 1 + 0.25 * normCSQ), 0.5)
        lhsC = Expr.mul(Expr.add(lhsC, 1 - 0.25 * normCSQ), 0.5)

        rhsC = Expr.hstack(Expr.hstack(rhsC,
                                       Expr.mul(Expr.sub(
                                               Expr.reshape(Expr.flatten(xtildeITranspose).pick(
                                                       [[l] for l in range((K + Q) * Nb) if
                                                        ((l < k * Nb) or (l >= k * Nb + Nb))]), K + Q - 1, Nb),
                                               Expr.repeat(hI.slice([k, 0], [k + 1, Nb]), K + Q - 1, 0)), 0.5)),
                           Expr.mul(Expr.add(
                                   Expr.reshape(Expr.flatten(xtildeRTranspose).pick(
                                           [[l] for l in range((K + Q) * Nb) if ((l < k * Nb) or (l >= k * Nb + Nb))]),
                                                K + Q - 1, Nb),
                                   Expr.repeat(hR.slice([k, 0], [k + 1, Nb]), K + Q - 1, 0)), 0.5))

        myModel.constraint(Expr.hstack(lhsC, rhsC), Domain.inQCone())

        # --------------- constraint (18e)

        lhsD = Expr.add(Expr.add(Expr.reshape(t.pick([[k, l] for l in range(K + Q) if l != k]), K + Q - 1, 1),
                                 Expr.reshape(Expr.mulDiag(0.5 * eta,
                                                           Expr.add(Expr.repeat(hRTranspose.slice([0, k], [Nb, k + 1]),
                                                                                K + Q - 1, 1),
                                                                    Expr.transpose(Expr.reshape(
                                                                            Expr.flatten(xtildeRTranspose).pick(
                                                                                    [[l] for l in range((K + Q) * Nb) if
                                                                                     ((l < k * Nb) or (
                                                                                                 l >= k * Nb + Nb))]),
                                                                            (K + Q) - 1, Nb)))), K + Q - 1, 1)),
                        Expr.reshape(Expr.mulDiag(0.5 * etaTilde,
                                                  Expr.sub(
                                                      Expr.repeat(hITranspose.slice([0, k], [Nb, k + 1]), K + Q - 1, 1),
                                                      Expr.transpose(Expr.reshape(
                                                              Expr.flatten(xtildeITranspose).pick(
                                                                      [[l] for l in range((K + Q) * Nb) if
                                                                       ((l < k * Nb) or (l >= k * Nb + Nb))]),
                                                              (K + Q) - 1, Nb)))), K + Q - 1, 1))

        rhsD = Expr.mul(Expr.sub(lhsD, 1 + 0.25 * normDSQ), 0.5)
        lhsD = Expr.mul(Expr.add(lhsD, 1 - 0.25 * normDSQ), 0.5)

        rhsD = Expr.hstack(Expr.hstack(rhsD,
                                       Expr.mul(Expr.add(
                                               Expr.reshape(Expr.flatten(xtildeITranspose).pick(
                                                       [[l] for l in range((K + Q) * Nb) if
                                                        ((l < k * Nb) or (l >= k * Nb + Nb))]), K + Q - 1, Nb),
                                               Expr.repeat(hI.slice([k, 0], [k + 1, Nb]), K + Q - 1, 0)), 0.5)),
                           Expr.mul(Expr.sub(Expr.repeat(hR.slice([k, 0], [k + 1, Nb]), K + Q - 1, 0),
                                             Expr.reshape(
                                                     Expr.flatten(xtildeRTranspose).pick(
                                                             [[l] for l in range((K + Q) * Nb) if
                                                              ((l < k * Nb) or (l >= k * Nb + Nb))]), K + Q - 1, Nb)),
                                    0.5))

        myModel.constraint(Expr.hstack(lhsD, rhsD), Domain.inQCone())

        # --------------- constraint (18f)

        lhsE = Expr.add(Expr.add(Expr.reshape(tBar.pick([[k, l] for l in range(K + Q) if l != k]), K + Q - 1, 1),
                                 Expr.reshape(Expr.mulDiag(0.5 * psi,
                                                           Expr.sub(Expr.repeat(hRTranspose.slice([0, k], [Nb, k + 1]),
                                                                                K + Q - 1, 1),
                                                                    Expr.transpose(Expr.reshape(
                                                                            Expr.flatten(xtildeITranspose).pick(
                                                                                    [[l] for l in range((K + Q) * Nb) if
                                                                                     ((l < k * Nb) or (
                                                                                                 l >= k * Nb + Nb))]),
                                                                            K + Q - 1, Nb)))), K + Q - 1, 1)),
                        Expr.reshape(Expr.mulDiag(0.5 * psiTilde,
                                                  Expr.sub(
                                                      Expr.repeat(hITranspose.slice([0, k], [Nb, k + 1]), K + Q - 1, 1),
                                                      Expr.transpose(Expr.reshape(
                                                              Expr.flatten(xtildeRTranspose).pick(
                                                                      [[l] for l in range((K + Q) * Nb) if
                                                                       ((l < k * Nb) or (l >= k * Nb + Nb))]),
                                                              K + Q - 1, Nb)))), K + Q - 1, 1))

        rhsE = Expr.mul(Expr.sub(lhsE, 1 + 0.25 * normESQ), 0.5)
        lhsE = Expr.mul(Expr.add(lhsE, 1 - 0.25 * normESQ), 0.5)

        rhsE = Expr.hstack(Expr.hstack(rhsE,
                                       Expr.mul(Expr.add(Expr.reshape(
                                               Expr.flatten(xtildeRTranspose).pick([[l] for l in range((K + Q) * Nb) if
                                                                                    ((l < k * Nb) or (
                                                                                                l >= k * Nb + Nb))]),
                                               K + Q - 1, Nb),
                                               Expr.repeat(hI.slice([k, 0], [k + 1, Nb]), K + Q - 1, 0)), 0.5)),
                           Expr.mul(Expr.add(Expr.reshape(
                                   Expr.flatten(xtildeITranspose).pick(
                                           [[l] for l in range((K + Q) * Nb) if ((l < k * Nb) or (l >= k * Nb + Nb))]),
                                   K + Q - 1, Nb),
                                   Expr.repeat(hR.slice([k, 0], [k + 1, Nb]), K + Q - 1, 0)), 0.5))

        myModel.constraint(Expr.hstack(lhsE, rhsE), Domain.inQCone())

        # --------------- constraint (18g)

        lhsF = Expr.add(Expr.add(Expr.reshape(tBar.pick([[k, l] for l in range(K + Q) if l != k]), K + Q - 1, 1),
                                 Expr.reshape(Expr.mulDiag(0.5 * phi,
                                                           Expr.add(Expr.repeat(hRTranspose.slice([0, k], [Nb, k + 1]),
                                                                                K + Q - 1, 1),
                                                                    Expr.transpose(Expr.reshape(
                                                                            Expr.flatten(xtildeITranspose).pick(
                                                                                    [[l] for l in range((K + Q) * Nb) if
                                                                                     ((l < k * Nb) or (
                                                                                                 l >= k * Nb + Nb))]),
                                                                            K + Q - 1, Nb)))), K + Q - 1, 1)),
                        Expr.reshape(Expr.mulDiag(0.5 * phiTilde,
                                                  Expr.add(
                                                      Expr.repeat(hITranspose.slice([0, k], [Nb, k + 1]), K + Q - 1, 1),
                                                      Expr.transpose(Expr.reshape(
                                                              Expr.flatten(xtildeRTranspose).pick(
                                                                      [[l] for l in range((K + Q) * Nb) if
                                                                       ((l < k * Nb) or (l >= k * Nb + Nb))]),
                                                              K + Q - 1, Nb)))), K + Q - 1, 1))

        rhsF = Expr.mul(Expr.sub(lhsF, 1 + 0.25 * normFSQ), 0.5)
        lhsF = Expr.mul(Expr.add(lhsF, 1 - 0.25 * normFSQ), 0.5)

        rhsF = Expr.hstack(Expr.hstack(rhsF,
                                       Expr.mul(Expr.sub(Expr.reshape(
                                               Expr.flatten(xtildeRTranspose).pick([[l] for l in range((K + Q) * Nb) if
                                                                                    ((l < k * Nb) or (
                                                                                                l >= k * Nb + Nb))]),
                                               K + Q - 1, Nb),
                                               Expr.repeat(hI.slice([k, 0], [k + 1, Nb]), K + Q - 1, 0)), 0.5)),
                           Expr.mul(Expr.sub(Expr.repeat(hR.slice([k, 0], [k + 1, Nb]), K + Q - 1, 0),
                                             Expr.reshape(
                                                     Expr.flatten(xtildeITranspose).pick(
                                                             [[l] for l in range((K + Q) * Nb) if
                                                              ((l < k * Nb) or (l >= k * Nb + Nb))]), K + Q - 1, Nb)),
                                    0.5))

        myModel.constraint(Expr.hstack(lhsF, rhsF), Domain.inQCone())

        # --------------- constraint (18g)
        secureLHS = Expr.add(
            Expr.sub(Expr.add(Expr.add(Expr.add(Expr.sum(Expr.mul(delta1[np.arange(K + Q) != k, :], gRTranspose)),
                                                Expr.sum(Expr.mul(delta2[np.arange(K + Q) != k, :], gITranspose))),
                                       Expr.sum(Expr.mulDiag(bCurrentR[:, np.arange(K + Q) != k].T,
                                                             Expr.transpose(Expr.reshape(
                                                                     Expr.flatten(xtildeRTranspose).pick(
                                                                             [[l] for l in range((K + Q) * Nb) if
                                                                              ((l < k * Nb) or (l >= k * Nb + Nb))]),
                                                                     (K + Q) - 1, Nb))))),
                              Expr.sum(Expr.mulDiag(bCurrentI[:, np.arange(K + Q) != k].T, Expr.transpose(Expr.reshape(
                                      Expr.flatten(xtildeITranspose).pick([[l] for l in range((K + Q) * Nb) if
                                                                           ((l < k * Nb) or (l >= k * Nb + Nb))]),
                                      (K + Q) - 1, Nb))))),
                     np.sum(0.5 * bCurrentNormSQ[np.arange(K + Q) != k] + aAbsSQ[np.arange(K + Q) != k])), scaledNoisePower)

        secureRHS = Expr.flatten(Expr.sub(
            Expr.add(Expr.mulElm(np.tile(aR[np.arange(K + Q) != k], (Nb, 1)), Expr.repeat(gRTranspose, K + Q - 1, 1)),
                     Expr.mulElm(np.tile(aI[np.arange(K + Q) != k], (Nb, 1)), Expr.repeat(gITranspose, K + Q - 1, 1))),
            Expr.transpose(Expr.reshape(
                    Expr.flatten(xtildeRTranspose).pick(
                            [[l] for l in range((K + Q) * Nb) if ((l < k * Nb) or (l >= k * Nb + Nb))]), (K + Q) - 1,
                    Nb))))
        secureRHS = Expr.vstack(secureRHS, Expr.flatten(Expr.sub(
            Expr.sub(Expr.mulElm(np.tile(aI[np.arange(K + Q) != k], (Nb, 1)), Expr.repeat(gRTranspose, K + Q - 1, 1)),
                     Expr.mulElm(np.tile(aR[np.arange(K + Q) != k], (Nb, 1)), Expr.repeat(gITranspose, K + Q - 1, 1))),
            Expr.transpose(Expr.reshape(Expr.flatten(xtildeITranspose).pick(
                    [[l] for l in range((K + Q) * Nb) if ((l < k * Nb) or (l >= k * Nb + Nb))]), (K + Q) - 1, Nb)))))
        secureRHS = Expr.vstack(
            Expr.vstack(Expr.mul(tau.pick([k]), 1 / np.sqrt(gammaR)), Expr.mul(tauBar.pick([k]), 1 / np.sqrt(gammaR))),
            Expr.mul(secureRHS, np.sqrt(0.5)))
        secureRHS = Expr.vstack(secureRHS, Expr.mul(Expr.sub(secureLHS, 1), 0.5))
        secureLHS = Expr.mul(Expr.add(secureLHS, 1), 0.5)

        myModel.constraint(Expr.vstack(secureLHS, secureRHS), Domain.inQCone())

        # ------------------------------------
        (eta1, eta2, etaBar1, etaBar2, chi1, chi2, chiBar1, chiBar2, etaNormSQ, etaBarNormSQ,
         chiNormSQ, chiBarNormSQ) = computeParameterSet4(k, gCurrent, xTildeCurrent)

        etaLHS = Expr.add(
            Expr.add(tau.pick([k]), Expr.dot(0.5 * eta1, Expr.sub(gRTranspose, xtildeR.slice([0, k], [Nb, k + 1])))),
            Expr.dot(0.5 * eta2, Expr.add(gITranspose, xtildeI.slice([0, k], [Nb, k + 1]))))

        etaRHS = Expr.mul(Expr.sub(etaLHS, 1 + 0.25 * etaNormSQ), 0.5)
        etaLHS = Expr.mul(Expr.add(etaLHS, 1 - 0.25 * etaNormSQ), 0.5)

        etaRHS = Expr.hstack(
            Expr.hstack(etaRHS, Expr.mul(Expr.sub(Expr.reshape(xtildeI.slice([0, k], [Nb, k + 1]), 1, Nb), gI), 0.5)),
            Expr.mul(Expr.add(Expr.reshape(xtildeR.slice([0, k], [Nb, k + 1]), 1, Nb), gR), 0.5))

        myModel.constraint(Expr.hstack(etaLHS, etaRHS), Domain.inQCone())

        # ------------------------------------
        etaBarLHS = Expr.add(
            Expr.add(tau.pick([k]), Expr.dot(0.5 * etaBar1, Expr.add(gRTranspose, xtildeR.slice([0, k], [Nb, k + 1])))),
            Expr.dot(0.5 * etaBar2, Expr.sub(gITranspose, xtildeI.slice([0, k], [Nb, k + 1]))))

        etaBarRHS = Expr.mul(Expr.sub(etaBarLHS, 1 + 0.25 * etaBarNormSQ), 0.5)
        etaBarLHS = Expr.mul(Expr.add(etaBarLHS, 1 - 0.25 * etaBarNormSQ), 0.5)

        etaBarRHS = Expr.hstack(Expr.hstack(etaBarRHS, Expr.mul(
            Expr.add(Expr.reshape(xtildeI.slice([0, k], [Nb, k + 1]), 1, Nb), gI), -0.5)),
                                Expr.mul(Expr.sub(gR, Expr.reshape(xtildeR.slice([0, k], [Nb, k + 1]), 1, Nb)), 0.5))

        myModel.constraint(Expr.hstack(etaBarLHS, etaBarRHS), Domain.inQCone())

        # ------------------------------------
        chiLHS = Expr.add(
            Expr.add(tauBar.pick([k]), Expr.dot(0.5 * chi1, Expr.sub(gRTranspose, xtildeI.slice([0, k], [Nb, k + 1])))),
            Expr.dot(0.5 * chi2, Expr.sub(gITranspose, xtildeR.slice([0, k], [Nb, k + 1]))))

        chiRHS = Expr.mul(Expr.sub(chiLHS, 1 + 0.25 * chiNormSQ), 0.5)
        chiLHS = Expr.mul(Expr.add(chiLHS, 1 - 0.25 * chiNormSQ), 0.5)

        chiRHS = Expr.hstack(
            Expr.hstack(chiRHS, Expr.mul(Expr.add(Expr.reshape(xtildeR.slice([0, k], [Nb, k + 1]), 1, Nb), gI), -0.5)),
            Expr.mul(Expr.add(Expr.reshape(xtildeI.slice([0, k], [Nb, k + 1]), 1, Nb), gR), 0.5))

        myModel.constraint(Expr.hstack(chiLHS, chiRHS), Domain.inQCone())

        # ------------------------------------
        chiBarLHS = Expr.add(Expr.add(tauBar.pick([k]), Expr.dot(0.5 * chiBar1, Expr.add(gRTranspose,
                                                                                         xtildeI.slice([0, k],
                                                                                                       [Nb, k + 1])))),
                             Expr.dot(0.5 * chiBar2, Expr.add(gITranspose, xtildeR.slice([0, k], [Nb, k + 1]))))

        chiBarRHS = Expr.mul(Expr.sub(chiBarLHS, 1 + 0.25 * chiBarNormSQ), 0.5)
        chiBarLHS = Expr.mul(Expr.add(chiBarLHS, 1 - 0.25 * chiBarNormSQ), 0.5)

        chiBarRHS = Expr.hstack(Expr.hstack(chiBarRHS, Expr.mul(
            Expr.sub(Expr.reshape(xtildeR.slice([0, k], [Nb, k + 1]), 1, Nb), gI), 0.5)),
                                Expr.mul(Expr.sub(gR, Expr.reshape(xtildeI.slice([0, k], [Nb, k + 1]), 1, Nb)), 0.5))

        myModel.constraint(Expr.hstack(chiBarLHS, chiBarRHS), Domain.inQCone())

    try:
        myModel.solve()
        xtilde = np.reshape(xtildeR.level() + 1j * xtildeI.level(), (Nb, K + Q))
        theta = thetaR.level() + 1j * thetaI.level()
        myModel.dispose()
        solFlag = 1
        return solFlag, xtilde, theta
    except OptimizeError:
        solFlag = 0
        return solFlag, np.zeros((Nb, K + Q), dtype=complex), np.zeros(Ns, dtype=complex)
    except SolutionError:
        solFlag = 0
        return solFlag, np.zeros((Nb, K + Q), dtype=complex), np.zeros(Ns, dtype=complex)
    except Exception:
        solFlag = 0
        return solFlag, np.zeros((Nb, K + Q), dtype=complex), np.zeros(Ns, dtype=complex)