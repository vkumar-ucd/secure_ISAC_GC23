# SCA-based beamforming optimization for IRS-enabled secure integrated sensing and communication
# Authors: (*, #)Vaibhav Kumar, (#, ^)Marwa Chafii, ($)A. Lee Swindlehurst, (*)Le-Nam Tran, and (*)Mark F. Flanagan
# DOI: 10.1109/GLOBECOM54140.2023.10437283
# Conference: IEEE Global Communications Conference, Kuala Lumpur, Malaysia, 2023
# (*): School of Electrical and Electronic Engineering, University College Dublin, Ireland
# (#): Engineering Division, New York University Abu Dhabi, United Arab Emirates
# (^): NYU WIRELESS, NYU Tandon School of Engineering, New York, United States of America
# ($): Center for Pervasive Communications and Computing, University of California, Irvine, CA, United States of America
# email: vaibhav.kumar@ieee.org / vaibhav.kumar@nyu.edu


from fn_generate_channels import *
from fn_find_initial_points import *
from fn_maximize_BPG import *
from fn_obj_calc import *
import matplotlib.pyplot as plt

np.random.seed(0)


# system parameters
class SysPar:
    def __init__(self):
        self.Nb = 4  # number of BS antennas
        self.Q = 4  # number of radar beamformers
        self.Mx = 6  # number of IRS rows
        self.K = 3  # number of communication users
        self.gamma = db2pow(10)  # SINR threshold for communication users
        self.gammaR = db2pow(0)  # SINR threshold for information leakage
        self.sigmaSquare = db2pow(-90 - 30)  # average noise power
        self.Pmax = db2pow(40 - 30)  # transmit power budget at BS
        self.epsilon = 1e-3  # convergence tolerance
        self.zeta = 0.001  # regularization parameter


# main function
def main():
    sp = SysPar()
    sp.Mz = sp.Mx  # number of IRS columns
    sp.Ns = sp.Mx * sp.Mz  # number of IRS elements

    # generate channel coefficients
    channels = generate_channels(sp)

    # obtain initial points
    breakFlag, xTildeCurrent, thetaVecCurrent = find_initial_points(sp, channels)

    if breakFlag == 1:
        print(f"Could not find initial feasible points. \nTry other channel by changing the seed.")
    else:
        relative_change = 1e3  # arbitrary large number
        true_BPG = []
        print("Maximizing beampattern gain", end='')
        while relative_change > sp.epsilon:
            sys.stdout.write('\rMaximizing beampattern gain ' + next(cursor_symbols))
            sys.stdout.flush()
            solFlag, xTildeCurrent, thetaVecCurrent = maximize_BPG(sp, channels, xTildeCurrent, thetaVecCurrent)
            if solFlag == 1:
                true_BPG.append(obj_calc(channels, xTildeCurrent, thetaVecCurrent))
            else:
                print(f"Could not solve the problem. \nTry other channel by changing the seed.")
                break

            if len(true_BPG) > 10:
                relative_change = abs(true_BPG[-1] - true_BPG[-2]) / true_BPG[-2]

        plt.plot(np.arange(len(true_BPG)), true_BPG)
        plt.xlabel('Iteration number')
        plt.ylabel('Instantaneous beampattern gain')
        plt.show()


if __name__ == '__main__':
    main()

