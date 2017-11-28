#!/usr/bin/python2.7
import os
import numpy as np

def psi_reader(direc_dat):
    with os.popen('pwd') as f:
        cdirec = f.readlines()[0]

    os.chdir(direc_dat)

    x_data = []; y_data = []

    with os.popen('echo psi*') as f:
        fileNames = f.readlines()[0].split()
        for i, fileName in enumerate(fileNames):
            psi_dat = np.loadtxt(fileName)
            x_data.append(float(fileName[4:-4]))
            y_data.append(psi_dat)

    os.chdir(cdirec[:-1])

    return x_data, y_data


if __name__ == "__main__":
    direc_dat = '/home/alice/Work/computational-microelectronics/project/raw-data'
    psi_reader(direc_dat)
