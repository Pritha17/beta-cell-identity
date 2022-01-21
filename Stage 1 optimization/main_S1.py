import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from pyswarms.single.global_best import GlobalBestPSO
import time
from experiments_S1 import *


if __name__ == '__main__':
    print(time.localtime())
    start_time = time.time()

    dim = 8  # number of free parameters

    # setting the parameter bounds
    filename1 = "parameter_space_S1.csv"
    df1 = pd.read_csv(filename1, decimal='.', delimiter=',')

    x_min = np.array(df1['minimum'])
    x_max = np.array(df1['maximum'])

    bounds = (x_min, x_max)

    # hyper-parameters of the PSO algorithm
    options = {'c1': 0.5, 'c2': 0.5, 'w': 0.9}

    # strategy for changing the hyper-parameter inertia weight
    strategy = {'w': "lin_decay"}

    n_particles = 1920  # number of particles for PSO
    iters = 250  # number of iterations

    for i in range (100):
        # global best PSO using the PySwarms toolkit
        optimizer = GlobalBestPSO(n_particles=n_particles, dimensions=dim, options=options, bounds=bounds, oh_strategy=None,
                                  bh_strategy="reflective", )
        cost, pos = optimizer.optimize(fitness, iters=iters)

        # saving the results to a csv file
        parameters = ['V_mG1', 'V_mGK', 'V_mPDH', 'k_HYD', 'k_PMCA', 'a_1', 'K_mPyr', 'V_O']

        df = pd.DataFrame(data={"parameters": parameters, "Values": pos, "Cost": cost})
        df.to_csv("results/estimated_parameters_S1_{0}.csv".format(i + 1), sep=',', index=False)

        print(time.localtime())
        print("end", time.time() - start_time)
