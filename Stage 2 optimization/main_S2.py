import numpy as np
import pandas as pd
from pyswarms.single.global_best import GlobalBestPSO
import time
from experiments_S2 import *


if __name__ == '__main__':
    print(time.localtime())
    start_time = time.time()

    dim = 37  # number of free parameters

    # setting the parameter bounds
    filename1 = "parameter_space_S2.csv"
    df1 = pd.read_csv(filename1, decimal='.', delimiter=',')

    x_min = np.array(df1['minimum'])
    x_max = np.array(df1['maximum'])

    bounds = (x_min, x_max)

    # hyper-parameters of the PSO algorithm
    options = {'c1': 0.5, 'c2': 0.5, 'w': 0.9}

    # strategy for changing the hyper-parameter inertia weight
    strategy = {'w': "lin_decay"}

    n_particles = 1120  # number of particles for PSO
    iters = 300  # number of iterations

    for i in range (100):
        # global best PSO using the PySwarms toolkit
        optimizer = GlobalBestPSO(n_particles=n_particles, dimensions=dim, options=options, bounds=bounds, oh_strategy=None,
                                  bh_strategy="reflective", )
        cost, pos = optimizer.optimize(fitness, iters=iters)

        # saving the results to a csv file
        parameters = ['AKT', 'AKTp', 'GSK3', 'GSK3p', 'JNK', 'JNKp', 'p38', 'p38p', 'FOXO1', 'FOXO1p', 'PDX1', 'MAFA',
		      'k_AKT', 'k_AKTp', 'k_AKT_JNK', 'k_GSK3', 'k_GSK3p', 'k_JNK_ROS', 'k_JNK_upr', 'k_p38_ROS',
		      'k_p38_upr', 'k_MAPKp', 'k_FOXO1_AKTp', 'k_FOXO1p_MAPKp', 'V_mPDX1', 'K_MAFA_PDX1', 'K_FOXO1_PDX1',
		      'k_PDX1_FOXO1_efflux', 'k_GSK3_PDX1_pdeg', 'V_mMAFA', 'K_PDX1_MAFA',
                      'K_FOXO1_MAFA', 'k_ROS_MAFA', 'k_GSK3_MAFA_pdeg', 'V_mINS', 'K_PDX1_INS', 'K_MAFA_INS']

        df = pd.DataFrame(data={"parameters": parameters, "Values": pos, "Cost": cost})
        df.to_csv("results/estimated_parameters_S2_{0}.csv".format(i+1), sep=',', index=False)

        print(time.localtime())
        print("end", time.time() - start_time)
