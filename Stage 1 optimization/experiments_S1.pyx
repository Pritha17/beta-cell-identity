# cython: language_level=3
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
cimport numpy as np
cimport cython
cimport openmp
from cython.operator cimport dereference as deref
from libc.math cimport pow, sqrt, exp
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libc.stdio cimport printf
from cython.parallel import prange, threadid
import time
from LSODA cimport *

@cython.boundscheck(False)
cdef initial_vals():
    '''
    :return: The initial values of the variables
    '''
    cdef vector[double] y0

    cdef double Gluc_ec = 3000          # microM
    cdef double Gluc_ic = 3000          # microM
    cdef double Pyr = 9                 # microM
    cdef double NADH_M = 60             # microM
    cdef double NAD_M = 250 - NADH_M    # microM
    cdef double ATP_C = 1400            # microM
    cdef double ADP_C = 2500 - ATP_C    # microM
    cdef double ATP_M = 11250           # microM
    cdef double ADP_M = 15000 - ATP_M   # microM
    cdef double PsiM = 130              # mV
    cdef double CaC = 0.1               # microM
    cdef double CaM = 0.1               # microM
    cdef double CaER = 200              # microM
    cdef double CaEc = 2000             # microM
    cdef double Ri = 0.088
    cdef double Vp = -64                # mV
    cdef double Pc = 1
    cdef double Po = 0
    cdef double J = 0.86
    cdef double n = 0

    y0.push_back(Gluc_ec)   #0
    y0.push_back(Gluc_ic)   #1
    y0.push_back(Pyr)       #2
    y0.push_back(NADH_M)    #3
    y0.push_back(NAD_M)     #4
    y0.push_back(ATP_C)     #5
    y0.push_back(ADP_C)     #6
    y0.push_back(ATP_M)     #7
    y0.push_back(ADP_M)     #8
    y0.push_back(PsiM)      #9
    y0.push_back(CaC)       #10
    y0.push_back(CaM)       #11
    y0.push_back(CaER)      #12
    y0.push_back(CaEc)      #13
    y0.push_back(Ri)        #14
    y0.push_back(Vp)        #15
    y0.push_back(Pc)        #16
    y0.push_back(Po)        #17
    y0.push_back(J)         #18
    y0.push_back(n)         #19

    return y0

@cython.boundscheck(False)
cdef unordered_map[int, double] parameter_vals() nogil:
    '''
    This function contains the known parameter values from the literature.
    :return: an unordered map of the parameter values
    '''

    # Volumes
    cdef double V_C = 0.882
    cdef double V_N = 0.117
    cdef double V_ER = 0.195
    cdef double V_M = 0.039

    # GLUT1
    cdef double K_mG1 = 3000  # microM

    # Glucokinase
    cdef double K_mATP = 500                # microM
    cdef double K_GK = 7000                 # microM
    cdef double n_GK = 1.7

    # J_PDH
    cdef double q_1 = 1
    cdef double q_2 = 0.1  # microM

    # J_O
    cdef double q_3 = 100   # microM
    cdef double q_4 = 143   # mV
    cdef double q_5 = 5     # mV

    # J_FIFO
    cdef double V_FIFO = 35000  # microM/s
    cdef double q_6 = 10000     # microM
    cdef double q_7 = 150       # mV
    cdef double q_8 = 8.5       # mV

    # J_ANT
    cdef double V_ANT = 5000    # microM/s
    cdef double alpha_C = 0.111
    cdef double alpha_M = 0.139
    cdef double F_prime = 0.037  # mV^-1

    # J_IP3
    cdef double V_IP3 = 0.198693    # s^-1
    cdef double K_IP3 = 1           #microM
    cdef double K_a = 0.3           # microM
    cdef double n_a = 3
    cdef double ip3 = 1

    # J_SERCA
    cdef double V_SERCA = 120   # microM/s
    cdef double K_p = 0.35      # microM
    cdef double K_e = 0.05      # microM

    # J_MCU
    cdef double V_MCU = 0.0006  # microM/s
    cdef double K_MCU = 0.968245
    cdef double p_1 = 0.1       # mv^-1

    # J_NCX
    cdef double V_NCX = 0.35    # microM/s
    cdef double p_2 = 0.016     # mV^-1

    # PsiM
    cdef double C_p = 1.8       # microM/mV

    # Ca_C
    cdef double f_C = 0.01

    # Ca_M
    cdef double f_M = 0.0003

    # Ca_ER
    cdef double f_ER = 0.01

    # R_i
    cdef double k_pos = 20       # microM^-4.s^-1
    cdef double k_neg = 0.02    # s^-1
    cdef double n_i = 4

    cdef double Vms = 2
    cdef double Sms = 14
    cdef double Tj = 30
    cdef double Vj = -40
    cdef double Sj = 7
    cdef double Vmf = -8
    cdef double Smf = 10
    cdef double Vn = -13
    cdef double Sn = 5.6
    cdef double Tn = 0.02
    cdef double V_tau = -75
    cdef double Tm = 0.0012820513
    cdef double RT_2F = 13.35
    cdef double gca = 1.47
    cdef double xf = 0.45
    cdef double gk = 2500
    cdef double Vk = -75
    cdef double gamma = 0.0045
    cdef double C_pm = 5310
    cdef double kminus = 2
    cdef double kstar = -0.000047
    cdef double gkatpbar = 6000
    cdef double K1 = 0.45
    cdef double K2 = 0.012


    cdef unordered_map[int, double] params

    params[0] = V_C
    params[1] =  V_N
    params[2] = V_ER
    params[3] = V_M
    params[4] = K_mG1
    params[5] = K_GK
    params[6] = n_GK
    params[7] = q_2
    params[8] = q_4
    params[9] = q_5
    params[10] = q_7
    params[11] = q_8
    params[12] = alpha_C
    params[13] = alpha_M
    params[14] = F_prime
    params[15] = n_a
    params[16] = ip3
    params[17] = p_1
    params[18] = p_2
    params[19] = f_C
    params[20] = f_M
    params[21] = f_ER
    params[22] = k_pos
    params[23] = k_neg
    params[24] = n_i
    params[25] = C_p

    params[26] = Vms
    params[27] = Sms
    params[28] = Tj
    params[29] = Vj
    params[30] = Sj
    params[31] = Vmf
    params[32] = Smf
    params[33] = Vn
    params[34] = Sn
    params[35] = Tn
    params[36] = V_tau
    params[37] = Tm
    params[38] = RT_2F
    params[39] = gca
    params[40] = xf
    params[41] = gk
    params[42] = Vk
    params[43] = gamma
    params[44] = C_pm
    params[45] = kminus
    params[46] = kstar
    params[47] = gkatpbar
    params[48] = K1
    params[49] = K2
    params[50] = K_mATP
    params[51] = q_1
    params[52] = q_3
    params[53] = V_FIFO
    params[54] = q_6
    params[55] = V_ANT
    params[56] = V_IP3
    params[57] = K_IP3
    params[58] = K_a
    params[59] = V_SERCA
    params[60] = K_p
    params[61] = K_e
    params[62] = V_MCU
    params[63] = K_MCU
    params[64] = V_NCX

    return params

@cython.boundscheck(False)
@cython.cdivision(True)
cdef unordered_map[char*, double] flux_data(double *X_data, vector[double] params) nogil:
    '''
    Defining the fluxes for glucose metabolism, calcium signalling and electrical activity
    :param X_data: the array of variable values
    :param params: the free parameters
    :return: an unordered map of the flux values
    '''

    # calling function containing the known parameter values from literature
    cdef unordered_map[int, double] pm = parameter_vals()

    # Influx of glucose through GLUT1
    cdef double J_G1I = (params[0] * X_data[0]) / (pm[4] + X_data[0])

    # Efflux of glucose through GLUT1
    cdef double J_G1E = (params[0] * X_data[1]) / (pm[4] + X_data[1])

    # rate of Glucokinase reaction
    cdef double J_GK = params[1] * ((pow(X_data[1], pm[6])) / (pow(pm[5], pm[6]) + pow(X_data[1], pm[6]))) * (X_data[5] / (pm[50] + X_data[5]))

    # Rate of pyruvate dehydrogenase (PDH) reaction
    cdef double J_PDH = params[2] * (X_data[2] / (params[6] + X_data[2])) * (1 / (pm[51] + X_data[3] / X_data[4])) * (X_data[11] / (pm[7] + X_data[11]))

    # Rate of NADH oxidation by ETC
    cdef double J_O = params[7] * (X_data[3] / (pm[52] + X_data[3])) * (1 / (1 + exp((X_data[9] - 143) / pm[9])))

    # Rate of proton flux through mitochondrial F1F0ATPase
    cdef double J_FIFO = pm[53] * (pm[54] / (pm[54] + X_data[7])) / (1 + exp((pm[10] - X_data[9]) / pm[11]))

    # Rate of ATP/ADP translocation across mitochondrial membrane by adenine nucleotide translocator (ANT)
    cdef double J_ANT = pm[55] * (1 - (pm[12] / pm[13]) * (X_data[5] / X_data[6]) * (X_data[8] / X_data[7]) * exp(-pm[14] * X_data[9])) / \
            ((1 + pm[12] * (X_data[5] / X_data[6]) * exp(-0.5 * pm[14] * X_data[9])) * (1 + (1 / pm[13]) * (X_data[8] / X_data[7])))

    # Ca2+ efflux from ER to cytosol through inositol trisphosphate receptor (IP3R)
    cdef double IR_a = (1 - X_data[14]) * 0.5 * 0.5

    cdef double J_IP3R = pm[56] * IR_a * (X_data[12] - X_data[10])

    # Ca2+ influx into ER from cytosol through sarco/endoplasmic reticulum Ca2+-ATPase (SERCA)
    cdef double J_SERCA = pm[59] * (pow(X_data[10], 2) / (pow(pm[60], 2) + pow(X_data[10], 2))) * (X_data[5] / (pm[61] + X_data[5]))

    # Ca2+ influx into mitochondria from cytosol through mitochondrial Ca2+ uniporter (MCU)
    cdef double J_MCU = pm[62] * (pow(X_data[10], 2) / (pow(pm[63], 2) + pow(X_data[10], 2))) * exp(pm[17] * X_data[9])

    # Ca2+ efflux from mitochondria to cytosol through mitochondrial Na+/Ca2+ exchanger (NCX)
    cdef double J_NCX = pm[64] * (X_data[11] / X_data[10]) * exp(pm[18] * X_data[9])

    # rate of ATP consumption in the cytosol
    cdef double J_HYD = (J_SERCA / 2) + params[3] * X_data[5]

    # Activation and inactivation functions.
    cdef double m_inf_s = 1 / (1 + exp((pm[26] - X_data[15]) / pm[27]))

    cdef double tau_j = pm[28] / (exp((X_data[15] - pm[29]) / (2 * pm[30])) + exp((pm[29] - X_data[15]) / (2 *  pm[30])))

    cdef double j_inf = 1 / (1 + exp((X_data[15] - pm[29]) / pm[30]))

    cdef double m_inf_f = 1 / (1 + exp((pm[31] - X_data[15]) / pm[32]))

    cdef double n_inf = 1 / (1 + exp((pm[33] - X_data[15]) / pm[34]))

    cdef double tau_n = pm[35] / (exp((X_data[15] - pm[36]) / 65) + exp((pm[36] - X_data[15]) / 20))

    cdef double alpha = m_inf_f / pm[37]

    cdef double beta = (1 - m_inf_f) / pm[37]

    # Currents

    cdef double ghk = X_data[13] * X_data[15] / (1 - exp(X_data[15] / pm[38]))

    cdef double Ica_f = pm[39] * X_data[17] * pm[40] * ghk

    cdef double Ica_s = pm[39] * m_inf_s * X_data[18] * (1 - pm[40]) * ghk

    cdef double I_ca = Ica_f + Ica_s

    cdef double I_k = pm[41] * X_data[19] * (X_data[15] - pm[42])

    cdef double g_katp = pm[47] * (1 + X_data[6] / pm[48]) / (1 + X_data[6] / pm[48] + X_data[5] / pm[49])

    cdef double I_katp = g_katp * (X_data[15] - pm[42])

    # Ca2+ influx through cell membrane
    cdef double J_pm = -pm[43] * I_ca - params[4] * X_data[10]     # the 1st term contains - because I_ca is negative

    # dictionary of the flux values defined above
    cdef unordered_map[char*, double] flux_dict

    flux_dict["J_G1I"] = J_G1I
    flux_dict["J_G1E"] = J_G1E
    flux_dict["J_GK"] = J_GK
    flux_dict["J_PDH"] = J_PDH
    flux_dict["J_O"] = J_O
    flux_dict["J_FIFO"] = J_FIFO
    flux_dict["J_ANT"] = J_ANT
    flux_dict["J_IP3R"] = J_IP3R
    flux_dict["J_SERCA"] = J_SERCA
    flux_dict["J_MCU"] = J_MCU
    flux_dict["J_NCX"] = J_NCX
    flux_dict["J_HYD"] = J_HYD
    flux_dict["m_inf_s"] = m_inf_s
    flux_dict["tau_j"] = tau_j
    flux_dict["j_inf"] = j_inf
    flux_dict["m_inf_f"] = m_inf_f
    flux_dict["n_inf"] = n_inf
    flux_dict["tau_n"] = tau_n
    flux_dict["alpha"] = alpha
    flux_dict["beta"] = beta
    flux_dict["ghk"] = ghk
    flux_dict["I_ca"] = I_ca
    flux_dict["I_k"] = I_k
    flux_dict["I_katp"] = I_katp
    flux_dict["J_pm"] = J_pm

    return flux_dict

@cython.boundscheck(False)
@cython.cdivision(True)
cdef void model(double t, double *X_data, double *dydt, vector[double] params) nogil:
    '''
    The ODE equations for the model
    :param t: time step
    :param X_data: the array of variable values
    :param dydt: the array where the derivatives of the variable will be stored
    :param params: the free parameters
    :return: an array of the derivatives of the variables
    '''

    # calling function containing the known parameter values from literature
    cdef unordered_map[int, double] pm = parameter_vals()

    # the fluxes for glucose metabolism, calcium signalling and electrical activity
    cdef unordered_map[char*, double] flux = flux_data(X_data, params)

    ####################################### ODEs #######################################################################

    # Rate of chane of extra-cellular glucose
    dydt[0] = 0

    # Rate of chane of intra-cellular glucose
    dydt[1] = (flux["J_G1I"] - flux["J_G1E"] - flux["J_GK"])

    # Rate of chane of pyruvate
    dydt[2] = flux["J_GK"] - flux["J_PDH"]

    # Rate of change of NADH/ NAD
    dydt[3] = flux["J_PDH"] - flux["J_O"]
    dydt[4] = -dydt[3]

    # Rate of change of cytosolic ATP/ ADP
    dydt[5] = flux["J_GK"] + pm[3] / pm[0] * flux["J_ANT"] - flux["J_HYD"]
    dydt[6] = -dydt[5]

    # Rate of change of mitochondrial ATP/ ADP
    dydt[7] = flux["J_FIFO"] - flux["J_ANT"]
    dydt[8] = -dydt[7]

    # Mitochondrial membrane potential
    dydt[9] = (params[5] * flux["J_O"] - flux["J_FIFO"] - flux["J_ANT"] - flux["J_NCX"] - 2 * flux["J_MCU"]) / pm[25]

    # Cytosolic Ca2+
    dydt[10] = pm[19] * (flux["J_pm"] + pm[2] / pm[0] * flux["J_IP3R"] - pm[2] / pm[0] * flux["J_SERCA"] - pm[3] / pm[0] * flux["J_MCU"] + pm[3] / pm[0] * flux["J_NCX"])

    # Mitochondrial Ca2+
    dydt[11] = pm[20] * (flux["J_MCU"] - flux["J_NCX"])

    # ER Ca2+
    dydt[12] = pm[21] * (flux["J_SERCA"] - flux["J_IP3R"])

    # Extracellular Ca2+
    dydt[13] = -pm[0] * flux["J_pm"]

    # fraction of inactive IP3 receptors
    dydt[14] = pm[22] * (pow(X_data[10], pm[24])) * ((1 - X_data[14]) / (1 + pow((X_data[10] / pm[58]), pm[15]))) - pm[23] * X_data[14]

    # Rate of change of cell membrane potential
    dydt[15] = -(flux['I_ca'] + flux['I_k'] + flux['I_katp']) / pm[44]

    # Rate of change of the fraction of channels closed
    dydt[16] = -flux['alpha'] * X_data[16] + flux['beta'] * X_data[17]

    # Rate of change of the fraction of channels open
    dydt[17] = pm[45] * (1 - X_data[17] - X_data[16]) - pm[46] * flux['ghk'] * X_data[17] + flux['alpha'] * X_data[16] - flux['beta'] * X_data[17]

    # Rate of change of the fraction of slow channels not inactivated.
    dydt[18] = (flux['j_inf'] - X_data[18]) / flux['tau_j']

    # Rate of actiavtion of the K+ channels
    dydt[19] = (flux['n_inf'] - X_data[19]) / flux['tau_n']

cdef double experiment_steady_state(vector[double] X_data_0, vector[double] params) nogil:
    '''
    :param X_data_0: the initial values of the variables
    :param params: the free parameters
    :return: the simulated value of the variable
    '''

    cdef:
        LSODA lsoda
        int istate = 1
        vector[vector[double]] X_data_simulated
        vector[double] yout
        vector[double] X_data_ini = X_data_0
        size_t neq = 20     # number of equations (ODEs)
        double t = 0
        double tout
        size_t iout, j, i, k, I, s
        int num_steps = 360
        vector[double] deviation
        double sum, rmse, avg_rmse


    X_data_simulated.push_back(X_data_ini)
    for iout in range(1, num_steps):
        tout += 10
        lsoda.lsoda_update(model, neq, X_data_ini, yout, &t, tout, &istate, params, 1e-8, 1e-8)
        #Update the y for next iteration
        for j in range(neq):
            X_data_ini[j] = yout[j+1]

        X_data_simulated.push_back(X_data_ini)

        if (istate <= 0):
            exit

    if (istate < 2):
        avg_rmse = 1000
    else:
        for j in range(2, 16):
            sum = 0.0
            for i in range(num_steps):
                sum += pow((X_data_simulated[i][j] / X_data_0[j] - 1), 2) # compare with steady-state values from the literature
            rmse = sqrt(sum / num_steps)
            deviation.push_back(rmse)

        sum = 0.0
        for s in range(deviation.size()):
            sum += deviation[s]

        avg_rmse = sum / deviation.size()

    return avg_rmse

cdef vector[double] steady_state_initials(vector[double] X_data_0, vector[double] params) nogil:
    '''
    :param X_data_0: initial values of the variables
    :param params: the free parameters
    :return: the simulated values of the variables measured in the experiment
    '''

    cdef:
        LSODA lsoda
        int istate = 1
        vector[double] yout
        vector[double] X_data_ini = X_data_0
        size_t neq = 20     # number of equations (ODEs)
        double t = 0
        double tout
        size_t iout, j, mv, pv
        int num_steps = 180

    for iout in range(1, num_steps):
        tout += 10
        lsoda.lsoda_update(model, neq, X_data_ini, yout, &t, tout, &istate, params, 1e-8, 1e-8)
        #Update the y for next iteration
        for j in range(neq):
            X_data_ini[j] = yout[j+1]

        if (istate <= 0):
            exit

    return X_data_ini

cdef double experiment_NADH(vector[double] X_data_0, vector[double] params, vector[double] t_measured,
vector[double] NADH_ratio_measured_1, vector[double] NADH_ratio_measured_2, vector[double] NADH_ratio_measured_3,
double mean_NADH_8, double mean_NADH_10, double mean_NADH_20) nogil:
    '''
    :param X_data_0: initial values of the variables
    :param params: the free parameters
    :param t_measured: time points in the experimental data
    :param NADH_ratio_measured_1, NADH_ratio_measured_2, NADH_ratio_measured_3: experimental data
    :param mean_NADH_8, mean_NADH_10, mean_NADH_20: mean of the experimental data
    :return: root mean squared error
    '''

    cdef:
        LSODA lsoda
        int istate_1 = 1, istate_2 = 1, istate_3 = 1
        vector[double] res_1, res_2, res_3
        vector[double] yout_1, yout_2, yout_3
        vector[double] X_data_ini = X_data_0
        size_t neq = 20     # number of equations (ODEs)
        double t = 0
        double tout
        size_t iout, j, k, I, s
        vector[double] NADH_ratio_simulated_1, NADH_ratio_simulated_2, NADH_ratio_simulated_3
        vector[double] deviation_1, deviation_2, deviation_3
        double sum_1 = 0.0, sum_2 = 0.0, sum_3 = 0.0

    I = t_measured.size()

    X_data_ini[0] = 8000
    res_1.push_back(X_data_0[3])
    for iout in range(1, I):
        tout = t_measured[iout]
        lsoda.lsoda_update(model, neq, X_data_ini, yout_1, &t, tout, &istate_1, params, 1e-8, 1e-8)
        #Update the y for next iteration
        for j in range(neq):
            X_data_ini[j] = yout_1[j+1]

        res_1.push_back(X_data_ini[3])

        if (istate_1 <= 0):
            exit

    t = 0
    X_data_ini = X_data_0
    X_data_ini[0] = 10000
    res_2.push_back(X_data_0[3])
    for iout in range(1, I):
        tout = t_measured[iout]
        lsoda.lsoda_update(model, neq, X_data_ini, yout_2, &t, tout, &istate_2, params, 1e-8, 1e-8)
        #Update the y for next iteration
        for j in range(neq):
            X_data_ini[j] = yout_2[j+1]

        res_2.push_back(X_data_ini[3])

        if (istate_2 <= 0):
            exit

    t = 0
    X_data_ini = X_data_0
    X_data_ini[0] = 20000
    res_3.push_back(X_data_0[3])
    for iout in range(1, I):
        tout = t_measured[iout]
        lsoda.lsoda_update(model, neq, X_data_ini, yout_3, &t, tout, &istate_3, params, 1e-8, 1e-8)
        #Update the y for next iteration
        for j in range(neq):
            X_data_ini[j] = yout_3[j+1]

        res_3.push_back(X_data_ini[3])

        if (istate_3 <= 0):
            exit

    if (istate_1 < 2 or istate_2 < 2 or istate_3 < 2):
        for i in range(I):
            NADH_ratio_simulated_1.push_back(1000)
            NADH_ratio_simulated_2.push_back(1000)
            NADH_ratio_simulated_3.push_back(1000)
    else:
        for i in range(I):
            NADH_ratio_simulated_1.push_back(res_1[i] / X_data_0[3])
            NADH_ratio_simulated_2.push_back(res_2[i] / X_data_0[3])
            NADH_ratio_simulated_3.push_back(res_3[i] / X_data_0[3])

    for k in range(I):
        deviation_1.push_back(pow((NADH_ratio_simulated_1[k] - NADH_ratio_measured_1[k]), 2))
        deviation_2.push_back(pow((NADH_ratio_simulated_2[k] - NADH_ratio_measured_2[k]), 2))
        deviation_3.push_back(pow((NADH_ratio_simulated_3[k] - NADH_ratio_measured_3[k]), 2))

    for s in range(I):
        sum_1 += deviation_1[s]
        sum_2 += deviation_2[s]
        sum_3 += deviation_3[s]

    cdef double mse_deviation = sqrt(sum_1 / I) / mean_NADH_8 + sqrt(sum_2 / I) / mean_NADH_10 + sqrt(sum_3 / I) / mean_NADH_20

    return mse_deviation

cdef double experiment_ATP(vector[double] X_data_0, vector[double] params, vector[double] t_measured,
vector[double] ATP_ratio_measured_1, vector[double] ATP_ratio_measured_2, vector[double] ATP_ratio_measured_3,
double mean_ATP_8, double mean_ATP_16, double mean_ATP_25) nogil:
    '''
    :param X_data_0: initial values of the variables
    :param params: the free parameters
    :param t_measured: time points in the experimental data
    :param ATP_ratio_measured_1, ATP_ratio_measured_2, ATP_ratio_measured_3: experimental data
    :param mean_ATP_8, mean_ATP_16, mean_ATP_25: mean of the experimental data
    :return: root mean squared error
    '''

    cdef:
        LSODA lsoda
        int istate_1 = 1, istate_2 = 1, istate_3 = 1
        vector[double] res_1, res_2, res_3
        vector[double] yout_1, yout_2, yout_3
        vector[double] X_data_ini = X_data_0
        size_t neq = 20     # number of equations (ODEs)
        double t = 0
        double tout
        size_t iout, j, k, I, s
        vector[double] ATP_ratio_simulated_1, ATP_ratio_simulated_2, ATP_ratio_simulated_3
        vector[double] deviation_1, deviation_2, deviation_3
        double sum_1 = 0.0, sum_2 = 0.0, sum_3 = 0.0

    I = t_measured.size()

    X_data_ini[0] = 8300
    res_1.push_back(X_data_0[5])
    for iout in range(1, I):
        tout = t_measured[iout]
        lsoda.lsoda_update(model, neq, X_data_ini, yout_1, &t, tout, &istate_1, params, 1e-8, 1e-8)
        #Update the y for next iteration
        for j in range(neq):
            X_data_ini[j] = yout_1[j+1]

        res_1.push_back(X_data_ini[5])

        if (istate_1 <= 0):
            exit

    t = 0
    X_data_ini = X_data_0
    X_data_ini[0] = 16700
    res_2.push_back(X_data_0[5])
    for iout in range(1, I):
        tout = t_measured[iout]
        lsoda.lsoda_update(model, neq, X_data_ini, yout_2, &t, tout, &istate_2, params, 1e-8, 1e-8)
        #Update the y for next iteration
        for j in range(neq):
            X_data_ini[j] = yout_2[j+1]

        res_2.push_back(X_data_ini[5])

        if (istate_2 <= 0):
            exit

    t = 0
    X_data_ini = X_data_0
    X_data_ini[0] = 25000
    res_3.push_back(X_data_0[5])
    for iout in range(1, I):
        tout = t_measured[iout]
        lsoda.lsoda_update(model, neq, X_data_ini, yout_3, &t, tout, &istate_3, params, 1e-8, 1e-8)
        #Update the y for next iteration
        for j in range(neq):
            X_data_ini[j] = yout_3[j+1]

        res_3.push_back(X_data_ini[5])

        if (istate_3 <= 0):
            exit

    if (istate_1 < 2 or istate_2 < 2 or istate_3 < 2):
        for i in range(I):
            ATP_ratio_simulated_1.push_back(1000)
            ATP_ratio_simulated_2.push_back(1000)
            ATP_ratio_simulated_3.push_back(1000)
    else:
        for i in range(I):
            ATP_ratio_simulated_1.push_back(res_1[i] / X_data_0[5])
            ATP_ratio_simulated_2.push_back(res_2[i] / X_data_0[5])
            ATP_ratio_simulated_3.push_back(res_3[i] / X_data_0[5])

    for k in range(I):
        deviation_1.push_back(pow((ATP_ratio_simulated_1[k] - ATP_ratio_measured_1[k]), 2))
        deviation_2.push_back(pow((ATP_ratio_simulated_2[k] - ATP_ratio_measured_2[k]), 2))
        deviation_3.push_back(pow((ATP_ratio_simulated_3[k] - ATP_ratio_measured_3[k]), 2))

    for s in range(I):
        sum_1 += deviation_1[s]
        sum_2 += deviation_2[s]
        sum_3 += deviation_3[s]

    cdef double mse_deviation = sqrt(sum_1 / I) / mean_ATP_8 + sqrt(sum_2 / I) / mean_ATP_16 + sqrt(sum_3 / I) / mean_ATP_25

    return mse_deviation

def fitness(params_np):
    '''
    Fitness function given a set of parameters
    :param params: the current set of parameters to be evaluated
    :return: the fitness value
    '''

    cdef:
        double[:] t_measured_NADH_view, t_measured_Ca_view, t_measured_ATP_view
        double[:] NADH_measured_8_ratio_view, NADH_measured_10_ratio_view, NADH_measured_20_ratio_view
        double[:] ATP_C_8_measured_ratio_view, ATP_C_16_measured_ratio_view, ATP_C_25_measured_ratio_view

        double[:, :] params_view = params_np        # 2D memoryview. defining a memoryview on a numpy array

        vector[double] t_measured_NADH, t_measured_Ca, t_measured_ATP
        vector[double] NADH_measured_8_ratio, NADH_measured_10_ratio, NADH_measured_20_ratio
        vector[double] ATP_C_8_measured_ratio, ATP_C_16_measured_ratio, ATP_C_25_measured_ratio

        double mean_NADH_8, mean_NADH_10, mean_NADH_20, mean_ATP_8, mean_ATP_16, mean_ATP_25

        vector[vector[double]] params
        vector[double] X_data_lit, X_data_initial, temp
        int num_particles = params_view.shape[0]

        int ti, j, k, pi

        int NUM_THREADS = 16
        int CHUNKSIZE = int(num_particles/NUM_THREADS)

        vector[double] CP1_vect, CP2_vect, CP3_vect, CP4_vect


    # Reading data from files
    df1 = pd.read_csv("data/NADH_data.csv", decimal='.', delimiter=',')
    t_measured_NADH_np = np.array(df1['Time (sec)'])
    t_measured_NADH_np = t_measured_NADH_np.astype('double')    # typecasting to double
    NADH_measured_8_ratio_np = np.array(df1['NADH at 8 mM relative to 3mM glucose'])
    NADH_measured_8_ratio_np = NADH_measured_8_ratio_np.astype('double')
    NADH_measured_10_ratio_np = np.array(df1['NADH at 10 mM relative to 3mM glucose'])
    NADH_measured_10_ratio_np = NADH_measured_10_ratio_np.astype('double')
    NADH_measured_20_ratio_np = np.array(df1['NADH at 20 mM relative to 3mM glucose'])
    NADH_measured_20_ratio_np = NADH_measured_20_ratio_np.astype('double')
    mean_NADH_8 = np.mean(NADH_measured_8_ratio_np)
    mean_NADH_10 = np.mean(NADH_measured_10_ratio_np)
    mean_NADH_20 = np.mean(NADH_measured_20_ratio_np)

    df3 = pd.read_csv("data/ATP_data.csv", decimal='.', delimiter=',')
    t_measured_ATP_np = np.array(df3['Time(sec)'])
    t_measured_ATP_np = t_measured_ATP_np.astype('double')
    ATP_C_8_measured_ratio_np = np.array(df3['ATP_C_8_ratio'])
    ATP_C_8_measured_ratio_np = ATP_C_8_measured_ratio_np.astype('double')
    ATP_C_16_measured_ratio_np = np.array(df3['ATP_C_16_ratio'])
    ATP_C_16_measured_ratio_np = ATP_C_16_measured_ratio_np.astype('double')
    ATP_C_25_measured_ratio_np = np.array(df3['ATP_C_25_ratio'])
    ATP_C_25_measured_ratio_np = ATP_C_25_measured_ratio_np.astype('double')

    mean_ATP_8 = np.mean(ATP_C_8_measured_ratio_np)
    mean_ATP_16 = np.mean(ATP_C_16_measured_ratio_np)
    mean_ATP_25 = np.mean(ATP_C_25_measured_ratio_np)

    t_measured_NADH_view = t_measured_NADH_np         # 1D memoryview
    NADH_measured_8_ratio_view = NADH_measured_8_ratio_np
    NADH_measured_10_ratio_view = NADH_measured_10_ratio_np
    NADH_measured_20_ratio_view = NADH_measured_20_ratio_np

    t_measured_ATP_view = t_measured_ATP_np
    ATP_C_8_measured_ratio_view = ATP_C_8_measured_ratio_np
    ATP_C_16_measured_ratio_view = ATP_C_16_measured_ratio_np
    ATP_C_25_measured_ratio_view = ATP_C_25_measured_ratio_np

    # casting memoryview into vector

    for ti in range(t_measured_NADH_view.shape[0]):
        t_measured_NADH.push_back(t_measured_NADH_view[ti])
        NADH_measured_8_ratio.push_back(NADH_measured_8_ratio_view[ti])
        NADH_measured_10_ratio.push_back(NADH_measured_10_ratio_view[ti])
        NADH_measured_20_ratio.push_back(NADH_measured_20_ratio_view[ti])

    for ti in range(t_measured_ATP_view.shape[0]):
        t_measured_ATP.push_back(t_measured_ATP_view[ti])
        ATP_C_8_measured_ratio.push_back(ATP_C_8_measured_ratio_view[ti])
        ATP_C_16_measured_ratio.push_back(ATP_C_16_measured_ratio_view[ti])
        ATP_C_25_measured_ratio.push_back(ATP_C_25_measured_ratio_view[ti])

    for j in range(params_view.shape[0]):
        for k in range(params_view.shape[1]):
            temp.push_back(params_view[j][k])
        params.push_back(temp)
        temp.clear()

    # the initial values of the variables
    X_data_lit = initial_vals()

    # resizing vector; the first argument specifies the number of elements and the second the values of the elements
    CP1_vect.resize(num_particles, 1)
    CP2_vect.resize(num_particles, 1)
    CP3_vect.resize(num_particles, 1)

    # the parallel part without the GIL
    for pi in prange(num_particles, nogil=True, schedule='static', num_threads=NUM_THREADS, chunksize=CHUNKSIZE):

        CP1_vect[openmp.omp_get_thread_num() * 0 + pi] = experiment_steady_state(X_data_lit, params[pi])

        X_data_initial = steady_state_initials(X_data_lit, params[pi])

        CP2_vect[openmp.omp_get_thread_num() * 0 + pi] = experiment_NADH(X_data_initial, params[pi], t_measured_NADH,
            NADH_measured_8_ratio, NADH_measured_10_ratio, NADH_measured_20_ratio, mean_NADH_8, mean_NADH_10, mean_NADH_20)

        CP3_vect[openmp.omp_get_thread_num() * 0 + pi] = experiment_ATP(X_data_initial, params[pi], t_measured_ATP,
            ATP_C_8_measured_ratio, ATP_C_16_measured_ratio, ATP_C_25_measured_ratio, mean_ATP_8, mean_ATP_16, mean_ATP_25)

    CP1 = np.array(CP1_vect)
    CP2 = np.array(CP2_vect)
    CP3 = np.array(CP3_vect)

    CP = CP1 + CP2 + CP3

    return CP