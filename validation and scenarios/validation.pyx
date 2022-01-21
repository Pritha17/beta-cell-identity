# cython: language_level=3
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
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
from scipy.stats import t
from LSODA cimport *

@cython.boundscheck(False)
cdef vector[double] initial_vals(vector[double] params):
    '''
    :param params: estimated parameters from stage 2 optimization
    :return: 1D vector of the initial values of the variables
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

    cdef double AKT = params[0]     # microM
    cdef double AKTp = params[1]    # microM
    cdef double GSK3 = params[2]    # microM
    cdef double GSK3p = params[3]   # microM
    cdef double JNK = params[4]     # microM
    cdef double JNKp = params[5]    # microM
    cdef double p38 = params[6]     # microM
    cdef double p38p = params[7]    # microM
    cdef double FOXO1 = params[8]   # microM
    cdef double FOXO1p = params[9]  # microM
    cdef double PDX1 = params[10]   # microM
    cdef double MAFA = params[11]   # microM
    cdef double INSm = 1.3424       # microM

    cdef double INS_UF = 5.93064    # microM
    cdef double Chap = 0.593064     # microM
    cdef double INS_ic = 189.234    # microM
    cdef double Rs = 0.01           # microM
    cdef double Rh = 0.1            # microM
    cdef double Ins_ec = 0.00009    # microM

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

    y0.push_back(AKT)       #20
    y0.push_back(AKTp)      #21
    y0.push_back(GSK3)      #22
    y0.push_back(GSK3p)     #23
    y0.push_back(JNK)       #24
    y0.push_back(JNKp)      #25
    y0.push_back(p38)       #26
    y0.push_back(p38p)      #27
    y0.push_back(FOXO1)     #28
    y0.push_back(FOXO1p)    #29
    y0.push_back(PDX1)      #30
    y0.push_back(MAFA)      #31
    y0.push_back(INSm)      #32
    y0.push_back(INS_UF)    #33
    y0.push_back(Chap)      #34
    y0.push_back(INS_ic)    #35
    y0.push_back(Rs)        #36
    y0.push_back(Rh)        #37
    y0.push_back(Ins_ec)    #38

    return y0

@cython.boundscheck(False)
cdef unordered_map[int, double] parameter_vals(vector[double] metblsm_par):
    '''
    This function contains the known parameter values from the literature.
    :param metblsm_par: estimated parameters from stage 1 optimization
    :return: an unordered map of the paraeter values
    '''

    # Volumes
    cdef double V_C = 0.882
    cdef double V_N = 0.117
    cdef double V_ER = 0.195
    cdef double V_M = 0.039

    # GLUT1
    cdef double V_mG1 = metblsm_par[0]    #microM/s
    cdef double K_mG1 = 3000  # microM

    # Glucokinase
    cdef double V_mGK = metblsm_par[1]  # microM/s
    cdef double K_mATP = 500  # microM
    cdef double K_GK = 7000  # microM
    cdef double n_GK = 1.7

    # J_PDH
    cdef double V_mPDH = metblsm_par[2]   # microM/s
    cdef double K_mPyr = metblsm_par[6]  # microM
    cdef double q_1 = 1
    cdef double q_2 = 0.1  # microM

    # J_O
    cdef double V_O = metblsm_par[7]   # microM/s
    cdef double q_3 = 100   # microM
    cdef double q_4 = 143   # mV
    cdef double q_5 = 5     # mV

    # J_FIFO
    cdef double V_FIFO = 35000  # microM/s
    cdef double q_6 = 10000     # microM
    cdef double q_7 = 150       # mV
    cdef double q_8 = 8.5       # mV

    # J_ANT
    cdef double V_ANT = 5000  # microM/s
    cdef double alpha_C = 0.111
    cdef double alpha_M = 0.139
    cdef double F_prime = 0.037  # mV^-1

    # J_IP3
    cdef double V_IP3 = 0.198693    # s^-1
    cdef double K_IP3 = 1   #microM
    cdef double ip3 = 1

    # J_SERCA
    cdef double V_SERCA = 120   # microM/s
    cdef double K_p = 0.35  # microM
    cdef double K_e = 0.05  # microM

    # J_MCU
    cdef double V_MCU = 0.0006  # microM/s
    cdef double K_MCU = 0.9682455   # microM
    cdef double p_1 = 0.1       # mv^-1

    # J_NCX
    cdef double V_NCX = 0.35    # microM/s
    cdef double p_2 = 0.016     # mV^-1

    # J_HYD
    cdef double k_HYD = metblsm_par[3]     # microM/s

    # PsiM
    cdef double a_1 = metblsm_par[5]
    cdef double C_p = 1.8  # microM/mV

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
    cdef double k_PMCA = metblsm_par[4]

    # protein folding
    cdef double K_uu = 0.0593064    # microM
    cdef double d_CP = 0.291835     # s^-1
    cdef double Ku_1 = 5.93064      # microM
    cdef double Ku_2 = 0.0355838    # microM
    cdef double Ku_3 = 1.18613      # microM
    cdef double q_f = 0.8
    cdef double d_uu = 0.06727      # s^-1
    cdef double k_IS = 0.0002041    # s^-1
    cdef double k_IC = 351.115      # s^-1
    cdef double k_tr_INS = 0.035584 # s^-1
    cdef double k_dm_INS = 0.000006637  # s^-1

    # ROS
    cdef double A_s = 5     # microM
    cdef double A_h = 3     # microM
    cdef double q_r = 9.76612 # microM
    cdef double k_rs = 4000     # microM^-1s^-1
    cdef double d_rh = 500 # microM^-1s^-1

    #insulin secretion
    cdef double CaC_min = 0.055 # microM

    cdef unordered_map[int, double] pm

    pm[0] = V_C
    pm[1] =  V_N
    pm[2] = V_ER
    pm[3] = V_M
    pm[4] = V_mG1
    pm[5] = K_mG1
    pm[6] = V_mGK
    pm[7] = K_GK
    pm[8] = n_GK
    pm[9] = K_mATP
    pm[10] = V_mPDH
    pm[11] = K_mPyr
    pm[12] = q_1
    pm[13] = q_2
    pm[14] = V_O
    pm[15] = q_3
    pm[16] = q_4
    pm[17] = q_5
    pm[18] = V_FIFO
    pm[19] = q_6
    pm[20] = q_7
    pm[21] = q_8
    pm[22] = V_ANT
    pm[23] = alpha_C
    pm[24] = alpha_M
    pm[25] = F_prime
    pm[26] = V_IP3
    pm[27] = K_IP3
    pm[28] = ip3
    pm[29] = V_SERCA
    pm[30] = K_p
    pm[31] = K_e
    pm[32] = V_MCU
    pm[33] = K_MCU
    pm[34] = p_1
    pm[35] = V_NCX
    pm[36] = p_2
    pm[37] = k_HYD
    pm[38] = a_1
    pm[39] = C_p
    pm[40] = f_C
    pm[41] = f_M
    pm[42] = f_ER
    pm[43] = k_pos
    pm[44] = k_neg
    pm[45] = n_i
    pm[46] = Vms
    pm[47] = Sms
    pm[48] = Tj
    pm[49] = Vj
    pm[50] = Sj
    pm[51] = Vmf
    pm[52] = Smf
    pm[53] = Vn
    pm[54] = Sn
    pm[55] = Tn
    pm[56] = V_tau
    pm[57] = Tm
    pm[58] = RT_2F
    pm[59] = gca
    pm[60] = xf
    pm[61] = gk
    pm[62] = Vk
    pm[63] = gamma
    pm[64] = C_pm
    pm[65] = kminus
    pm[66] = kstar
    pm[67] = gkatpbar
    pm[68] = K1
    pm[69] = K2
    pm[70] = k_PMCA
    pm[71] = K_uu
    pm[72] = d_CP
    pm[73] = Ku_1
    pm[74] = Ku_2
    pm[75] = Ku_3
    pm[76] = q_f
    pm[77] = A_s
    pm[78] = A_h
    pm[79] = q_r
    pm[80] = k_rs
    pm[81] = d_rh
    pm[82] = CaC_min
    pm[83] = d_uu
    pm[84] = k_IS
    pm[85] = k_IC
    pm[86] = k_tr_INS
    pm[87] = k_dm_INS

    return pm

@cython.boundscheck(False)
@cython.cdivision(True)
cdef unordered_map[char*, double] flux_data(double *X_data, vector[double] params, vector[double] metblsm_par):
    '''
    Defining the fluxes for glucose metabolism, calcium signalling and electrical activity
    :param X_data: the array of variable values
    :param params: estimated parameters from stage 2 optimization
    :param metblsm_par: estimated parameters from stage 1 optimization
    :return: an unordered map of the flux values
    '''

    # calling function containing the known parameter values from literature
    cdef unordered_map[int, double] pm = parameter_vals(metblsm_par)

    # Influx of glucose through GLUT1
    cdef double J_G1I = (pm[4] * X_data[0]) / (pm[5] + X_data[0])

    # Efflux of glucose through GLUT1
    cdef double J_G1E = (pm[4] * X_data[1]) / (pm[5] + X_data[1])

    # rate of Glucokinase reaction
    cdef double J_GK = pm[6] * ((pow(X_data[1], pm[8])) / (pow(pm[7], pm[8]) + pow(X_data[1], pm[8]))) * (X_data[5] / (pm[9] + X_data[5]))

    # Rate of pyruvate dehydrogenase (PDH) reaction
    cdef double J_PDH = pm[10] * (X_data[2] / (pm[11] + X_data[2])) * (1 / (pm[12] + X_data[3] / X_data[4])) * (X_data[11] / (pm[13] + X_data[11]))

    # Rate of NADH oxidation by ETC
    cdef double J_O = pm[14] * (X_data[3] / (pm[15] + X_data[3])) * (1 / (1 + exp((X_data[9] - pm[16]) / pm[17])))

    # Rate of proton flux through mitochondrial F1F0ATPase
    cdef double J_FIFO = pm[18] * (pm[19] / (pm[19] + X_data[7])) / (1 + exp((pm[20] - X_data[9]) / pm[21]))

    # Rate of ATP/ADP translocation across mitochondrial membrane by adenine nucleotide translocator (ANT)
    cdef double J_ANT = pm[22] * (1 - (pm[23] / pm[24]) * (X_data[5] / X_data[6]) * (X_data[8] / X_data[7]) * exp(-pm[25] * X_data[9])) / \
            ((1 + pm[23] * (X_data[5] / X_data[6]) * exp(-0.5 * pm[25] * X_data[9])) * (1 + (1 / pm[24]) * (X_data[8] / X_data[7])))

    # Ca2+ efflux from ER to cytosol through inositol trisphosphate receptor (IP3R)
    cdef double IR_a = 0.5 * (1 - X_data[14]) * (pow(pm[28], 3) / (pow(pm[27], 3) + pow(pm[28], 3)))

    cdef double J_IP3R = pm[26] * IR_a * (X_data[12] - X_data[10])

    # Ca2+ influx into ER from cytosol through sarco/endoplasmic reticulum Ca2+-ATPase (SERCA)
    cdef double J_SERCA = pm[29] * (pow(X_data[10], 2) / (pow(pm[30], 2) + pow(X_data[10], 2))) * (X_data[5] / (pm[31] + X_data[5]))

    # Ca2+ influx into mitochondria from cytosol through mitochondrial Ca2+ uniporter (MCU)
    cdef double J_MCU = pm[32] * (pow(X_data[10], 2) / (pow(pm[33], 2) + pow(X_data[10], 2))) * exp(pm[34] * X_data[9])

    # Ca2+ efflux from mitochondria to cytosol through mitochondrial Na+/Ca2+ exchanger (NCX)
    cdef double J_NCX = pm[35] * (X_data[11] / X_data[10]) * exp(pm[36] * X_data[9])

    # rate of ATP consumption in the cytosol
    cdef double J_HYD = (J_SERCA / 2) + pm[37] * X_data[5]

    # Activation and inactivation functions.
    cdef double m_inf_s = 1 / (1 + exp((pm[46] - X_data[15]) / pm[47]))

    cdef double tau_j = pm[48] / (exp((X_data[15] - pm[49]) / (2 * pm[50])) + exp((pm[49] - X_data[15]) / (2 *  pm[50])))

    cdef double j_inf = 1 / (1 + exp((X_data[15] - pm[49]) / pm[50]))

    cdef double m_inf_f = 1 / (1 + exp((pm[51] - X_data[15]) / pm[52]))

    cdef double n_inf = 1 / (1 + exp((pm[53] - X_data[15]) / pm[54]))

    cdef double tau_n = pm[55] / (exp((X_data[15] - pm[56]) / 65) + exp((pm[56] - X_data[15]) / 20))

    cdef double alpha = m_inf_f / pm[57]

    cdef double beta = (1 - m_inf_f) / pm[57]

    # Currents

    cdef double ghk = X_data[13] * X_data[15] / (1 - exp(X_data[15] / pm[58]))

    cdef double Ica_f = pm[59] * X_data[17] * pm[60] * ghk

    cdef double Ica_s = pm[59] * m_inf_s * X_data[18] * (1 - pm[60]) * ghk

    cdef double I_ca = Ica_f + Ica_s

    cdef double I_k = pm[61] * X_data[19] * (X_data[15] - pm[62])

    cdef double g_katp = pm[67] * (1 + X_data[6] / pm[68]) / (1 + X_data[6] / pm[68] + X_data[5] / pm[69])

    cdef double I_katp = g_katp * (X_data[15] - pm[62])

    # Ca2+ influx through cell membrane
    cdef double J_pm = -pm[63] * I_ca - pm[70] * X_data[10]     # the 1st term contains - because I_ca is negative

    # Insulin signalling
    cdef double v_akt_ins = params[12] * X_data[20] * X_data[38] / 0.00009
    cdef double v_aktp = params[13] * X_data[21]
    cdef double v_akt_jnk = params[14] * X_data[21] * X_data[25]

    cdef double v_gsk3_akt = params[15] * X_data[22] * X_data[21]
    cdef double v_gsk3p = params[16] * X_data[23]

    # unfolded protin response
    cdef double v_upr = (X_data[33] / pm[73]) / (1 + X_data[33] / pm[73] + X_data[34] / (pm[74] * (1 + X_data[33] / pm[75])))

    # stress-activated kinases
    cdef double v_jnk_ros = params[17] * X_data[24] * X_data[37] / 0.1
    cdef double v_jnk_upr = params[18] * X_data[24] * v_upr
    cdef double v_jnkp = params[21] * X_data[25]

    #cdef double v_p38_ros = params[19] * X_data[26] * X_data[37]
    cdef double v_p38_ros = params[19] * X_data[26] * X_data[37] / 0.1
    cdef double v_p38_upr = params[20] * X_data[26] * v_upr
    cdef double v_p38p = params[21] * X_data[27]

    # FOXO1
    cdef double v_foxo1c_akt = params[22] * X_data[28] * X_data[21]
    cdef double v_foxo1c_mapk = params[23] * X_data[29] * (X_data[25] + X_data[27])

    # PDX1
    cdef double v_pdx1_prod = params[24] * (X_data[31] / (params[25] + X_data[31])) * (params[26] / (params[26] + X_data[28]))
    cdef double v_pdx1_efflux_foxo1 = params[27] * X_data[30] * X_data[28]
    cdef double v_pdx1_pdeg_gsk3 = params[28] * X_data[30] * X_data[22]

    # MAFA
    cdef double v_mafa_prod = params[29] * (X_data[30] / (params[30] + X_data[30])) * (X_data[28] / (params[31] + X_data[28]))
    cdef double v_mafa_efflux_ros = params[32] * X_data[31] * X_data[37] / 0.1
    cdef double v_mafa_pdeg_gsk3 = params[33] * X_data[31] * X_data[22]

    # Insulin mRNA
    cdef double v_ins_prod = params[34] * (X_data[30] / (params[35] + X_data[30])) * (X_data[31] / (params[36] + X_data[31]))
    cdef double v_ins_deg = pm[87] * X_data[32]

    # insulin folding
    cdef double v_proinsulin_prod = pm[86] * X_data[32] * (1 / (1 + v_upr))
    cdef double v_folding_deg_chap = pm[83] * X_data[34] * X_data[33] / (X_data[33] + pm[71])
    cdef double v_chap_act = v_upr / (1 + v_upr)
    cdef double v_chap_deg = pm[72] * X_data[34]
    cdef double v_insulin_folding = pm[76] * v_folding_deg_chap

    # insulin secretion
    cdef double v_insulin_secretion = pm[84] * X_data[35] * max(0, (X_data[10] - pm[82])) / pm[82]
    cdef double v_insulin_clearance = pm[85] * X_data[38]

    # ROS
    cdef double v_superoxide_prod = pm[79] * J_O
    cdef double v_superoxide_to_h2o2 = pm[80] * X_data[36] * pm[77]
    cdef double v_h2o2_clearance = pm[81] * X_data[37] * pm[78]
    cdef double v_h2o2_prod_er = 3 * v_folding_deg_chap

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

    flux_dict["k_akt_ins"] = v_akt_ins
    flux_dict["k_aktp"] = v_aktp
    flux_dict["k_akt_jnk"] = v_akt_jnk

    flux_dict["k_g_akt"] = v_gsk3_akt
    flux_dict["k_gsk3p"] = v_gsk3p

    flux_dict["k_upr"] = v_upr
    flux_dict["k_jnk_ros"] = v_jnk_ros
    flux_dict["k_jnk_upr"] = v_jnk_upr
    flux_dict["k_jnkp"] = v_jnkp

    flux_dict["k_p38_ros"] = v_p38_ros
    flux_dict["k_p38_upr"] = v_p38_upr
    flux_dict["k_p38p"] = v_p38p

    flux_dict["k_f1c_akt"] = v_foxo1c_akt
    flux_dict["k_foxo1c_mapk"] = v_foxo1c_mapk

    flux_dict["k_p_prod"] = v_pdx1_prod
    flux_dict["k_pdx1_efflux_foxo1"] = v_pdx1_efflux_foxo1
    flux_dict["k_pdx1_pdeg_gsk3"] = v_pdx1_pdeg_gsk3

    flux_dict["k_m_prod"] = v_mafa_prod
    flux_dict["k_mafa_efflux_ros"] = v_mafa_efflux_ros
    flux_dict["k_mafa_pdeg_gsk3"] = v_mafa_pdeg_gsk3

    flux_dict["k_ins_prod"] = v_ins_prod
    flux_dict["k_ins_deg"] = v_ins_deg
    flux_dict["k_proinsulin_prod"] = v_proinsulin_prod
    flux_dict["k_folding_deg_chap"] = v_folding_deg_chap

    flux_dict["k_chap_act"] = v_chap_act
    flux_dict["k_chap_deg"] = v_chap_deg
    flux_dict["k_ins_folding"] = v_insulin_folding
    flux_dict["k_ins_secretion"] = v_insulin_secretion
    flux_dict["k_ins_clearance"] = v_insulin_clearance

    flux_dict["k_Rs_prod"] = v_superoxide_prod
    flux_dict["k_superoxide_to_h2o2"] = v_superoxide_to_h2o2
    flux_dict["k_h2o2_clearance"] = v_h2o2_clearance
    flux_dict["k_h2o2_prod_er"] = v_h2o2_prod_er

    return flux_dict

@cython.boundscheck(False)
@cython.cdivision(True)
cdef void model(double t, double *X_data, double *dydt, vector[double] params, vector[double] metblsm_par,
int ros_signal, int insulin_signal, int aktp_signal, int foxo1_signal, int gsk3_signal):
    '''
    The ODE equations for the model
    :param t: time step
    :param X_data: the array of variable values
    :param dydt: the array where the derivatives of the variable will be stored
    :param params: estimated parameters from stage 2 optimization
    :param metblsm_par: estimated parameters from stage 1 optimization
    :return: an array of the derivatives of the variables
    '''

    # calling function containing the known parameter values from literature
    cdef unordered_map[int, double] pm = parameter_vals(metblsm_par)

    # the fluxes for glucose metabolism, calcium signalling and electrical activity
    cdef unordered_map[char*, double] flux = flux_data(X_data, params, metblsm_par)

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
    dydt[9] = (pm[38] * flux["J_O"] - flux["J_FIFO"] - flux["J_ANT"] - flux["J_NCX"] - 2 * flux["J_MCU"]) / pm[39]

    # Cytosolic Ca2+
    dydt[10] = pm[40] * (flux["J_pm"] + pm[2] / pm[0] * flux["J_IP3R"] - pm[2] / pm[0] * flux["J_SERCA"] - pm[3] / pm[0] * flux["J_MCU"] + pm[3] / pm[0] * flux["J_NCX"])

    # Mitochondrial Ca2+
    dydt[11] = pm[41] * (flux["J_MCU"] - flux["J_NCX"])

    # ER Ca2+
    dydt[12] = pm[42] * (flux["J_SERCA"] - flux["J_IP3R"])

    # Extracellular Ca2+
    dydt[13] = -pm[0] * flux["J_pm"]

    # fraction of inactive IP3 receptors
    dydt[14] = pm[43] * (pow(X_data[10], pm[45])) * ((1 - X_data[14]) / (1 + pow((X_data[10] / 0.3), pm[13]))) - pm[44] * X_data[14]

    # Rate of change of cell membrane potential
    dydt[15] = -(flux["I_ca"] + flux["I_k"] + flux["I_katp"]) / pm[64]

    # Rate of change of the fraction of channels closed
    dydt[16] = -flux["alpha"] * X_data[16] + flux["beta"] * X_data[17]

    # Rate of change of the fraction of channels open
    dydt[17] = pm[65] * (1 - X_data[17] - X_data[16]) - pm[66] * flux["ghk"] * X_data[17] + flux["alpha"] * X_data[16] - flux["beta"] * X_data[17]

    # Rate of change of the fraction of slow channels not inactivated.
    dydt[18] = (flux["j_inf"] - X_data[18]) / flux["tau_j"]

    # Rate of actiavtion of the K+ channels
    dydt[19] = (flux["n_inf"] - X_data[19]) / flux["tau_n"]

    # rate of change of AKT
    dydt[20] = -flux["k_akt_ins"] + flux["k_aktp"] + flux["k_akt_jnk"]
    dydt[21] = -dydt[20]

    # rate of change of GSK3
    dydt[22] = -flux["k_g_akt"] + flux["k_gsk3p"]
    dydt[23] = -dydt[22]

    # rate of change of JNK
    dydt[24] = flux["k_jnkp"] - flux["k_jnk_ros"] - flux["k_jnk_upr"]
    dydt[25] = -dydt[24]

    # rate of change of p38MAPK
    dydt[26] = flux["k_p38p"] - flux["k_p38_ros"] - flux["k_p38_upr"]
    dydt[27] = -dydt[26]

    # rate of change of FOXO1
    dydt[28] = -flux["k_f1c_akt"] + flux["k_foxo1c_mapk"]
    dydt[29] = -dydt[28]

    # rate of change of PDX1
    dydt[30] = flux["k_p_prod"] - flux["k_pdx1_efflux_foxo1"] - flux["k_pdx1_pdeg_gsk3"]

    # rate of change of MAFA
    dydt[31] = flux["k_m_prod"] - flux["k_mafa_efflux_ros"] - flux["k_mafa_pdeg_gsk3"]

    # rate of change of Insulin mRNA
    dydt[32] = flux["k_ins_prod"] - flux["k_ins_deg"] - flux["k_proinsulin_prod"]

    # rate of change of translated unfolded Insulin protein
    dydt[33] = flux["k_proinsulin_prod"] - flux["k_folding_deg_chap"]

    # rate of change of ER chaperones
    dydt[34] = flux["k_chap_act"] - flux["k_chap_deg"]

    # rate of change of folded Insulin protein
    dydt[35] = flux["k_ins_folding"] - flux["k_ins_secretion"]

    # rate of change of superoxide
    dydt[36] = flux["k_Rs_prod"] - flux["k_superoxide_to_h2o2"]

    # rate of change of hydrogen peroxide (H2O2)
    if ros_signal == 1:
        dydt[37] = 0
    else:
        dydt[37] = flux["k_superoxide_to_h2o2"] + flux["k_h2o2_prod_er"] - flux["k_h2o2_clearance"]

    # extra-cellular insulin
    if insulin_signal == 1:
        dydt[38] = 0
    else:
        dydt[38] = flux["k_ins_secretion"] - flux["k_ins_clearance"]


cdef vector[double] steady_state_initials(vector[double] X_data_0, vector[double] params_est_1, vector[double] params_est_2):
    '''
    :param X_data_0: the vector of initial values
    :param params_est_1: the estimated parameters from stage 1 optimization
    :param params_est_2: the estimated parameters from stage 2 optimization
    :return: steady-state values of the variables
    '''

    cdef:
        LSODA lsoda
        int istate = 1
        vector[double] yout
        vector[double] X_data_ini = X_data_0
        size_t neq = 39     # number of equations (ODEs)
        double t = 0
        double tout = 0
        size_t iout, j, mv, pv
        int num_steps = 360

    for iout in range(1, num_steps):
        tout += 10
        lsoda.lsoda_update(model, neq, X_data_ini, yout, &t, tout, &istate, params_est_2, 1e-8, 1e-8, params_est_1, 0, 0, 0, 0, 0)
        #Update the y for next iteration
        for j in range(neq):
            X_data_ini[j] = yout[j+1]

        if (istate <= 0):
            exit


    return X_data_ini

@cython.boundscheck(False)
def validation_PDX1_INSm_gluc(vector[double] X_data_0, vector[double] t_measured, vector[double] params_est_1, vector[double] params_est_2):
    '''
    :param X_data_0: the vector of initial values
    :param t_measured: the vector of time-points measured in the experiment
    :param params_est_1: the estimated parameters from stage 1 optimization
    :param params_est_2: the estimated parameters from stage 2 optimization
    :return: the simulated values of the concerned variables
    '''

    cdef:
        LSODA lsoda
        int istate = 1
        vector[double] res_1, res_2
        vector[double] yout
        vector[double] X_data_ini = X_data_0
        size_t neq = 39     # number of equations (ODEs)
        double t = 0
        double tout
        size_t iout, j, i, k, I, s
        vector[double] PDX1_ratio_simulated, INSm_ratio_simulated

    I = t_measured.size()

    X_data_ini[0] = 25000
    res_1.push_back(X_data_ini[30] / X_data_0[30])
    res_2.push_back(X_data_ini[32] / X_data_0[32])

    for iout in range(1, I):
        tout = t_measured[iout]
        lsoda.lsoda_update(model, neq, X_data_ini, yout, &t, tout, &istate, params_est_2, 1e-8, 1e-8, params_est_1, 0, 0, 0, 0, 0)
        #Update the y for next iteration
        for j in range(neq):
            X_data_ini[j] = yout[j+1]

        res_1.push_back(X_data_ini[30] / X_data_0[30])
        res_2.push_back(X_data_ini[32] / X_data_0[32])

        if (istate <= 0):
            exit

    if istate < 2:
        for i in range(I):
            PDX1_ratio_simulated.push_back(1000)
            INSm_ratio_simulated.push_back(1000)
    else:
        for i in range(I):
            PDX1_ratio_simulated.push_back(res_1[i])
            INSm_ratio_simulated.push_back(res_2[i])


    return PDX1_ratio_simulated, INSm_ratio_simulated


@cython.boundscheck(False)
def validation_PDX1_H2O2(vector[double] X_data_0, vector[double] t_measured, vector[double] params_est_1, vector[double] params_est_2):
    '''
    :param X_data_0: the vector of initial values
    :param t_measured: the vector of time-points measured in the experiment
    :param params_est_1: the estimated parameters from stage 1 optimization
    :param params_est_2: the estimated parameters from stage 2 optimization
    :return: the simulated values of the concerned variables
    '''

    cdef:
        LSODA lsoda
        #LSODA_ODE_SYSTEM_TYPE f = model
        int istate_1 = 1, istate_2 = 1
        vector[double] res_1, res_2
        vector[double] yout
        vector[double] X_data_ini = X_data_0
        size_t neq = 39     # number of equations (ODEs)
        double t = 0
        double tout
        size_t iout, j, i, k, I, s
        double PDX1_control_simulated, PDX1_H2O2_simulated

    I = t_measured.size()

    X_data_ini[0] = 11100

    for iout in range(0, I):
        tout = t_measured[iout]
        lsoda.lsoda_update(model, neq, X_data_ini, yout, &t, tout, &istate_1, params_est_2, 1e-8, 1e-8, params_est_1, 0, 0, 0, 0, 0)
        #Update the y for next iteration
        for j in range(neq):
            X_data_ini[j] = yout[j+1]

        res_1.push_back(X_data_ini[30] / X_data_0[30])
        #print(X_data_ini[30], X_data_0[30])
        if (istate_1 <= 0):
            exit

    X_data_ini = X_data_0
    X_data_ini[0] = 11100
    X_data_ini[37] = 50
    t = 0
    for iout in range(0, I):
        tout = t_measured[iout]
        lsoda.lsoda_update(model, neq, X_data_ini, yout, &t, tout, &istate_2, params_est_2, 1e-8, 1e-8, params_est_1, 1, 0, 0, 0, 0)
        #Update the y for next iteration
        for j in range(neq):
            X_data_ini[j] = yout[j+1]

        res_2.push_back(X_data_ini[30] / X_data_0[30])
        #print(X_data_ini[30], X_data_0[30])
        if (istate_2 <= 0):
            exit

    if (istate_1 < 2 or istate_2 < 2):
        PDX1_control_simulated = 1000
        PDX1_H2O2_simulated = 1000
    else:
        PDX1_control_simulated = res_1[0]
        PDX1_H2O2_simulated = res_2[0]

    return PDX1_control_simulated, PDX1_H2O2_simulated

def validation_PDX1_AKT_GSK3(vector[double] X_data_0, vector[double] params_est_1, vector[double] params_est_2):
    '''
    :param X_data_0: the vector of initial values
    :param params_est_1: the estimated parameters from stage 1 optimization
    :param params_est_2: the estimated parameters from stage 2 optimization
    :return: the simulated values of the concerned variables
    '''

    cdef:
        LSODA lsoda
        int istate_1 = 1, istate_2 = 1, istate_3 = 1
        vector[double] res_1, res_2, res_3
        vector[double] yout
        vector[double] X_data_ini = X_data_0
        size_t neq = 39     # number of equations (ODEs)
        double t = 0
        double tout = 0
        size_t iout, j, i, k, I, s
        double PDX1_aktpI, PDX1_gsk3I, PDX1_aktpI_gsk3I
        int num_steps = 10000

    X_data_ini[21] = 0.000000000001
    res_1.push_back(X_data_ini[30] / X_data_0[30])
    for iout in range(1, num_steps):
        tout += 10
        lsoda.lsoda_update(model, neq, X_data_ini, yout, &t, tout, &istate_1, params_est_2, 1e-8, 1e-8, params_est_1, 0, 0, 1, 0, 0)
        #Update the y for next iteration
        for j in range(neq):
            X_data_ini[j] = yout[j+1]

        res_1.push_back(X_data_ini[30] / X_data_0[30])

        if (istate_1 <= 0):
            exit

    t = 0
    tout = 0
    X_data_ini = X_data_0
    X_data_ini[22] = 0.000000000001
    res_2.push_back(X_data_ini[30] / X_data_0[30])
    for iout in range(1, num_steps):
        tout += 10
        lsoda.lsoda_update(model, neq, X_data_ini, yout, &t, tout, &istate_2, params_est_2, 1e-8, 1e-8, params_est_1, 0, 0, 0, 0, 1)
        #Update the y for next iteration
        for j in range(neq):
            X_data_ini[j] = yout[j+1]

        res_2.push_back(X_data_ini[30] / X_data_0[30])

        if (istate_2 <= 0):
            exit

    t = 0
    tout = 0
    X_data_ini = X_data_0
    X_data_ini[21] = 0.1
    X_data_ini[22] = 0.1
    res_3.push_back(X_data_ini[30] / X_data_0[30])
    for iout in range(1, num_steps):
        tout += 10
        lsoda.lsoda_update(model, neq, X_data_ini, yout, &t, tout, &istate_3, params_est_2, 1e-8, 1e-8, params_est_1, 0, 0, 1, 0, 1)
        #Update the y for next iteration
        for j in range(neq):
            X_data_ini[j] = yout[j+1]

        res_3.push_back(X_data_ini[30] / X_data_0[30])

        if (istate_3 <= 0):
            exit

    if (istate_1 < 2 or istate_2 < 2):
        PDX1_aktpI = 1000
        PDX1_gsk3I
        PDX1_aktpI_gsk3I = 1000
    else:
        PDX1_aktpI = res_1[num_steps-1]
        PDX1_gsk3I = res_2[num_steps-1]
        PDX1_aktpI_gsk3I = res_3[num_steps-1]

    return PDX1_aktpI, PDX1_gsk3I, PDX1_aktpI_gsk3I

def validation_MAFA_H202(vector[double] X_data_0, vector[double] t_measured, vector[double] params_est_1, vector[double] params_est_2):
    '''
    :param X_data_0: the vector of initial values
    :param t_measured: the vector of time-points measured in the experiment
    :param params_est_1: the estimated parameters from stage 1 optimization
    :param params_est_2: the estimated parameters from stage 2 optimization
    :return: the simulated values of the concerned variables
    '''

    cdef:
        LSODA lsoda
        int istate_1 = 1
        vector[double] MAFA_simulated
        vector[double] yout
        vector[double] X_data_ini = X_data_0
        size_t neq = 39     # number of equations (ODEs)
        double t = 0
        double tout = 0
        size_t iout, j, i, k, I, s

    I = t_measured.size()
    X_data_ini[37] = 50

    for iout in range(1, I):
        tout = t_measured[iout]
        lsoda.lsoda_update(model, neq, X_data_ini, yout, &t, tout, &istate_1, params_est_2, 1e-8, 1e-8, params_est_1, 1, 0, 0, 0, 0)
        #Update the y for next iteration
        for j in range(neq):
            X_data_ini[j] = yout[j+1]

        MAFA_simulated.push_back(X_data_ini[31] / X_data_0[31])

        if (istate_1 <= 0):
            exit

    return MAFA_simulated

def validation_MAFA_gluc(vector[double] X_data_0, vector[double] params_est_1, vector[double] params_est_2):
    '''
    :param X_data_0: the vector of initial values
    :param params_est_1: the estimated parameters from stage 1 optimization
    :param params_est_2: the estimated parameters from stage 2 optimization
    :return: the simulated values of the concerned variables
    '''

    cdef:
        LSODA lsoda
        #LSODA_ODE_SYSTEM_TYPE f = model
        int istate_1 = 1, istate_2 = 1
        vector[double] MAFA_simulated
        vector[double] yout
        vector[double] X_data_ini = X_data_0, X_data_ini_2
        size_t neq = 39     # number of equations (ODEs)
        double t = 0
        double tout = 0
        size_t iout, j, i, k, I, s
        int num_steps = 10000

    X_data_ini[0] = 4000
    for iout in range(1, num_steps):
        tout += 10
        lsoda.lsoda_update(model, neq, X_data_ini, yout, &t, tout, &istate_1, params_est_2, 1e-8, 1e-8, params_est_1, 0, 0, 0, 0, 0)
        #Update the y for next iteration
        for j in range(neq):
            X_data_ini[j] = yout[j+1]

        if (istate_1 <= 0):
            exit
    MAFA_simulated.push_back(X_data_ini[31] / X_data_ini[31])

    t = 0
    X_data_ini_2 = X_data_ini
    X_data_ini_2[0] = 16700
    for iout in range(1, num_steps):
        tout += 10
        lsoda.lsoda_update(model, neq, X_data_ini_2, yout, &t, tout, &istate_2, params_est_2, 1e-8, 1e-8, params_est_1, 0, 0, 0, 0, 0)
        #Update the y for next iteration
        for j in range(neq):
            X_data_ini_2[j] = yout[j+1]

        if (istate_2 <= 0):
            exit
    MAFA_simulated.push_back(X_data_ini_2[31] / X_data_ini[31])

    return MAFA_simulated


def validation_experiments():
    '''
    The fitness function
    :return: the fitness value
    '''

    cdef:
        double[:] t_measured_pdx1_gluc_view, t_measured_mafa_view, t_measured_pdx1_h2o2_view, t_measured_mafa_h2o2_view
        double[:] pdx1_gluc_view, insm_gluc_view, pdx1_control_view, pdx1_h2o2_view, mafa_h2o2_view, mafa_gluc_view

        double[:] params_est_1_view, params_est_2_view

        vector[double] X_data_initial
        vector[double] t_measured_pdx1_gluc, t_measured_pdx1_h2o2, t_measured_mafa_h2o2  # 1D vector
        vector[double] pdx1_gluc, insm_gluc, mafa_h2o2, mafa_gluc, params_est_1
        double pdx1_control, pdx1_h2o2

        int i, j, k, ti, mi, pi


    # Reading data from files
    df1 = pd.read_csv("data/validation_PDX1_INSm_glucose.csv", decimal=".", delimiter=",")
    t_measured_pdx1_gluc_np = np.array(df1["Time(sec)"])
    t_measured_pdx1_gluc_np = t_measured_pdx1_gluc_np.astype("double")    # typecasting to double
    pdx1_gluc_np = np.array(df1["PDX1/PDX1(control)"])
    pdx1_gluc_np = pdx1_gluc_np.astype("double")
    insm_gluc_np = np.array(df1["relative INS mRNA"])
    insm_gluc_np = insm_gluc_np.astype("double")

    df2 = pd.read_csv("data/validation_PDX1_H2O2.csv", decimal=".", delimiter=",")
    t_measured_pdx1_h2o2_np = np.array(df2["Time(sec)"])
    t_measured_pdx1_h2o2_np = t_measured_pdx1_h2o2_np.astype("double")
    pdx1_control_np = np.array(df2["PDX1(control)"])
    pdx1_control_np = pdx1_control_np.astype("double")
    pdx1_h2o2_np = np.array(df2["PDX1(H2O2)"])
    pdx1_h2o2_np = pdx1_h2o2_np.astype("double")

    df3 = pd.read_csv("data/validation_MAFA_H202.csv", decimal=".", delimiter=",")
    t_measured_mafa_h2o2_np = np.array(df3["Time (secs)"])
    t_measured_mafa_h2o2_np = t_measured_mafa_h2o2_np.astype("double")
    mafa_h2o2_np = np.array(df3["Relative density"])
    mafa_h2o2_np = mafa_h2o2_np.astype("double")

    df4 = pd.read_csv("data/validation_MAFA_glucose.csv", decimal=".", delimiter=",")
    mafa_gluc_np = np.array(df4["Relative MAFA mRNA level"])
    mafa_gluc_np = mafa_gluc_np.astype("double")

    # reading estimated parameters from metabolism
    df5 = pd.read_csv("estimated_parameters/estimated_parameters_S1_1.csv", decimal=".", delimiter=",")
    params_est_1_np = np.array(df5["Values"])

    t_measured_pdx1_gluc_view = t_measured_pdx1_gluc_np         # 1D memoryview
    t_measured_mafa_view = t_measured_mafa_np
    t_measured_pdx1_h2o2_view = t_measured_pdx1_h2o2_np
    t_measured_mafa_h2o2_view = t_measured_mafa_h2o2_np
    pdx1_gluc_view = pdx1_gluc_np
    insm_gluc_view = insm_gluc_np
    pdx1_control_view = pdx1_control_np
    pdx1_h2o2_view = pdx1_h2o2_np
    mafa_h2o2_view = mafa_h2o2_np
    mafa_gluc_view = mafa_gluc_np
    params_est_1_view = params_est_1_np


    # casting memoryview into vector
    for ti in range(t_measured_pdx1_gluc_view.shape[0]):
        t_measured_pdx1_gluc.push_back(t_measured_pdx1_gluc_view[ti])
        pdx1_gluc.push_back(pdx1_gluc_view[ti])
        insm_gluc.push_back(insm_gluc_view[ti])

    for ti in range(t_measured_mafa_view.shape[0]):
        t_measured_mafa.push_back(t_measured_mafa_view[ti])
        mafa_gluc_4.push_back(mafa_gluc_4_view[ti])
        mafa_gluc_16.push_back(mafa_gluc_16_view[ti])

    for ti in range(t_measured_pdx1_h2o2_view.shape[0]):
        t_measured_pdx1_h2o2.push_back(t_measured_pdx1_h2o2_view[ti])
        pdx1_control = pdx1_control_view[ti]
        pdx1_h2o2 = pdx1_h2o2_view[ti]

    for ti in range(t_measured_mafa_h2o2_view.shape[0]):
        t_measured_mafa_h2o2.push_back(t_measured_mafa_h2o2_view[ti])
        mafa_h2o2.push_back(mafa_h2o2_view[ti])

    for ti in range(mafa_gluc_view.shape[0]):
        mafa_gluc.push_back(mafa_gluc_view[ti])

    for ti in range(params_est_1_view.shape[0]):
        params_est_1.push_back(params_est_1_view[ti])


    param_files = sorted(glob.glob("estimated_parameters/parameters_S2_files/*.csv"))
    params_est_2 = []
    for f in param_files:
        df = pd.read_csv(f)
        params_est_2.append(np.array(df['Values']))
    params_est_2 = np.array(params_est_2)
    print(params_est_2.shape)
    num_files = params_est_2.shape[0]

    PDX1_ratio_sim = []
    INSm_ratio_sim = []
    PDX1_control_sim = []
    PDX1_H2O2_sim = []
    PDX1_aktpI_sim = []
    PDX1_gsk3I_sim = []
    PDX1_aktpI_gsk3I_sim = []
    FOXO1_sim = []
    PDX1_sim = []
    MAFA_sim_H2O2 = []
    MAFA_sim_gluc = []

    for pi in range(num_files):
        X_data_lit = initial_vals(params_est_2[pi])
        X_data_initial = steady_state_initials(X_data_lit, params_est_1, params_est_2[pi])

        PDX1_ratio_simulated, INSm_ratio_simulated = validation_PDX1_INSm_gluc(X_data_initial, t_measured_pdx1_gluc, params_est_1, params_est_2[pi])

        PDX1_ratio_sim.append(PDX1_ratio_simulated)
        INSm_ratio_sim.append(INSm_ratio_simulated)

        PDX1_control_simulated, PDX1_H2O2_simulated = validation_PDX1_H2O2(X_data_initial, t_measured_pdx1_h2o2, params_est_1, params_est_2[pi])

        PDX1_control_sim.append(PDX1_control_simulated)
        PDX1_H2O2_sim.append(PDX1_H2O2_simulated)

        PDX1_aktpI, PDX1_gsk3I, PDX1_aktpI_gsk3I = validation_PDX1_AKT_GSK3(X_data_initial, params_est_1, params_est_2[pi])

        PDX1_aktpI_sim.append(PDX1_aktpI)
        PDX1_gsk3I_sim.append(PDX1_gsk3I)
        PDX1_aktpI_gsk3I_sim.append(PDX1_aktpI_gsk3I)

        MAFA_simulated_H2O2 = validation_MAFA_H202(X_data_initial, t_measured_mafa_h2o2, params_est_1, params_est_2[pi])
        MAFA_sim_H2O2.append(MAFA_simulated_H2O2)

        MAFA_simulated_gluc = validation_MAFA_gluc(X_data_initial, params_est_1, params_est_2[pi])
        MAFA_sim_gluc.append(MAFA_simulated_gluc)

    fontsz = 14
    mksize = 12
    ticksz = 13

    fig, axs = plt.subplots(3, 2, figsize=(12,8))
    fig.subplots_adjust(hspace=0.34, wspace=0.16, left=0.07, right=0.97, top=0.95, bottom=0.14)

    for axi in axs.flat:
        axi.tick_params(axis='both', which='major', labelsize=ticksz)

    X1 = ['Control', '48 h (gluc)', '72 h (gluc)']
    simulated_PDX1 = np.mean(PDX1_ratio_sim, axis=0)
    simulated_INSm = np.mean(INSm_ratio_sim, axis=0)
    error_PDX1 = np.std(PDX1_ratio_sim, axis=0)
    error_INSm = np.std(INSm_ratio_sim, axis=0)
    X_axis = np.arange(len(X1))

    axs[0, 0].bar(X_axis - 0.08, pdx1_gluc, yerr=[0, 0.022, 0.059], capsize=5, width=0.15, label='experimental')
    axs[0, 0].bar(X_axis + 0.08, simulated_PDX1, yerr=error_PDX1, capsize=5, width=0.15, label='simulated')
    axs[0, 0].set_xticks(X_axis)
    axs[0, 0].set_xticklabels(X1)

    axs[0, 0].set_ylabel("PDX1(t)/PDX1(C)", fontsize=fontsz)
    axs[0, 0].set_title("(a) PDX1 (25 mM glucose)", fontsize=fontsz)
    axs[0, 0].text(1.0, 0.9, '0.98', fontsize = fontsz, color='red')
    axs[0, 0].text(2.0, 0.76, '1.35', fontsize = fontsz, color='red')

    axs[0, 1].bar(X_axis - 0.08, insm_gluc, yerr=[0, 0.076, 0.13], capsize=5, width=0.15, label='experimental')
    axs[0, 1].bar(X_axis + 0.08, simulated_INSm, yerr=error_INSm, capsize=5, width=0.15, label='simulated')
    axs[0, 1].set_xticks(X_axis)
    axs[0, 1].set_xticklabels(X1)
    axs[0, 1].set_ylabel("INSm(t)/INSm(C)", fontsize=fontsz)
    axs[0, 1].set_title("(b) Insulin mRNA (25 mM glucose)", fontsize=fontsz)
    axs[0, 1].text(1.0, 0.84, '1.41', fontsize = fontsz, color='red')
    axs[0, 1].text(2.0, 0.84, '1.32', fontsize = fontsz, color='red')

    X2 = ['Control', 'H2O2']
    measured_PDX1_H2O2 = [1.0, 0.03]
    simulated_PDX1_H2O2 = [1.0, 0.2]
    error_PDX1_H2O2 = [0.0, np.std(PDX1_H2O2_sim)]
    X_axis = np.arange(len(X2))

    axs[1, 0].bar(X_axis - 0.08, measured_PDX1_H2O2, yerr=[0, 0.01], capsize=5, width=0.15, label='experimental')
    axs[1, 0].bar(X_axis + 0.08, simulated_PDX1_H2O2, yerr=error_PDX1_H2O2, capsize=5, width=0.15, label='simulated')
    axs[1, 0].set_xticks(X_axis)
    axs[1, 0].set_xticklabels(X2)
    #axs[1, 0].set_xlabel("Experiment", fontsize=fontsz)
    axs[1, 0].set_ylabel("PDX1($H_2O_2$)/PDX1(C)", fontsize=fontsz)
    axs[1, 0].set_title("(c) PDX1 (50 $\mu$M $H_2O_2$)", fontsize=fontsz)
    axs[1, 0].text(1.03, 0.25, '6.67', fontsize = fontsz, color='red')

    X3 = ['Control', 'AKT-I', 'GSK3-I', '(AKT+GSK3)-I']
    measured_PDX1_AKT = [1.0, 0.5, 1.75, 1.25]
    simulated_PDX1_AKT = [1.0, 0.1, np.mean(PDX1_gsk3I_sim), np.mean(PDX1_aktpI_gsk3I_sim)]
    error_PDX1_AKT = [0.0, np.std(PDX1_aktpI_sim), np.std(PDX1_gsk3I_sim), np.std(PDX1_aktpI_gsk3I_sim)]
    X_axis = np.arange(len(X3))

    axs[1, 1].bar(X_axis - 0.08, measured_PDX1_AKT, yerr=[0, 0.058, 0.433, 0.397], capsize=5, width=0.15, label='experimental')
    axs[1, 1].bar(X_axis + 0.08, simulated_PDX1_AKT, yerr=error_PDX1_AKT, capsize=5, width=0.15, label='simulated')
    axs[1, 1].set_xticks(X_axis)
    axs[1, 1].set_xticklabels(X3)
    axs[1, 1].set_ylabel("PDX1(I)/PDX1(C)", fontsize=fontsz)
    axs[1, 1].set_title("(d) PDX1 (AKT and/or GSK3 inhibition)", fontsize=fontsz)
    axs[1, 1].text(1.04, 0.25, '0.2', fontsize = fontsz, color='red')
    axs[1, 1].text(2.0, 2.05, '1.02', fontsize = fontsz, color='red')
    axs[1, 1].text(3.0, 0.66, '0.26', fontsize = fontsz, color='red')

    X4 = ['Control', '30 min', '60 min', '90 min']
    measured_MAFA_H2O2 = mafa_h2o2
    simulated_MAFA_H2O2 = np.mean(MAFA_sim_H2O2, axis=0)
    simulated_MAFA_H2O2 = np.insert(simulated_MAFA_H2O2, 0, 1.0)
    error_MAFA_H2O2 = np.std(MAFA_sim_H2O2, axis=0)
    error_MAFA_H2O2 = np.insert(error_MAFA_H2O2, 0, 0.0)
    X_axis = np.arange(len(X4))

    axs[2, 0].bar(X_axis - 0.08, measured_MAFA_H2O2, capsize=5, width=0.15, label='experimental')
    axs[2, 0].bar(X_axis + 0.08, simulated_MAFA_H2O2, yerr=error_MAFA_H2O2, capsize=5, width=0.15, label='simulated')
    axs[2, 0].set_xticks(X_axis)
    axs[2, 0].set_xticklabels(X4)
    axs[2, 0].set_xlabel("Experiment", fontsize=fontsz)
    axs[2, 0].set_ylabel("MAFA(t)/MAFA(C)", fontsize=fontsz)
    axs[2, 0].set_title("(e) MAFA (50 $\mu$M $H_2O_2$)", fontsize=fontsz)
    axs[2, 0].text(0.95, 0.7, '2.45', fontsize = fontsz, color='red')
    axs[2, 0].text(1.9, 0.38, '18.82', fontsize = fontsz, color='red')
    axs[2, 0].text(2.9, 0.26, '30.23', fontsize = fontsz, color='red')

    X5 = ['Gluc 4mM', 'Gluc 16.7mM']
    measured_MAFA_gluc = mafa_gluc
    simulated_MAFA_gluc = np.mean(MAFA_sim_gluc, axis=0)
    error_MAFA_gluc = np.std(MAFA_sim_gluc, axis=0)
    X_axis = np.arange(len(X5))

    axs[2, 1].bar(X_axis - 0.08, measured_MAFA_gluc, yerr=[0.1, 0.05], capsize=5, width=0.15, label='experimental')
    axs[2, 1].bar(X_axis + 0.08, simulated_MAFA_gluc, yerr=error_MAFA_gluc, capsize=5, width=0.15, label='simulated')
    axs[2, 1].set_xticks(X_axis)
    axs[2, 1].set_xticklabels(X5)
    axs[2, 1].set_xlabel("Experiment", fontsize=fontsz)
    axs[2, 1].set_ylabel("MAFA(G)/MAFA(G=4)", fontsize=fontsz)
    axs[2, 1].set_title("(f) MAFA (Glucose)", fontsize=fontsz)
    axs[2, 1].text(1.02, 0.8, '1.29', fontsize = fontsz, color='red')

    axs.flatten()[-2].legend(loc='upper center', fontsize=fontsz, bbox_to_anchor=(1.1, -0.2), ncol=2)
    #fig.tight_layout()
    plt.savefig('figures/final_validation/Figure_quant_val.png')
