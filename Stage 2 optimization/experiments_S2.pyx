# cython: language_level=3
import numpy as np
import pandas as pd
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
cdef vector[double] initial_vals(vector[double] params) nogil:
    '''
    This function assigns the initial values of the variables.
    :param params: 1D vector of parame values assigned to one particle
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
cdef unordered_map[int, double] parameter_vals(vector[double] metblsm_par) nogil:
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
cdef unordered_map[char*, double] flux_data(double *X_data, vector[double] params, vector[double] metblsm_par) nogil:
    '''
    Defining the fluxes for glucose metabolism, calcium signalling and electrical activity
    :param X_data: the array of variable values
    :param params: the free parameters
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
int ros_signal, int insulin_signal) nogil:
    '''
    The ODE equations for the model
    :param t: time step
    :param X_data: the array of variable values
    :param dydt: the array where the derivatives of the variable will be stored
    :param params: the free parameters
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


@cython.boundscheck(False)
cdef double experiment_mapk_ros(vector[double] X_data_0, vector[double] params, vector[double] t_measured_mapk,
vector[double] JNK_ratio_measured, vector[double] p38_ratio_measured, vector[double] t_measured_insulin_mrna,
vector[double] insulin_mrna_ratio_measured, vector[double] metblsm_par, double mean_jnk, double mean_p38, double mean_insm) nogil:
    '''
    Replication of experiment number 1 through simulaiton
    Integration is done using the LSODA algorithm
    :param gluc_stm: the glucose stimulation used for this experiment
    :param X_data_0: the vector of initial values
    :param params: the vector of free parameters
    :param t_measured_mapk, t_measured_insulin_mrna: the vectors of time-points measured in the experiment
    :param JNK_ratio_measured, p38_ratio_measured, insulin_mrna_ratio_measured: experimental data
    :param metblsm_par: estimated parameters from stage 1 optimization
    :param mean_jnk, mean_p38, mean_insm: mean of the experimental data
    :return: the mean squared error between the measured and simulated values
    '''

    cdef:
        LSODA lsoda
        int istate_1 = 1, istate_2 = 1
        vector[double] res_1, res_2, res_3
        vector[double] yout
        vector[double] X_data_ini = X_data_0
        size_t neq = 39     # number of equations (ODEs)
        double t = 0
        double tout
        size_t iout, j, i, k, I, s
        vector[double] JNK_ratio_simulated, p38_ratio_simulated, insulin_mrna_ratio_simulated
        vector[double] deviation_jnk, deviation_p38, deviation_insulin_mrna

    I = t_measured_mapk.size()

    # JNK & p38
    X_data_ini[37] = 50
    res_1.push_back(X_data_ini[25] / X_data_ini[24])
    res_2.push_back(X_data_ini[27] / X_data_ini[26])

    for iout in range(1, I):
        tout = t_measured_mapk[iout]
        lsoda.lsoda_update(model, neq, X_data_ini, yout, &t, tout, &istate_1, params, 1e-8, 1e-8, metblsm_par, 1, 0)
        #Update the y for next iteration
        for j in range(neq):
            X_data_ini[j] = yout[j+1]

        res_1.push_back(X_data_ini[25] / X_data_ini[24])
        res_2.push_back(X_data_ini[27] / X_data_ini[26])

        if (istate_1 <= 0):
            exit

    # insulin mRNA
    X_data_ini = X_data_0
    X_data_ini[37] = 50
    res_3.push_back(X_data_ini[32] / X_data_0[32])
    t = 0
    for iout in range(1, I):
        tout = t_measured_insulin_mrna[iout]
        lsoda.lsoda_update(model, neq, X_data_ini, yout, &t, tout, &istate_2, params, 1e-8, 1e-8, metblsm_par, 1, 0)
        #Update the y for next iteration
        for j in range(neq):
            X_data_ini[j] = yout[j+1]

        res_3.push_back(X_data_ini[32] / X_data_0[32])

        if (istate_2 <= 0):
            exit

    if (istate_1 < 2 or istate_2 < 2):
        for i in range(I):
            JNK_ratio_simulated.push_back(1000)
            p38_ratio_simulated.push_back(1000)
            insulin_mrna_ratio_simulated.push_back(1000)
    else:
        for i in range(I):
            JNK_ratio_simulated.push_back(res_1[i])
            p38_ratio_simulated.push_back(res_2[i])
            insulin_mrna_ratio_simulated.push_back(res_3[i])

    for k in range(I):
        deviation_jnk.push_back(pow((JNK_ratio_simulated[k] - JNK_ratio_measured[k]), 2))
        deviation_p38.push_back(pow((p38_ratio_simulated[k] - p38_ratio_measured[k]), 2))
        deviation_insulin_mrna.push_back(pow((insulin_mrna_ratio_simulated[k] - insulin_mrna_ratio_measured[k]), 2))

    cdef double sum_1 = 0.0, sum_2 = 0.0, sum_3 = 0.0
    for s in range(I):
        sum_1 += deviation_jnk[s]
        sum_2 += deviation_p38[s]
        sum_3 += deviation_insulin_mrna[s]


    cdef double min_jnk = 1000, max_jnk = 0, min_p38 = 1000, max_p38 = 0, min_insm = 1000, max_insm = 0
    for i in range(I):
        if JNK_ratio_measured[i] < min_jnk:
            min_jnk = JNK_ratio_measured[i]
        if JNK_ratio_measured[i] >  max_jnk:
            max_jnk = JNK_ratio_measured[i]
        if p38_ratio_measured[i] < min_p38:
            min_p38 = p38_ratio_measured[i]
        if p38_ratio_measured[i] >  max_p38:
            max_p38 = p38_ratio_measured[i]
        if insulin_mrna_ratio_measured[i] < min_insm:
            min_insm = insulin_mrna_ratio_measured[i]
        if insulin_mrna_ratio_measured[i] >  max_insm:
            max_insm = insulin_mrna_ratio_measured[i]

    cdef double mse_deviation = sqrt(sum_1 / I) / (max_jnk - min_jnk) + sqrt(sum_2 / I) / (max_p38 - min_p38) + sqrt(sum_3 / I) / (max_insm - min_insm)

    return mse_deviation

@cython.boundscheck(False)
cdef double experiment_aktp_gsk3(vector[double] X_data_0, vector[double] params, vector[double] t_measured,
vector[double] aktp_ratio_measured, vector[double] gsk3_ratio_measured, vector[double] metblsm_par, double mean_aktp,
double mean_gsk3) nogil:
    '''
    Replication of experiment number 1 through simulaiton
    Integration is done using the LSODA algorithm
    :param gluc_stm: the glucose stimulation used for this experiment
    :param X_data_0: the vector of initial values
    :param params: the vector of free parameters
    :param t_measured: the vector of time-points measured in the experiment
    :param aktp_ratio_measured, gsk3_ratio_measured: experimental data
    :param metblsm_par: estimated parameters from stage 1 optimization
    :param mean_aktp, mean_gsk3: mean of the experimental data
    :return: the mean squared error between the measured and simulated values of insulin secretion rates
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
        vector[double] aktp_ratio_simulated, gsk3_ratio_simulated
        vector[double] deviation_aktp, deviation_gsk3

    I = t_measured.size()

    X_data_ini[38] = 0.12
    res_1.push_back(X_data_ini[21] / X_data_0[21])
    res_2.push_back(X_data_ini[22] / X_data_0[22])

    for iout in range(1, I):
        tout = t_measured[iout]
        lsoda.lsoda_update(model, neq, X_data_ini, yout, &t, tout, &istate, params, 1e-8, 1e-8, metblsm_par, 0, 1)
        #Update the y for next iteration
        for j in range(neq):
            X_data_ini[j] = yout[j+1]

        res_1.push_back(X_data_ini[21] / X_data_0[21])
        res_2.push_back(X_data_ini[22] / X_data_0[22])

        if (istate <= 0):
            exit

    if (istate < 2):
        for i in range(I):
            aktp_ratio_simulated.push_back(1000)
            gsk3_ratio_simulated.push_back(1000)
    else:
        for i in range(I):
            aktp_ratio_simulated.push_back(res_1[i])
            gsk3_ratio_simulated.push_back(res_2[i])

    for k in range(I):
        deviation_aktp.push_back(pow((aktp_ratio_simulated[k] - aktp_ratio_measured[k]), 2))
        deviation_gsk3.push_back(pow((gsk3_ratio_simulated[k] - gsk3_ratio_measured[k]), 2))

    cdef double sum_1 = 0.0, sum_2 = 0.0
    for s in range(I):
        sum_1 += deviation_aktp[s]
        sum_2 += deviation_gsk3[s]


    cdef double min_akt = 1000, max_akt = 0, min_gsk3 = 1000, max_gsk3 = 0
    for i in range(I):
        if aktp_ratio_measured[i] < min_akt:
            min_akt = aktp_ratio_measured[i]
        if aktp_ratio_measured[i] >  max_akt:
            max_akt = aktp_ratio_measured[i]
        if gsk3_ratio_measured[i] < min_gsk3:
            min_gsk3 = gsk3_ratio_measured[i]
        if gsk3_ratio_measured[i] >  max_gsk3:
            max_gsk3 = gsk3_ratio_measured[i]

    cdef double mse_deviation = sqrt(sum_1 / I) / (max_akt - min_akt) + sqrt(sum_2 / I) / (max_gsk3 - min_gsk3)

    return mse_deviation

@cython.boundscheck(False)
cdef double experiment_aktp_akt_ratio(vector[double] X_data_0, vector[double] params, vector[double] t_measured,
vector[double] aktp_akt_ratio_measured, vector[double] metblsm_par, double mean_aktp_akt) nogil:
    '''
    Replication of experiment number 1 through simulaiton
    Integration is done using the LSODA algorithm
    :param gluc_stm: the glucose stimulation used for this experiment
    :param X_data_0: the vector of initial values
    :param params: the vector of free parameters
    :param t_measured: the vector of time-points measured in the experiment
    :param aktp_akt_ratio_measured: experimental data
    :param metblsm_par: estimated parameters from stage 1 optimization
    :param mean_aktp_akt: mean of the experimental data
    :return: the mean squared error between the measured and simulated values of insulin secretion rates
    '''

    cdef:
        LSODA lsoda
        int istate = 1
        vector[double] res_1
        vector[double] yout
        vector[double] X_data_ini = X_data_0
        size_t neq = 39     # number of equations (ODEs)
        double t = 0
        double tout
        size_t iout, j, i, k, I, s
        vector[double] aktp_akt_ratio_simulated
        vector[double] deviation

    I = t_measured.size()

    X_data_ini[0] = 5500
    res_1.push_back(X_data_ini[21] / X_data_ini[20])

    for iout in range(1, I):
        tout = t_measured[iout]
        lsoda.lsoda_update(model, neq, X_data_ini, yout, &t, tout, &istate, params, 1e-8, 1e-8, metblsm_par, 0, 0)
        #Update the y for next iteration
        for j in range(neq):
            X_data_ini[j] = yout[j+1]

        res_1.push_back(X_data_ini[21] / X_data_ini[20])

        if (istate <= 0):
            exit

    if (istate < 2):
        for i in range(I):
            aktp_akt_ratio_simulated.push_back(1000)
    else:
        for i in range(I):
            aktp_akt_ratio_simulated.push_back(res_1[i])

    for k in range(I):
        deviation.push_back(pow((aktp_akt_ratio_simulated[k] - aktp_akt_ratio_measured[k]), 2))

    cdef double sum_1 = 0.0
    for s in range(I):
        sum_1 += deviation[s]


    cdef double min_akt = 1000, max_akt = 0
    for i in range(I):
        if aktp_akt_ratio_measured[i] < min_akt:
            min_akt = aktp_akt_ratio_measured[i]
        if aktp_akt_ratio_measured[i] >  max_akt:
            max_akt = aktp_akt_ratio_measured[i]

    cdef double mse_deviation = sqrt(sum_1 / I) / (max_akt - min_akt)

    return mse_deviation


def fitness(params_np):
    '''
    Fitness function given a set of parameters
    :param params: the current set of parameters to be evaluated
    :return: the fitness value
    '''

    cdef:
        double[:, :] params_view = params_np                    # 2D memoryview. defining a memoryview on a numpy array

        double[:] t_measured_mapk_view, t_measured_insm_view, t_measured_aktp_gsk3_view, t_measured_aktp_ratio_view
        double[:] JNK_ratio_view, p38_ratio_view, INSm_ratio_view, AKTp_ratio_view, GSK3_ratio_view, AKTp_AKT_ratio_view

        double[:] metblsm_par_view

        vector[vector[double]] params               # 2D vector
        vector[double] X_data_initial, X_data_lit
        vector[double] t_measured_mapk, t_measured_insm, t_measured_aktp_gsk3, t_measured_aktp_ratio  # 1D vector
        vector[double] JNK_ratio, p38_ratio, INSm_ratio, AKTp_ratio, GSK3_ratio, AKTp_AKT_ratio, metblsm_par

        double mean_jnk, mean_p38, mean_insm, mean_aktp, mean_gsk3, mean_aktp_akt

        int i, j, k, ti, mi, pi
        int num_particles = params_view.shape[0]

        vector[double] CP1_vect, CP2_vect, CP3_vect
        int NUM_THREADS = 16
        int CHUNKSIZE = int(num_particles/NUM_THREADS)

    # Reading data from files
    df1 = pd.read_csv("data/MAPK_ros_data.csv", decimal=".", delimiter=",")
    t_measured_mapk_np = np.array(df1["Time (sec)"])
    t_measured_mapk_np = t_measured_mapk_np.astype("double")    # typecasting to double
    JNK_ratio_np = np.array(df1["JNKp/JNK"])
    JNK_ratio_np = JNK_ratio_np.astype("double")
    p38_ratio_np = np.array(df1["p38p/p38"])
    p38_ratio_np = p38_ratio_np.astype("double")
    mean_jnk = np.mean(JNK_ratio_np)
    mean_p38 = np.mean(p38_ratio_np)

    df2 = pd.read_csv("data/insulin_mRNA_ROS_data.csv", decimal=".", delimiter=",")
    t_measured_insm_np = np.array(df2["Time (sec)"])
    t_measured_insm_np = t_measured_insm_np.astype("double")
    INSm_ratio_np = np.array(df2["insulin mRNA fraction"])
    INSm_ratio_np = INSm_ratio_np.astype("double")
    mean_insm = np.mean(INSm_ratio_np)

    df3 = pd.read_csv("data/akt_gsk3_data.csv", decimal=".", delimiter=",")
    t_measured_aktp_gsk3_np = np.array(df3["Time (sec)"])
    t_measured_aktp_gsk3_np = t_measured_aktp_gsk3_np.astype("double")
    AKTp_ratio_np = np.array(df3["AKTp(fraction)"])
    AKTp_ratio_np = AKTp_ratio_np.astype("double")
    GSK3_ratio_np = np.array(df3["GSK3(fraction)"])
    GSK3_ratio_np = GSK3_ratio_np.astype("double")
    mean_aktp = np.mean(AKTp_ratio_np)
    mean_gsk3 = np.mean(GSK3_ratio_np)

    df4 = pd.read_csv("data/aktp_akt_ratio.csv", decimal=".", delimiter=",")
    t_measured_aktp_ratio_np = np.array(df4["Time(sec)"])
    t_measured_aktp_ratio_np = t_measured_aktp_ratio_np.astype("double")
    AKTp_AKT_ratio_np = np.array(df4["AKTp/AKT"])
    AKTp_AKT_ratio_np = AKTp_AKT_ratio_np.astype("double")
    mean_aktp_akt = np.mean(AKTp_AKT_ratio_np)

    # reading estimated parameters from metabolism
    df5 = pd.read_csv("estimated_parameter_files/estimated_parameters_S1_1.csv", decimal=".", delimiter=",")
    metblsm_par_np = np.array(df5["Values"])

    t_measured_mapk_view = t_measured_mapk_np         # 1D memoryview
    t_measured_insm_view = t_measured_insm_np
    t_measured_aktp_gsk3_view = t_measured_aktp_gsk3_np
    t_measured_aktp_ratio_view = t_measured_aktp_ratio_np
    JNK_ratio_view = JNK_ratio_np
    p38_ratio_view = p38_ratio_np
    INSm_ratio_view = INSm_ratio_np
    AKTp_ratio_view = AKTp_ratio_np
    GSK3_ratio_view = GSK3_ratio_np
    AKTp_AKT_ratio_view = AKTp_AKT_ratio_np
    metblsm_par_view = metblsm_par_np

    # casting memoryview into vector
    for ti in range(t_measured_mapk_view.shape[0]):
        t_measured_mapk.push_back(t_measured_mapk_view[ti])
        JNK_ratio.push_back(JNK_ratio_view[ti])
        p38_ratio.push_back(p38_ratio_view[ti])

    for ti in range(t_measured_insm_view.shape[0]):
        t_measured_insm.push_back(t_measured_insm_view[ti])
        INSm_ratio.push_back(INSm_ratio_view[ti])

    for ti in range(t_measured_aktp_gsk3_view.shape[0]):
        t_measured_aktp_gsk3.push_back(t_measured_aktp_gsk3_view[ti])
        AKTp_ratio.push_back(AKTp_ratio_view[ti])
        GSK3_ratio.push_back(GSK3_ratio_view[ti])

    for ti in range(t_measured_aktp_ratio_view.shape[0]):
        t_measured_aktp_ratio.push_back(t_measured_aktp_ratio_view[ti])
        AKTp_AKT_ratio.push_back(AKTp_AKT_ratio_view[ti])

    for ti in range(metblsm_par_view.shape[0]):
        metblsm_par.push_back(metblsm_par_view[ti])

    cdef vector[double] temp
    for j in range(params_view.shape[0]):
        for k in range(params_view.shape[1]):
            temp.push_back(params_view[j][k])
        params.push_back(temp)
        temp.clear()

    # resizing vector; the first argument specifies the number of elements and the second the values of the elements
    CP1_vect.resize(num_particles, 1)
    CP2_vect.resize(num_particles, 1)
    CP3_vect.resize(num_particles, 1)

    # the parallel part without the GIL
    for pi in prange(num_particles, nogil=True, schedule="dynamic", num_threads=NUM_THREADS, chunksize=CHUNKSIZE):

        X_data_initial = initial_vals(params[pi])

        CP1_vect[openmp.omp_get_thread_num() * 0 + pi] = experiment_mapk_ros(X_data_initial, params[pi],
        t_measured_mapk, JNK_ratio, p38_ratio, t_measured_insm, INSm_ratio, metblsm_par, mean_jnk, mean_p38, mean_insm)

        CP2_vect[openmp.omp_get_thread_num() * 0 + pi] = experiment_aktp_gsk3(X_data_initial, params[pi],
        t_measured_aktp_gsk3, AKTp_ratio, GSK3_ratio, metblsm_par, mean_aktp, mean_gsk3)

        CP3_vect[openmp.omp_get_thread_num() * 0 + pi] = experiment_aktp_akt_ratio(X_data_initial, params[pi],
        t_measured_aktp_ratio, AKTp_AKT_ratio, metblsm_par, mean_aktp_akt)

    # casting vectors into Numpy:
    CP1 = np.array(CP1_vect)
    CP2 = np.array(CP2_vect)
    CP3 = np.array(CP3_vect)

    CP = CP1 + CP2 + CP3

    return CP