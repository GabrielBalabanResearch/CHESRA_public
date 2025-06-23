import pandas as pd
import sys
import sympy as sp
import numpy as np
import os
from pyDOE import lhs

p = '../../'
data_path = '../../'
sys.path.insert(0, p)
from CHESRA.data_classes.shear_sommer import ShearSommer
from CHESRA.data_classes.shear_dokos import ShearDokos

########################################################################################################################
# Definition of functions under consideration
########################################################################################################################

def psi_CH1(I1, I2, I3, I4f, I4s, I4n, I5f, I5s, I5n, I8fs, I8fn, I8ns, p):
    p1, p2, p3 = p

    I1 = (I1 - 3) ** 2
    I2 = (I2 - 3) ** 2
    I4n = (I4n - 1) ** 2
    I4f = (I4f - 1) ** 2
    I4s = (I4s - 1) ** 2
    I5f = (I5f - 1) ** 2
    I5s = (I5s - 1) ** 2
    I5n = (I5n - 1) ** 2
    I8fs = (I8fs) ** 2
    I8ns = (I8ns) ** 2
    I8fn = (I8fn) ** 2

    return (p1 + I1)*(p2+p3*(I8fs+I5f))

def psi_CH2(I1, I2, I3, I4f, I4s, I4n, I5f, I5s, I5n, I8fs, I8fn, I8ns, p):
    p1, p2, p3, p4 = p

    I1 = (I1 - 3) ** 2
    I2 = (I2 - 3) ** 2
    I4n = (I4n - 1) ** 2
    I4f = (I4f - 1) ** 2
    I4s = (I4s - 1) ** 2
    I5f = (I5f - 1) ** 2
    I5s = (I5s - 1) ** 2
    I5n = (I5n - 1) ** 2
    I8fs = (I8fs) ** 2
    I8ns = (I8ns) ** 2
    I8fn = (I8fn) ** 2

    return p1*(p2+I5f)*(p3+I1)*(p4+I5s)

def psi_MA(I1, I2, I3, I4f, I4s, I4n, I5f, I5s, I5n, I8fs, I8fn, I8ns, p):
    mu, af, bf, an, bn = p

    I1 = (I1 - 3) ** 2
    I2 = (I2 - 3) ** 2
    I4n = (I4n - 1) ** 2
    I4f = (I4f - 1) ** 2
    I4s = (I4s - 1) ** 2
    I5f = (I5f - 1) ** 2
    I5s = (I5s - 1) ** 2
    I5n = (I5n - 1) ** 2
    I8fs = (I8fs) ** 2
    I8ns = (I8ns) ** 2
    I8fn = (I8fn) ** 2

    return mu/2 * I2 + af/(2*bf) * (sp.exp(bf*I4f)-1) + an/(2*bn)*(sp.exp(bn*I4n)-1)

def psi_HO(I1, I2, I3, I4f, I4s, I4n, I5f, I5s, I5n, I8fs, I8fn, I8ns, p):
    p1, p2, p3, p4, p5, p6, p7, p8 = p

    psi = p1 / (2 * p2) * sp.exp(p2 * (I1 - 3)) + p3 / (2 * p4) * (sp.exp(p4 * (I4f - 1) ** 2) - 1) \
          + p5 / (2 * p6) * (sp.exp(p6 * (I4s - 1) ** 2) - 1) \
          + p7 / (2 * p8) * (sp.exp(p8 * I8fs ** 2) - 1)

    return psi

def psi_CL(Eff, Efs, Efn, Esf, Ess, Esn, Enf, Ens, Enn, p):
    a, bff, bfn, bfs, bnn, bns, bss = p

    Q = bff*Eff**2 + 2*bfn*((Efn+Enf)/2)**2 + 2*bfs*((Efs+Esf)/2)**2 \
        + bnn*Enn**2 + 2*bns*((Ens+Esn)/2)**2 + bss*Ess**2

    return 0.5 * a * (sp.exp(Q) - 1)

def psi_SFL(Eff, Efs, Efn, Esf, Ess, Esn, Enf, Ens, Enn, p):
    aff, afn, afs, ann, ans, ass, bff, bfn, bfs, bnn, bns, bss = p

    res = 0.5 * aff * (sp.exp(bff*Eff**2)-1) + 0.5 * afn * (sp.exp(bfn*(0.5*(Efn+Enf))**2)-1) \
           + 0.5 * afs * (sp.exp(bfs*(0.5*(Efs+Esf))**2)-1) \
           + 0.5 * ann * (sp.exp(bnn*Enn**2)-1) + 0.5 * ans * (sp.exp(bns*(0.5*(Ens+Esn))**2)-1) \
           + 0.5 * ass * (sp.exp(bss*Ess**2)-1)

    return res

def psi_PZL(Eff, Efs, Efn, Esf, Ess, Esn, Enf, Ens, Enn, p):
    aff, afn, afs, ann, ans, ass, bff, bfn, bfs, bnn, bns, bss = p

    res = bff*Eff**2/(aff-abs(Eff))**2 + bfn*(0.5*(Efn+Enf))**2/(afn - abs(0.5*(Efn+Enf)))**2 \
          + bfs*(0.5*(Efs+Esf))**2/(afs - abs(0.5*(Efs+Esf)))**2 + bnn*Enn**2/(ann-abs(Enn))**2 \
          + bns*(0.5*(Ens+Esn))**2/(ans - abs(0.5*(Ens+Esn)))**2 + bss*Ess**2/(ass-abs(Ess))**2

    return res

def psi_GL(Eff, Efs, Efn, Esf, Ess, Esn, Enf, Ens, Enn, p):
    C, bf, bt, bfs = p

    Q = bf * Eff**2 + bt*(Ess**2 + Enn**2 + Esn**2 + Ens**2) + bfs*(Efs**2 + Esf**2 + Efn**2 + Enf**2)
    return C/2 * (sp.exp(Q) - 1)


########################################################################################################################
# Perform the inverse tests
########################################################################################################################

psi_lst = [psi_CH1, psi_CH2, psi_MA, psi_HO, psi_CL] #, psi_SFL, psi_PZL]
num_params = [3, 4, 5, 8, 7] #, 12, 12]
psi_names = ['CH1', 'CH2', 'MA', 'HO', 'CL', 'SFL', 'PZL']
param_names = {'CH1':['$p_{1}$', '$p_{2}$', '$p_{3}$'],
               'CH2':['$p_{1}$', '$p_{2}$', '$p_{3}$', '$p_{4}$'],
               'MA':['$\mu$', '$a_{f}$', '$b_{f}$', '$a_{n}$', '$b_{n}$'],
               'HO':['$a$', '$b$', '$a_{f}$', '$b_{f}$', '$a_{n}$', '$b_{n}$', '$a_{fs}$', '$b_{fs}$'],
               'CL':['$a$', '$b_{ff}$', '$b_{fn}$', '$b_{fs}$', '$b_{nn}$', '$b_{ns}$', '$b_{ss}$'],
               'SFL':['$a_{ff}$', '$a_{fn}$', '$a_{fs}$', '$a_{nn}$', '$a_{ns}$', '$a_{ss}$', '$b_{ff}$', '$b_{fn}$',
                    '$b_{fs}$', '$b_{nn}$', '$b_{ns}$', '$b_{ss}$'],
               'PZL':['$a_{ff}$', '$a_{fn}$', '$a_{fs}$', '$a_{nn}$', '$a_{ns}$', '$a_{ss}$', '$k_{ff}$', '$k_{fn}$',
                    '$k_{fs}$', '$k_{nn}$', '$k_{ns}$', '$k_{ss}$'],
               'GL':['$C$', '$b_{f}$', '$b_{t}$', '$b_{fs}$']}

data_lst = [ShearDokos(), ShearSommer()]
data_names = ['Shear Dokos', 'Shear Sommer']
#
#
data_save_params = {'variable':[], 'value':[], 'model':[], 'dataset':[]}
data_save_gof = {'GoF':[], 'model':[], 'dataset':[], 'nfev':[]}
data_save_sRSS = {'model':[], 'dataset':[], 'sRSS':[]}
os.makedirs('plot_data', exist_ok=True)

for d, data in enumerate(data_lst):
    data_name = data_names[d]
    print(data_name)

    for i in range(len(psi_lst)):
        sRSS_max = 0
        sRSS_min = 1
        psi = psi_lst[i]
        p = num_params[i]
        psi_name = psi_names[i]
        print(psi_name)
        if psi == psi_CH1 or psi == psi_CH2 or psi == psi_HO or psi == psi_MA:
            var = 'I'
        else:
            var = 'E'

        # define the initial parameter sets for the optimization
        num_samples = 100
        lhs_samples = lhs(p, samples=num_samples)
        param_ranges = (0, 100)
        initial_points = []
        for l in range(num_samples):
            scaled_params = [param_ranges[0] + lhs_samples[l][j] * (param_ranges[1] - param_ranges[0])
                             for j in range(p)]
            initial_points.append(scaled_params)

        data_path = '../../'

        # perform the optimization from each initial parameter set
        trial = 0
        for initial_guess in initial_points:
            try:
                SSE, Y_a, res, nfev, p_best = data.fit_psi_to_data(psi, p, data_path, plot=False, var=var, init_vals=initial_guess)
                sRSS = (SSE / np.sum((np.array(Y_a) - np.mean(Y_a)) ** 2))
                data_save_gof['GoF'].append(sRSS)
                data_save_gof['model'].append(psi_name)
                data_save_gof['dataset'].append(data_name)
                data_save_gof['nfev'].append(nfev)

                for param_ind, param_name in enumerate(param_names[psi_name]):
                    data_save_params['model'].append(psi_name)
                    data_save_params['dataset'].append(data_name)
                    data_save_params['variable'].append(param_name)
                    data_save_params['value'].append(p_best[param_ind])

                if sRSS > sRSS_max:
                    sRSS_max = sRSS
                    SSE, Y_a, res, nfev, p_best = data.fit_psi_to_data(psi, p, data_path, plot=True, var=var,
                                                                       save_to='plot_data/worst/', psi_name=psi_names[i],
                                                                       init_vals=p_best,
                                                                       not_vary=range(len(p_best)))
                if sRSS < sRSS_min:
                    sRSS_min = sRSS
                    SSE, Y_a, res, nfev, p_best = data.fit_psi_to_data(psi, p, data_path, plot=True, var=var,
                                                                       save_to='plot_data/best/', psi_name=psi_names[i],
                                                                       init_vals=p_best,
                                                                       not_vary=range(len(p_best)))

            # if the optimization does not converge try another set of parameters
            except Exception as e:
                if trial < 5:
                    lhs_samples = lhs(p, samples=num_samples)
                    scaled_params = [param_ranges[0] + lhs_samples[0][j] * (param_ranges[1] - param_ranges[0])
                                     for j in range(p)]
                    initial_points.append(scaled_params)
                    trial += 1


        data_save_sRSS['sRSS'].append(sRSS_max)
        data_save_sRSS['model'].append(psi_name)
        data_save_sRSS['dataset'].append(data_name)


params_dist_df = pd.DataFrame(data_save_params)
gof_dists_df = pd.DataFrame(data_save_gof)
sRSS_df = pd.DataFrame(data_save_sRSS)

params_dist_df.to_csv('plot_data/params_dist_df.csv')
gof_dists_df.to_csv('plot_data/gof_dists_df.csv')
sRSS_df.to_csv('plot_data/sRSS_df.csv')

