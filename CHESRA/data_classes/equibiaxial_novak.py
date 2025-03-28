import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import csv
from scipy import optimize
from lmfit import minimize, Parameters, report_fit
from CHESRA.continuum_mechanics import *
import time
import pandas as pd


class EquibiaxialNovak:
    def __init__(self):
        self.exp_type = 'biaxial'
        self.name = 'Equibiaxial Novak'

    def add_parameters(self, fit_params, num_params, i, init_vals):
        """
        Adds a parameter to the optimization parameter list for every parameter in the strain energy function and for every shear dataset i
        :param fit_params: parameters to be optimized (type: lmfit.Parameters())
        :param num_params: number of parameters in the strain energy function
        :param i: index of shear dataset
        """
        # add a parameter in the list for every parameter in the strain energy function
        # mins = [0, 0.001, 0.007, 1, 1, 1, 1, 1, 1]
        # maxs = [10, 0.003, 0.05, 8, 10, 10, 8, 10, 10]
        for n in range(num_params):
            if len(init_vals) == 1:
                fit_params.add('p%i_%i' % ((n + 1), (i + 1)), value=init_vals[0], min=1e-3, max=np.inf)
            else:
                fit_params.add('p%i_%i' % ((n + 1), (i + 1)), value=init_vals[n], min=0.0, max=np.inf)

            # restrict the parameters for each dataset to be equal (i.e. equal to the first set)
            if i > 0:
                fit_params['p%i_%i' % ((n + 1), (i + 1))].expr = 'p%i_1' % ((n + 1))

    def get_data(self, shear, loc, num, path):
        """
        Extracts the experimental data biaxial data from novak et al.
        :param shear: type of shear, i.e. 'ff' or 'ss'
        :return: amount of stretch lambda and stress sigma as a list
        """
        filename = 'CHESRA/data/novak/'
        name = loc + '_' + shear
        lam_data = []
        sigma_data = []
        with open(path + filename + '%s_1_%s.csv' % (name, num)) as csvfile:
            reader = csv.DictReader(csvfile)
            for line in reader:
                lam_data.append(float(line['x']))
                sigma_data.append(float(line['y']))
        return np.array(lam_data), sigma_data

    def optimize_psi(self, psi, num_params, path, locs, var='I', init_vals=[1]):
        """
        Run the optimization to find the best parameters for the given psi.
        :param psi: strain energy function (callable function with arguments I1, I2, I3, I4f, I4s, I4n, I5f, I5s, I5n, I8fs, I8fn, I8ns, p)
        :param num_params: number of parameters to be optimized in psi (i.e. len(p))
        :param shear_lst: list of shear types under consideration
        :return: list containing optimized parameters, reduced chi squared
        """

        # calculate the symbolic stress tensor function from psi for the given shear type
        if var == 'I':
            sigma_function = calculate_stress_tensor_function_diagonal(psi, num_params)
        elif var == 'E':
            sigma_function = calculate_stress_tensor_function_diagonal_E(psi, num_params)

        p_best_dict = {}
        SSE_all = 0
        Y_a = []  # stores the experimental data to calculate the
        res_all = []
        nfev_dict = {}

        for loc in locs:
            for num in range(1, 3):
                # initialize fit parameters for lmfit
                fit_params = Parameters()

                lam_f_data_all = {}
                lam_s_data_all = {}
                sigma_data_all = {}
                for s, shear in enumerate(['ff', 'ss']):
                    # get the experimental data
                    lam_data, sigma_data = self.get_data(shear, loc, num, path)
                    Y_a.extend(sigma_data)
                    lam_f_data_all[shear] = lam_data
                    lam_s_data_all[shear] = lam_data
                    sigma_data_all[shear] = sigma_data

                    # add the parameters to the paramters list to be optimized
                    self.add_parameters(fit_params, num_params, s, init_vals)

                # run the global fit to all the data sets and measure the time needed
                start_time = time.time()
                result = minimize(self.objective, fit_params,
                                  args=(
                                  lam_f_data_all, lam_s_data_all, sigma_function, num_params, sigma_data_all, var))
                end_time = time.time()
                # print("Needed %.1f s for optimization." % (end_time - start_time))

                # return the best parameters (only the first [num_params] corresponding to the first dataset since the other
                # parameters are the same anyways)
                p_best = list(result.params.valuesdict().values())[:num_params]
                p_best_dict[(loc, num)] = p_best

                # return reduced chi squared (quantifies quality of the fit)
                SSE = result.chisqr
                SSE_all += SSE
                res_all.append(result.residual)
                nfev_dict[(loc, num)] = result.nfev

        return p_best_dict, SSE_all, Y_a, res_all, nfev_dict

    def objective(self, params, lam_f_data_all, lam_s_data_all, sigma_function, num_params, sigma_data_all, var):
        """
        The objective function to be minimized.
        :param params: parameters to be optimized (type: lmfit.Parameters())
        :param gamma_data_all: dict. with a list of the gamma values of the exp. data as values and the shear tyype as key
        :param sigma_function: the symbolic sigma function
        :param num_params: number of parameters in the strain energy function
        :param sigma_data_all: dict. with a list of the sigma values of the exp. data as values and the shear tyype as key
        :return: array containing the difference of each data point to the corresponding fitted function output
        """
        resid = []
        i = 0
        std_fact = 0.3
        # make residual per data point
        for shear in sigma_data_all.keys():
            for j in range(len(sigma_data_all[(shear)])):
                resid.append((sigma_data_all[(shear)][j] - self.eval_sigma(shear, lam_f_data_all[
                    (shear)][j], lam_s_data_all[(shear)][j], params, sigma_function, num_params, i, var)))
            i += 1
        return np.array(resid)

    def eval_sigma(self, shear, lam_f, lam_s, params, sigma_function, num_params, i, var):
        """
        Evaluates the sigma function with the current set of parameters.
        :param shear: type of shear, i.e. 'ns', 'nf', 'sn', 'sf', 'fn' or 'fs'
        :param gamma: amount of shear
        :param params: parameters to be optimized (type: lmfit.Parameters())
        :param sigma_function: the symbolic sigma function
        :param num_params: number of parameters in the strain energy function
        :param i: index of the dataset
        :return: evaluated stress tensor
        """
        p = []
        # extract the current fit parameters
        for n in range(num_params):
            p.append(params['p%i_%i' % (n + 1, i + 1)].value)

        # evaluate the stress tensor with these parameters
        if var == 'I':
            return evaluate_stress_tensor_function_diagonal(shear, lam_f, lam_s, p, sigma_function)
        elif var == 'E':
            return evaluate_stress_tensor_function_diagonal_E(shear, lam_f, lam_s, p, sigma_function)

    def print_params(self, params, num_params, digits):
        novak2_inner1 = params[('innerLV', 1)][1::]
        novak2_inner2 = params[('innerLV', 2)][1::]
        novak2_middle1 = params[('middleLV', 1)][1::]
        novak2_middle2 = params[('middleLV', 2)][1::]
        novak2_outer1 = params[('outerLV', 1)][1::]
        novak2_outer2 = params[('outerLV', 2)][1::]
        novak2_sept1 = params[('septum', 1)][1::]
        novak2_sept2 = params[('septum', 2)][1::]

        params_str = r'\textbf{Novak}'
        params_line2 = r'(equibiax.)'
        names = [(r'\textbf{sub-endo.}', 'sp. 1'), (r'\textbf{sub-endo.}', 'sp. 2'), (r'\textbf{mid-myo.}', 'sp. 1'),
                 (r'\textbf{mid-myo.}', 'sp. 2'), (r'\textbf{sub-epi.}', 'sp. 1'), (r'\textbf{sub-epi.}', 'sp. 2'),
                 (r'\textbf{mid-sept.}', 'sp. 1'), (r'\textbf{mid-sept.}', 'sp. 2')]
        data = [novak2_inner1, novak2_inner2, novak2_middle1, novak2_middle2, novak2_outer1, novak2_outer2,
                novak2_sept1, novak2_sept2]

        print(params_str + ' &'*(num_params-1) + r' \\')
        print(params_line2 + ' &'*(num_params-1) + r' \\')
        for d, n in zip(data, names):
            string = n[0]
            string2 = n[1]
            for i in range(num_params-1):
                number = round(d[i], digits)
                if number == 0.0:
                    number = '%.1E' % d[i]
                    number = number[:3] + r'\times 10^{-%s}' % number[-1]
                else:
                    number = str(number)
                string = string + ' & ' + r'\multirow{2}{*}{$ ' + number + r' $}'
                string2 = string2 + ' &'
            print(string + r' \\')
            print(string2 + r' \\')
            print(r'\hline')

    def fit_data(self, ind, num_params, data_path, plot=False, save_to='./', fs=18, only_data=False, with_HO=True, digits=3, init_vals=[1]):


        def psi_eq(I1, I2, I3, I4f, I4s, I4n, I5f, I5s, I5n, I8fs, I8fn, I8ns, p):
            """
            Strain-energy-function psi as defined in eq. 5.38 in the paper.
            :param I_i: invariants
            :param p: list of parameters [p1, p2, ... , pn]
            :return: psi
            """
            for par in range(1, num_params):
                locals()[f"p{par}"] = p[par]

            # define strain energy function from paper
            psi = eval(ind[0][0])

            return psi

        psi = psi_eq


        # define strain energy function to be optimized and number of parameters in the function
        num_params = num_params  # number of parameters to be optimized, i.e. the length of the parameter list p from psi

        # define the types of shear for which the function should be fitted (default is all but maybe at the beginning we only want to do one type)
        locs = ['innerLV', 'middleLV', 'outerLV', 'septum']

        try:
            # do the fit
            p_best_dict, SSE_fit_all, Y_a, res, nfev = self.optimize_psi(psi, num_params, data_path, locs=locs, init_vals=init_vals)

            if plot is True:
                titles = {'septum': 'septum', 'innerLV': 'sub-endo', 'middleLV': 'mid-myo',
                          'outerLV': 'sub-epi'}


                lam_lst = np.linspace(1.0, 1.36, 20)  # lambda values to be plotted

                # calculate the symbolic stress tensor function from psi
                sigma_function = calculate_stress_tensor_function_diagonal(psi, num_params)

                data_save1 = {'x':lam_lst}
                data_save2 = {'x': lam_lst}
                data_save = {'x': lam_lst}
                for s, shear in enumerate(['ff', 'ss']):
                    for l, loc in enumerate(locs):
                        for num in range(1, 3):
                            sigma_lst = []
                            for lam in lam_lst:
                                lam_f = lam
                                lam_s = lam

                                # evaluate sigma for the given shear, gamma and fit parameters
                                sigma_lst.append(
                                    evaluate_stress_tensor_function_diagonal(shear, lam_f, lam_s,
                                                                             p_best_dict[(loc, num)], sigma_function))

                            data_save[(shear, titles[loc], num)] = sigma_lst
                            if num == 1:
                                data_save1['%s, %s' % (shear, titles[loc])] = sigma_lst
                            elif num == 2:
                                data_save2['%s, %s' % (shear, titles[loc])] = sigma_lst



                df = pd.DataFrame(data=data_save1)
                df.to_csv(save_to + 'data_equibiax_novak1.csv')
                df = pd.DataFrame(data=data_save2)
                df.to_csv(save_to + 'data_equibiax_novak2.csv')
                df = pd.DataFrame(data=data_save)
                df.to_csv(save_to + 'data_equibiax_novak.csv')


        except Exception as e:

            if plot is True:
                print(e)
            SSE_fit_all = np.inf
            Y_a = [1]
            res = [1]
            nfev = np.inf
            p_best_dict = {}

        return SSE_fit_all, Y_a, res, nfev, p_best_dict


    def fit_psi_to_data(self, psi, num_params, data_path, plot=False, save_to='./', fs=18, only_data=False, var='E', init_vals=[1]):

        # define strain energy function to be optimized and number of parameters in the function
        num_params = num_params  # number of parameters to be optimized, i.e. the length of the parameter list p from psi

        # define the types of shear for which the function should be fitted (default is all but maybe at the beginning we only want to do one type)
        locs = ['innerLV', 'middleLV', 'outerLV', 'septum']


        # do the fit
        p_best_dict, SSE_fit_all, Y_a, res, nfev = self.optimize_psi(psi, num_params, data_path, locs=locs, var=var, init_vals=init_vals)

        if plot is True:
            # define fixed color for each shear type
            colors = {1: 'royalblue', 2: 'crimson'}
            markers = {1: 'o', 2: 's'}
            titles = {'septum': 'mid-septum', 'innerLV': 'sub-endocardium', 'middleLV': 'mid-myocardium',
                      'outerLV': 'sub-epicardium'}
            # colors = {'septum': 'crimson', 'innerLV': 'royalblue', 'middleLV': 'cadetblue', 'outerLV': 'orange'}
            # markers = {'septum': 's', 'innerLV': 'o', 'middleLV': 'p', 'outerLV': 'd'}
            # labels = {'septum': 'septum', 'innerLV': 'inner LV', 'middleLV': 'middle LV', 'outerLV': 'outer LV'}

            lam_lst = np.linspace(1.0, 1.36, 20)  # lambda values to be plotted

            # calculate the symbolic stress tensor function from psi
            sigma_function = calculate_stress_tensor_function_diagonal_E(psi, num_params)


            fig = plt.figure(figsize=(6.4, 9.6))  #

            axs = fig.subplots(4, 2, sharex=True, sharey=True)
            data_save = {'x':lam_lst}
            for s, shear in enumerate(['ff', 'ss']):
                n = 1
                for l, loc in enumerate(locs):
                    for num in range(1, 3):
                        sigma_lst = []
                        sigma_lst_paper = []
                        for lam in lam_lst:
                            lam_f = lam
                            lam_s = lam

                            # evaluate sigma for the given shear, gamma and fit parameters
                            sigma_lst.append(
                                evaluate_stress_tensor_function_diagonal_E(shear, lam_f, lam_s,
                                                                         p_best_dict[(loc, num)], sigma_function))
                        if only_data is False:
                            data_save[(shear, loc, num)] = sigma_lst
                            axs[l, s].plot(lam_lst, sigma_lst, color=colors[num])

                        # plot the experimental data
                        lam_data, sigma_data = self.get_data(shear, loc, num, path=data_path)
                        if s == 0:
                            axs[l, s].plot(lam_data, sigma_data, marker=markers[num], ls='', fillstyle='none',
                                           color=colors[num], label=f'sp. {n}')
                        else:
                            axs[l, s].plot(lam_data, sigma_data, marker=markers[num], ls='', fillstyle='none',
                                           color=colors[num])
                        # axs[l, s].set_ylabel("$\sigma_{%s}$ [kPa]" % shear, fontsize=fs)
                        axs[l, s].tick_params(axis='both', which='major', labelsize=fs)
                        axs[l, s].spines['right'].set_visible(False)
                        axs[l, s].spines['top'].set_visible(False)
                        axs[l, s].set_title(titles[loc], fontsize=fs)

                        n += 1
                    # axs[l, 0].text(.05, .95, labels[loc], transform=axs[l, 0].transAxes, ha="left", va="top", fontsize=fs)
                    axs[l, 0].legend(frameon=False, fontsize=fs, handletextpad=0)
                    axs[l, 0].set_ylim((0, 18))
                    axs[l, 0].set_xlim((1, 1.36))
                    axs[l, 1].set_ylim((0, 18))
                    axs[l, 1].set_xlim((1, 1.36))
                    axs[-1, s].set_xlabel("extension $\lambda_{%s}$" % shear, fontsize=fs)
            fig.supylabel("stress $\sigma$ [kPa]", fontsize=fs)

            df = pd.DataFrame(data=data_save)
            df.to_csv(save_to + 'data_equibiax_novak.csv')
            plt.tight_layout()
            plt.show()
            fig.savefig(save_to + 'plot_equibiaxial_novak.png', dpi=300)
            fig.savefig(save_to + 'plot_equibiaxial_novak.pdf', dpi=300)
            fig.savefig(save_to + 'plot_equibiaxial_novak.svg', dpi=300)


        return SSE_fit_all, Y_a, res, nfev, p_best_dict