import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import csv
from scipy import optimize
from lmfit import minimize, Parameters, report_fit
from CHESRA.continuum_mechanics import *
import time
import pandas as pd


class BiaxialNovak:
    def __init__(self):
        self.exp_type = 'biaxial'
        self.name = 'Biaxial Novak'

    def add_parameters(self, fit_params, num_params, i, init_vals):
        """
        Adds a parameter to the optimization parameter list for every parameter in the strain energy function and for every shear dataset i
        :param fit_params: parameters to be optimized (type: lmfit.Parameters())
        :param num_params: number of parameters in the strain energy function
        :param i: index of shear dataset
        """
        # add a parameter in the list for every parameter in the strain energy function
        for n in range(num_params):
            if len(init_vals) == 1:
                fit_params.add('p%i_%i' % ((n + 1), (i + 1)), value=init_vals[0], min=0.0, max=np.inf)
            else:
                fit_params.add('p%i_%i' % ((n + 1), (i + 1)), value=init_vals[n], min=0.0, max=np.inf)

            # restrict the parameters for each dataset to be equal (i.e. equal to the first set)
            if i > 0:
                fit_params['p%i_%i' % ((n + 1), (i + 1))].expr = 'p%i_1' % ((n + 1))

    def get_data(self, shear, ratio, path):
        """
        Extracts the experimental data biaxial data from novak et al.
        :param shear: type of shear, i.e. 'ff' or 'ss'
        :return: amount of stretch lambda and stress sigma as a list
        """
        filename = 'CHESRA/data/novak/'
        name = 'outerLV_' + shear
        lam_data = []
        sigma_data = []
        with open(path + filename + '%s_%s.csv' % (name, ratio)) as csvfile:
            reader = csv.DictReader(csvfile)
            for line in reader:
                lam_data.append(float(line['x']))
                sigma_data.append(float(line['y']))
        return np.array(lam_data), sigma_data

    def optimize_psi(self, psi, num_params, path, shear_ratio=[('ff', 1.15), ('ff', 1), ('ss', 1.10), ('ss', 1.15),
                                                               ('ss', 1.20), ('ss', 1)], var='I', init_vals=[1]):
        """
        Run the optimization to find the best parameters for the given psi.
        :param psi: strain energy function (callable function with arguments I1, I2, I3, I4f, I4s, I4n, I5f, I5s, I5n, I8fs, I8fn, I8ns, p)
        :param num_params: number of parameters to be optimized in psi (i.e. len(p))
        :param shear_lst: list of shear types under consideration
        :return: list containing optimized parameters, reduced chi squared
        """
        lam_f_data_all = {}  # dict that stores experimental lam_f values for each shear type
        lam_s_data_all = {}  # dict that stores experimental lam_s values for each shear type
        sigma_data_all = {}  # dict that stores measured S values for each shear type

        # initialize fit parameters for lmfit
        fit_params = Parameters()

        # calculate the symbolic stress tensor function from psi for the given shear type
        if var == 'I':
            sigma_function = calculate_stress_tensor_function_diagonal(psi, num_params)
        else:
            sigma_function = calculate_stress_tensor_function_diagonal_E(psi, num_params)
        i = 0
        Y_a = []  # stores the experimental data to calculate the COD
        for (shear, ratio) in shear_ratio:
            # get the experimental data
            lam_data, sigma_data = self.get_data(shear, ratio, path)
            Y_a.extend(sigma_data)
            if shear == 'ff':
                lam_f_data = lam_data
                if ratio == 1:
                    lam_s_data = lam_data
                else:
                    lam_s_data = [ratio] * len(lam_data)
            elif shear == 'ss':
                lam_s_data = lam_data
                if ratio == 1:
                    lam_f_data = lam_data
                else:
                    lam_f_data = [ratio] * len(lam_data)

            lam_f_data_all[(shear, ratio)] = lam_f_data
            lam_s_data_all[(shear, ratio)] = lam_s_data
            sigma_data_all[(shear, ratio)] = sigma_data

            # add the parameters to the paramters list to be optimized
            self.add_parameters(fit_params, num_params, i, init_vals)

            i += 1

        # run the global fit to all the data sets and measure the time needed
        start_time = time.time()
        result = minimize(self.objective, fit_params,
                          args=(lam_f_data_all, lam_s_data_all, sigma_function, num_params, sigma_data_all, var))
        end_time = time.time()
        # print("Needed %.1f s for optimization." % (end_time - start_time))

        # return the best parameters (only the first [num_params] corresponding to the first dataset since the other
        # parameters are the same anyways)
        p_best = list(result.params.valuesdict().values())[:num_params]

        # return reduced chi squared (quantifies quality of the fit)
        SSE = result.chisqr
        return p_best, SSE, Y_a, result.residual, result.nfev

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
        # make residual per data set
        for (shear, ratio) in sigma_data_all.keys():
            for j in range(len(sigma_data_all[(shear, ratio)])):
                resid.append((sigma_data_all[(shear, ratio)][j] - self.eval_sigma(shear, lam_f_data_all[
                    (shear, ratio)][j], lam_s_data_all[(shear, ratio)][j], params, sigma_function, num_params, i, var)))
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
        params_str = r'\textbf{Novak}'
        params_line2 = r'(biaxial)'
        for i in range(1, num_params):
            number = round(params[i], digits)
            if number == 0.0:
                number = '%.1E' % params[i]
                number = number[:3] + r'\times 10^{-%s}' % number[-1]
            else:
                number = str(number)
            params_str = params_str + ' & ' + r'\multirow{2}{*}{$ ' + number + r' $}'
            params_line2 = params_line2 + ' &'
        params_str = params_str + r' \\'
        params_line2 = params_line2 + r' \\'
        print(params_str)
        print(params_line2)
        print(r'\hline')

    def fit_data(self, ind, num_params, data_path, plot=False, save_to='./', fs=18, axs = 0, only_data=False, with_HO=True, digits=3,
                 init_vals=[1]):

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
        shear_ratio = [('ff', 1.15), ('ss', 1.20), ('ss', 1.15), ('ss', 1.10)]

        try:
            # do the fit
            p_best, SSE_fit, Y_a, res, nfev = self.optimize_psi(psi, num_params, data_path, shear_ratio=shear_ratio, init_vals=init_vals)

            if plot is True:

                lam_lst = np.linspace(1.0, 1.36, 20)  # lambda values to be plotted

                # calculate the symbolic stress tensor function from psi
                sigma_function = calculate_stress_tensor_function_diagonal(psi, num_params)


                data_save = {'x': lam_lst}
                for shear, ratio in shear_ratio:

                    sigma_lst = []
                    for lam in lam_lst:
                        if shear == 'ff':
                            lam_f = lam
                            if ratio == 1:
                                lam_s = lam
                            else:
                                lam_s = ratio
                        elif shear == 'ss':
                            lam_s = lam
                            if ratio == 1:
                                lam_f = lam
                            else:
                                lam_f = ratio

                        # evaluate sigma for the given shear, gamma and fit parameters
                        sigma_lst.append(
                            evaluate_stress_tensor_function_diagonal(shear, lam_f, lam_s, p_best, sigma_function))

                    data_save["%s, %.2f" % (shear, ratio)] = sigma_lst






                df = pd.DataFrame(data=data_save)
                df.to_csv(save_to + 'data_biaxial_novak.csv')




        except Exception as e:

            if plot is True:
                print(e)
            SSE_fit = np.inf
            Y_a = [1]
            res = [1]
            nfev = np.inf
            p_best = [1] * num_params

        return SSE_fit, Y_a, res, nfev, p_best


    def fit_psi_to_data(self, psi, num_params, data_path, plot=False, save_to='./', fs=18, only_data=False, var='I', init_vals=[1]):

        # define strain energy function to be optimized and number of parameters in the function
        num_params = num_params  # number of parameters to be optimized, i.e. the length of the parameter list p from psi

        # define the types of shear for which the function should be fitted (default is all but maybe at the beginning we only want to do one type)
        shear_ratio = [('ff', 1.15), ('ss', 1.20), ('ss', 1.15), ('ss', 1.10)]


        # do the fit
        p_best, SSE_fit, Y_a, res, nfev = self.optimize_psi(psi, num_params, data_path, shear_ratio=shear_ratio, var=var, init_vals=init_vals)

        if plot is True:

            # define fixed color for each shear type
            colors = {('ff', 1.15): 'cadetblue', ('ff', 1): 'royalblue', ('ss', 1.10): 'crimson',
                        ('ss', 1.15): 'cadetblue',
                        ('ss', 1.20): 'royalblue', ('ss', 1): 'royalblue'}
            markers = {('ff', 1.15): 's', ('ff', 1): 'o', ('ss', 1.10): 'd', ('ss', 1.15): 's',
                        ('ss', 1.20): 'o', ('ss', 1): 'o'}

            lam_lst = np.linspace(1.0, 1.36, 20)  # lambda values to be plotted

            # calculate the symbolic stress tensor function from psi
            sigma_function = calculate_stress_tensor_function_diagonal_E(psi, num_params)

            fig, axs = plt.subplots(1, 2, sharey=True)

            for shear, ratio in shear_ratio:

                sigma_lst = []
                sigma_lst_paper = []
                for lam in lam_lst:
                    if shear == 'ff':
                        lam_f = lam
                        if ratio == 1:
                            lam_s = lam
                        else:
                            lam_s = ratio
                    elif shear == 'ss':
                        lam_s = lam
                        if ratio == 1:
                            lam_f = lam
                        else:
                            lam_f = ratio

                    # evaluate sigma for the given shear, gamma and fit parameters
                    sigma_lst.append(
                        evaluate_stress_tensor_function_diagonal_E(shear, lam_f, lam_s, p_best, sigma_function))

                if shear == 'ff':
                    s = 0
                    other = 'ss'
                else:
                    s = 1
                    other = 'ff'
                if only_data is False:
                    axs[s].plot(lam_lst, sigma_lst, color=colors[(shear, ratio)])

                # plot the experimental data
                lam_data, sigma_data = self.get_data(shear, ratio, path=data_path)
                if ratio == 1:
                    axs[s].plot(lam_data, sigma_data, marker=markers[(shear, ratio)], ls='', fillstyle='none',
                                color=colors[(shear, ratio)],
                                label='$\lambda_{%s} = \lambda_{%s}$' % (other, shear))
                else:
                    axs[s].plot(lam_data, sigma_data, marker=markers[(shear, ratio)], ls='', fillstyle='none',
                                color=colors[(shear, ratio)],
                                label='$\lambda_{%s} = %s$' % (other, ratio))
                axs[s].set_ylabel("stress $\sigma_{%s}$ [kPa]" % shear, fontsize=fs)
                axs[s].set_xlabel("extension $\lambda_{%s}$" % shear, fontsize=fs)
                axs[s].tick_params(axis='both', which='major', labelsize=fs)

                axs[s].legend(loc="upper left", frameon=False, fontsize=fs, handletextpad=0)
            axs[0].set_ylim((0, 18))
            axs[0].set_xlim((1, 1.36))
            axs[1].set_ylim((0, 18))
            axs[1].set_xlim((1, 1.36))

            for ax in axs:
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)



            plt.tight_layout()
            plt.show()
            fig.savefig(save_to + 'plot_biaxial_novak.png', dpi=300)
            fig.savefig(save_to + 'plot_biaxial_novak.pdf', dpi=300)
            fig.savefig(save_to + 'plot_biaxial_novak.svg', dpi=300)



        return SSE_fit, Y_a, res, nfev, p_best