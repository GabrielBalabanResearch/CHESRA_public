import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import csv
from scipy import optimize
from lmfit import minimize, Parameters, report_fit
from continuum_mechanics import *
import time
import pandas as pd


class BiaxialYin:
    def __init__(self):
        self.exp_type = 'biaxial'
        self.name = 'Biaxial Yin'

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
        Extracts the experimental data in figure 7 for the given shear type.
        :param shear: type of shear, i.e. 'ns', 'nf', 'sn', 'sf', 'fn' or 'fs'
        :return: amount of shear gamma and shear stress sigma as a list
        """
        filename = 'Data/yin/holzapfel_ogden_8'
        if shear == 'ff':
            name = 'a'
        elif shear == 'ss':
            name = 'b'

        Eii_data = []
        sigma_data = []
        with open(path + filename + '%s_%s.csv' % (name, ratio)) as csvfile:
            reader = csv.DictReader(csvfile)
            for line in reader:
                Eii_data.append(float(line['x']))
                sigma_data.append(float(line['y']))
        return np.array(Eii_data), sigma_data

    def eval_sigma(self, shear, lam_f, lam_s, params, S_function, num_params, i, var):
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
            return evaluate_S_tensor_function(shear, lam_f, lam_s, p, S_function)
        elif var == 'E':
            return evaluate_S_tensor_function_E(shear, lam_f, lam_s, p, S_function)

    def objective(self, params, lam_f_data_all, lam_s_data_all, S_function, num_params, S_data_all, var):
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
        for (shear, ratio) in S_data_all.keys():
            for j in range(len(S_data_all[(shear, ratio)])):
                resid.append(
                    (S_data_all[(shear, ratio)][j] - self.eval_sigma(shear, lam_f_data_all[(shear, ratio)][j],
                                                                        lam_s_data_all[(shear, ratio)][j], params,
                                                                        S_function, num_params, i, var)))
            i += 1
        return np.array(resid)

    def optimize_psi(self, psi, num_params, path, shear_lst=['ff', 'ss'], ratio_lst=['circ', 'squa', 'tria'], var='I', init_vals=[1]):
        """
        Run the optimization to find the best parameters for the given psi.
        :param psi: strain energy function (callable function with arguments I1, I2, I3, I4f, I4s, I4n, I5f, I5s, I5n, I8fs, I8fn, I8ns, p)
        :param num_params: number of parameters to be optimized in psi (i.e. len(p))
        :param shear_lst: list of shear types under consideration
        :return: list containing optimized parameters, reduced chi squared
        """
        ratio_to_value = {'circ': 0.48, 'squa': 1.02, 'tria': 2.05}
        lam_f_data_all = {}  # dict that stores experimental lam_f values for each shear type
        lam_s_data_all = {}  # dict that stores experimental lam_s values for each shear type
        S_data_all = {}  # dict that stores measured S values for each shear type

        # initialize fit parameters for lmfit
        fit_params = Parameters()

        # calculate the symbolic stress tensor function from psi for the given shear type
        if var == 'I':
            S_function = calculate_S_tensor_function(psi, num_params)
        elif var == 'E':
            S_function = calculate_S_tensor_function_E(psi, num_params)
        i = 0
        Y_a = []  # stores the experimental data to calculate the COD
        for shear in shear_lst:
            for ratio in ratio_lst:
                # get the experimental data
                Eii_data, S_data = self.get_data(shear, ratio, path)
                Y_a.extend(S_data)
                r = ratio_to_value[ratio]
                if shear == 'ff':
                    Eff_data = Eii_data
                    Ess_data = Eff_data / r
                elif shear == 'ss':
                    Ess_data = Eii_data
                    Eff_data = Ess_data * r
                lam_f_data = np.sqrt(2 * Eff_data + 1)
                lam_s_data = np.sqrt(2 * Ess_data + 1)
                lam_f_data_all[(shear, ratio)] = lam_f_data
                lam_s_data_all[(shear, ratio)] = lam_s_data
                S_data_all[(shear, ratio)] = S_data

                # add the parameters to the paramters list to be optimized
                self.add_parameters(fit_params, num_params, i, init_vals)

                i += 1

        # run the global fit to all the data sets and measure the time needed
        start_time = time.time()
        result = minimize(self.objective, fit_params,
                          args=(lam_f_data_all, lam_s_data_all, S_function, num_params, S_data_all, var))
        end_time = time.time()
        # print("Needed %.1f s for optimization." % (end_time - start_time))

        # return the best parameters (only the first [num_params] corresponding to the first dataset since the other
        # parameters are the same anyways)
        p_best = list(result.params.valuesdict().values())[:num_params]

        # return reduced chi squared (quantifies quality of the fit)
        SSE = result.chisqr
        return p_best, SSE, Y_a, result.residual, result.nfev

    def print_params(self, params, num_params, digits):
        params_str = r'\textbf{Yin}'
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

    def fit_data(self, ind, num_params, data_path, plot=False, save_to='./', fs=18, only_data=False, with_HO=True, digits=3,
                 init_vals=[1]):
        """ Takes in a function found by the GA ane performs the following steps: finds the best parameters, calcualtes the stress tensor function,
            evaluates the stress tensor function at each x-value from a given experiental dataset, and calculates the SSE.
        Args:
            ind: An given function found by the GA
            num_params: number of parameters. This is typically 8.
            biax_shears: shear types you would like to include in the fit.
            data_path: data path of the experimental data.
            plot: would you like to plot this data?
            save to: where would you like to save this data to?
        """

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

        try:
            # define the types of shear for which the function should be fitted (default is all but maybe at the beginning we only want to do one type)
            shear_lst = ['ff', 'ss']
            ratio_lst = ['circ', 'squa', 'tria']
            ratio_to_value = {'circ': 0.48, 'squa': 1.02, 'tria': 2.05}

            # do the fit
            p_best, SSE_fit, Y_a, res, nfev = self.optimize_psi(psi, num_params, path=data_path, shear_lst=shear_lst,
                                                         ratio_lst=ratio_lst, init_vals=init_vals)

            if plot is True:

                # calculate the symbolic stress tensor function from psi
                S_function = calculate_S_tensor_function(psi, num_params)


                Eii_lst = np.linspace(0.01, 0.2, 50)
                data_save = {'x':Eii_lst}
                for s in range(len(shear_lst)):
                    shear = shear_lst[s]
                    for ratio in ratio_lst:
                        S_lst = []

                        for Eii in Eii_lst:
                            r = ratio_to_value[ratio]
                            if shear == 'ff':
                                Eff = Eii
                                Ess = Eff / r
                            elif shear == 'ss':
                                Ess = Eii
                                Eff = Ess * r
                            lam_f = np.sqrt(2 * Eff + 1)
                            lam_s = np.sqrt(2 * Ess + 1)

                            # evaluate sigma for the given shear, gamma and fit parameters
                            S_lst.append(evaluate_S_tensor_function(shear, lam_f, lam_s, p_best, S_function))

                        # plot the results from the function
                        if only_data is False:
                            data_save['%s, %s' % (shear, ratio_to_value[ratio])] = S_lst


                df = pd.DataFrame(data=data_save)
                df.to_csv(save_to + 'data_biaxial_yin.csv')




        except Exception as e:

            if plot is True:
                print(e)
            SSE_fit = np.inf
            Y_a = [1]
            res = [1]
            nfev = np.inf

        return SSE_fit, Y_a, res, nfev, p_best


    def fit_psi_to_data(self, psi, num_params, data_path, plot=False, save_to='./', fs=18, only_data=False, var='E', init_vals=[1]):
        """ Takes in a given function performs the following steps: finds the best parameters, calcualtes the stress tensor function,
            evaluates the stress tensor function at each x-value from a given experiental dataset, and calculates the SSE.
        Args:
            ind: An given function found by the GA
            num_params: number of parameters. This is typically 8.
            biax_shears: shear types you would like to include in the fit.
            data_path: data path of the experimental data.
            plot: would you like to plot this data?
            save to: where would you like to save this data to?
        """

        # define the types of shear for which the function should be fitted (default is all but maybe at the beginning we only want to do one type)
        shear_lst = ['ff', 'ss']
        ratio_lst = ['circ', 'squa', 'tria']
        ratio_to_value = {'circ': 0.48, 'squa': 1.02, 'tria': 2.05}
        colors = {'circ': 'royalblue', 'squa': 'cadetblue', 'tria': 'crimson'}
        markers = {'circ': 'o', 'squa': 's', 'tria': 'v'}

        # do the fit
        p_best, SSE_fit, Y_a, res, nfev = self.optimize_psi(psi, num_params, path=data_path, shear_lst=shear_lst,
                                                     ratio_lst=ratio_lst, var=var, init_vals=init_vals)

        if plot is True:

            # calculate the symbolic stress tensor function from psi
            if var == 'E':
                S_function = calculate_S_tensor_function_E(psi, num_params)
            else:
                S_function = calculate_S_tensor_function(psi, num_params)

            # do the plot
            fig, axs = plt.subplots(1, 2)

            for s in range(len(shear_lst)):
                shear = shear_lst[s]
                for ratio in ratio_lst:
                    S_lst = []
                    S_lst_paper = []
                    Eii_data, S_data = self.get_data(shear, ratio, path=data_path)
                    Eii_lst = np.linspace(0.01, 0.2, 50)
                    for Eii in Eii_lst:
                        r = ratio_to_value[ratio]
                        if shear == 'ff':
                            Eff = Eii
                            Ess = Eff / r
                        elif shear == 'ss':
                            Ess = Eii
                            Eff = Ess * r
                        lam_f = np.sqrt(2 * Eff + 1)
                        lam_s = np.sqrt(2 * Ess + 1)

                        # evaluate sigma for the given shear, gamma and fit parameters
                        if var == 'E':
                            S_lst.append(evaluate_S_tensor_function_E(shear, lam_f, lam_s, p_best, S_function))
                        else:
                            S_lst.append(evaluate_S_tensor_function(shear, lam_f, lam_s, p_best, S_function))

                    # plot the results from the function
                    if only_data is False:
                        if shear == 'ff' and ratio == 'circ':
                            axs[s].plot(Eii_lst[:25], S_lst[:25], color=colors[ratio])

                        else:
                            axs[s].plot(Eii_lst, S_lst, color=colors[ratio])


                    # plot the experimental data
                    Eii_data, S_data = self.get_data(shear, ratio, path=data_path)
                    axs[s].plot(Eii_data, S_data, marker=markers[ratio], ls='', fillstyle='none',
                                color=colors[ratio], label='r = %s' % ratio_to_value[ratio])
                    axs[s].set_ylabel("stress $S_{%s}$ [kPa]" % shear, fontsize=fs)
                    axs[s].set_xlabel("strain $E_{%s}$" % shear, fontsize=fs)
                    # axs[s].set_ylabel("stress [kPa]", fontsize=fs)
                    # axs[s].set_xlabel("strain", fontsize=fs)

            axs[0].legend(loc="upper left", frameon=False, fontsize=fs, handletextpad=0)
            axs[0].set_ylim((0, 18))
            axs[0].set_yticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
            axs[0].set_xlim((0, 0.2))
            axs[1].set_ylim((0, 12))
            axs[1].set_xlim((0, 0.2))
            axs[0].tick_params(axis='both', which='major', labelsize=fs)
            axs[1].tick_params(axis='both', which='major', labelsize=fs)

            for ax in axs:
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)

            plt.tight_layout()
            plt.show()
            fig.savefig(save_to + 'plot_biaxial.png', dpi=300)
            fig.savefig(save_to + 'plot_biaxial.pdf', dpi=300)
            fig.savefig(save_to + 'plot_biaxial.svg', dpi=300)



        return SSE_fit, Y_a, res, nfev, p_best
