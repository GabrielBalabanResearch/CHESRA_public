import matplotlib.pyplot as plt
import csv
from lmfit import minimize, Parameters
import numpy as np
import pandas as pd
import sympy as sp
from CHESRA.continuum_mechanics import calculate_stress_tensor_function, evaluate_stress_tensor_function, \
    calculate_stress_tensor_function_E, evaluate_stress_tensor_function_E


class ShearDokos:
    def __init__(self):
        self.exp_type = 'shear'

    def get_data(self, shear, path):
        """
        Extracts the experimental data in figure 7 for the given shear type.
        :param shear: type of shear, i.e. 'ns', 'nf', 'sn', 'sf', 'fn' or 'fs'
        :param path: path reference to the file 'Data/dokos/holzapfel_ogden_7'
        :return: amount of shear gamma and shear stress sigma as a list
        """
        filename = 'CHESRA/data/dokos/holzapfel_ogden_7'
        gamma_data = []
        sigma_data = []
        with open(path + filename + '_%s.csv' % shear) as csvfile:
            reader = csv.DictReader(csvfile)
            for line in reader:
                gamma_data.append(float(line['x']))
                sigma_data.append(float(line['y']))
        return gamma_data, sigma_data

    def eval_sigma(self, shear, gamma, params, sigma_function, num_params, i, var):
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
            return evaluate_stress_tensor_function(shear, gamma, p, sigma_function)
        elif var == 'E':
            return evaluate_stress_tensor_function_E(shear, gamma, p, sigma_function)


    def add_parameters(self, fit_params, num_params, i, init_vals, not_vary=[]):
        """
        Adds a parameter to the optimization parameter list for every parameter in the strain energy function and for every shear dataset i
        :param fit_params: parameters to be optimized (type: lmfit.Parameters())
        :param num_params: number of parameters in the strain energy function
        :param i: index of shear dataset
        :param init_vals: initial parameter values to be used
        :param not_vary: indices of parameters which should not be optimized
        """
        # add a parameter in the list for every parameter in the strain energy function
        for n in range(num_params):
            if n in not_vary:
                vary = False
            else:
                vary = True
            if len(init_vals) == 1:
                fit_params.add('p%i_%i' % ((n + 1), (i + 1)), vary=vary, value=init_vals[0], min=0.0, max=np.inf)
            else:
                fit_params.add('p%i_%i' % ((n + 1), (i + 1)), vary=vary, value=init_vals[n], min=0.0, max=np.inf)

            # restrict the parameters for each dataset to be equal (i.e. equal to the first set)
            if i > 0:
                fit_params['p%i_%i' % ((n + 1), (i + 1))].expr = 'p%i_1' % ((n + 1))

    def objective(self, params, gamma_data_all, sigma_function, num_params, sigma_data_all, var):
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
        for shear in gamma_data_all.keys():
            for j in range(len(sigma_data_all[shear])):
                resid.append(
                    (sigma_data_all[shear][j] - self.eval_sigma(shear, gamma_data_all[shear][j], params, sigma_function,
                                                                num_params, i, var)))
            i += 1
        return np.array(resid)

    def optimize_psi(self, psi, num_params, path='.', not_vary=[], var='I', shear_lst=['fs', 'fn', 'sf', 'sn', 'nf', 'ns'],
                     init_vals=[1]):
        """
        Run the optimization to find the best parameters for the given psi.
        :param psi: strain energy function (callable function with arguments I1, I2, I3, I4f, I4s, I4n, I5f, I5s, I5n, I8fs, I8fn, I8ns, p)
        :param num_params: number of parameters to be optimized in psi (i.e. len(p))
        :param shear_lst: list of shear types under consideration
        :return: list containing optimized parameters, reduced chi squared
        """

        gamma_data_all = {}  # dict that stores experimental gamma values for each shear type
        sigma_data_all = {}  # dict that stores measured sigma values for each shear type

        # initialize fit parameters for lmfit
        fit_params = Parameters()

        # calculate the symbolic stress tensor function from psi for the given shear type
        if var == 'I':
            sigma_function = calculate_stress_tensor_function(psi, num_params)
        elif var == 'E':
            sigma_function = calculate_stress_tensor_function_E(psi, num_params)

        i = 0
        Y_a = []  # stores the experimental data to calculate the COD
        for shear in shear_lst:
            # get the experimental data
            gamma_data, sigma_data = self.get_data(shear, path)
            Y_a.extend(sigma_data)
            gamma_data_all[shear] = gamma_data
            sigma_data_all[shear] = sigma_data

            # add the parameters to the paramters list to be optimized
            self.add_parameters(fit_params, num_params, i, init_vals, not_vary=not_vary)

            i += 1

        # run the global fit to all the data sets and measure the time needed
        result = minimize(self.objective, fit_params,
                          args=(gamma_data_all, sigma_function, num_params, sigma_data_all, var))
        # print("Needed %.1f s for optimization." % (end_time - start_time))

        # return the best parameters (only the first [num_params] corresponding to the first dataset since the other
        # parameters are the same anyways)
        p_best = list(result.params.valuesdict().values())[:num_params]

        # return reduced chi squared (quantifies quality of the fit)
        SSE = result.chisqr
        return p_best, SSE, Y_a, result.residual, result.nfev

    def fit_data(self, ind, num_params, data_path, not_vary=[], plot=False, save_to='./', fs=18, only_data=False,
                 with_HO=True, digits=3, init_vals=[1]):
        """ Takes in a function found by the GA ane performs the following steps: finds the best parameters, calcualtes the stress tensor function,
            evaluates the stress tensor function at each x-value from a given experiental dataset, and calculates the SSE.
        Args:
            ind: An given function found by the GA
            num_params: number of parameters. This is typically 8.
            shears: shear types you would like to include in the fit.
            data_path: data path of the experimental data.
            plot: would you like to plot this data?
            save to: where would you like to save this data to?
        """

        def psi_eq(I1, I2, I3, I4f, I4s, I4n, I5f, I5s, I5n, I8fs, I8fn, I8ns, p):
            """
            Potnetial strain-energy-function psi.
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
            shears = ['fs', 'fn', 'sf', 'sn', 'nf', 'ns']

            ## do the fit
            p_best, SSE_fit, Y_a, res, nfev = self.optimize_psi(psi, num_params, not_vary=not_vary, path=data_path,
                                                                shear_lst=shears, init_vals=init_vals)

            if plot is True:

                ## calculate the symbolic stress tensor function from psi
                sigma_function = calculate_stress_tensor_function(psi, num_params)


                gamma_lst = np.linspace(0.01, 0.5, 50)
                data_save = {'x': gamma_lst}
                for shear in shears:
                    sigma_lst = []
                    for gamma in gamma_lst:
                        # evaluate sigma for the given shear, gamma and fit parameters
                        sigma_lst.append(evaluate_stress_tensor_function(shear, gamma, p_best, sigma_function))
                    data_save[shear] = sigma_lst

                df = pd.DataFrame(data=data_save)
                df.to_csv(save_to + 'data_shear_dokos.csv')


        except Exception as e:
            if plot is True:
                print(e)
            SSE_fit = np.inf
            Y_a = [1]
            res = [1]
            nfev = np.inf
            p_best = [1] * num_params

        return SSE_fit, Y_a, res, nfev, p_best

    def fit_psi_to_data(self, psi, num_params, data_path, not_vary=[], plot=False, save_to='./', fs=18, only_data=False,
                   var='E', init_vals=[1], psi_name=''):
        """ Takes in a given function psi ane performs the following steps: finds the best parameters, calcualtes the stress tensor function,
            evaluates the stress tensor function at each x-value from a given experiental dataset, and calculates the SSE.
        Args:
            psi: An given function.
            num_params: number of parameters.
            data_path: data path of the experimental data.
            not_vary: list of parameters of psi that should not be optimized
            plot: would you like to plot this data?
            save to: where would you like to save this data to?
            fs: fontsize for the plot.
            only_data: should only the experimental data be plotted?
            var: variable in the function (either 'I' or 'E')
            init_vals: initial parameters for the optimization
            psi_name: name of the function psi
        """


        shears = ['fs', 'fn', 'sf', 'sn', 'nf', 'ns']

        ## do the fit
        p_best, SSE_fit, Y_a, res, nfev = self.optimize_psi(psi, num_params, not_vary=not_vary, path=data_path, shear_lst=shears, var=var, init_vals=init_vals)

        if plot is True:

            ## calculate the symbolic stress tensor function from psi
            if var == 'I':
                sigma_function = calculate_stress_tensor_function(psi, num_params)
            else:
                sigma_function = calculate_stress_tensor_function_E(psi, num_params)


            # do the plot
            colors = {'fs': 'blueviolet', 'fn': 'royalblue', 'sf': 'cadetblue', 'sn': 'darkkhaki', 'nf': 'orange',
                        'ns': 'crimson'}
            markers = {'fs': 'o', 'fn': 's', 'sf': 'd', 'sn': 'p', 'nf': 'v',
                        'ns': '*'}

            fig, axs = plt.subplots()

            gamma_lst = np.linspace(0.01, 0.5, 50)
            data_save = {'x':gamma_lst}
            for shear in shears:
                # read in experimental data
                gamma_data, sigma_data = self.get_data(shear, data_path)
                sigma_lst = []

                for gamma in gamma_lst:
                    # evaluate sigma for the given shear, gamma and fit parameters
                    if var == 'I':
                        sigma_lst.append(evaluate_stress_tensor_function(shear, gamma, p_best, sigma_function))
                    else:
                        sigma_lst.append(evaluate_stress_tensor_function_E(shear, gamma, p_best, sigma_function))

                if only_data is False:
                    data_save[shear] = sigma_lst
                    axs.plot(gamma_lst, sigma_lst, color=colors[shear])

                axs.plot(gamma_data, sigma_data, marker=markers[shear], ls='', fillstyle='none', color=colors[shear],
                        label='(%s)' % shear)

            axs.legend(loc="upper left", frameon=False, fontsize=fs, handletextpad=0)
            axs.set_ylim((-0.1, 16.1))
            axs.set_xlim((0, 0.6))
            axs.set_yticks([0, 2, 4, 6, 8, 10, 12, 14, 16])
            # axs.set_yticks([])
            # axs.set_xticks([])
            axs.spines['right'].set_visible(False)
            axs.spines['top'].set_visible(False)
            # axs.spines['bottom'].set_visible(False)
            # axs.spines['left'].set_visible(False)
            axs.set_ylabel("stress $\sigma_{ij}$ [kPa]", fontsize=fs)
            axs.set_xlabel("amount of shear $\gamma$", fontsize=fs)
            axs.tick_params(axis='both', which='major', labelsize=fs)
            # lab_e = ' | ga solution: ' + ps
            # plt.title("error = " + str(round(error[lab].tolist()[es], 2)) + lab_e)

            df = pd.DataFrame(data=data_save)
            df.to_csv(save_to + 'data_shear_dokos_%s.csv' % psi_name)
            # plt.semilogy()
            plt.tight_layout()
            plt.savefig(save_to + 'plot_shear.png', dpi=300, transparent=True)
            # plt.show()



        return SSE_fit, Y_a, res, nfev, p_best