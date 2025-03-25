#%% IMPORT FUNCTIONS
import sympy as sp
import pandas as pd
import numpy as np
import argparse
import re
from ga_functions import run_function
from ga_functions import start_ga
from data_classes.shear_sommer import ShearSommer
from data_classes.biaxial_sommer import BiaxialSommer
from data_classes.biaxial_yin import BiaxialYin
from data_classes.shear_dokos import ShearDokos
from data_classes.equibiaxial_novak import EquibiaxialNovak
from data_classes.biaxial_novak import BiaxialNovak

# Define a function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Run CHESRA with specified parameters.')
    parser.add_argument('--data_lst', nargs='+', default=[BiaxialYin(), EquibiaxialNovak(), BiaxialNovak(),
                                                          BiaxialSommer(), ShearDokos(), ShearSommer()],
                        help='List of datasets to include for SEF training')
    parser.add_argument('--alpha', type=float, required=True, help='Hyperparameter')
    parser.add_argument('--save_to', type=str, default='.', help='Path where to save the results')
    parser.add_argument('--max_gen', type=int, default=50, help='Number of generations')
    parser.add_argument('--pop_size', type=int, default=200, help='Population size')
    parser.add_argument('--hof_size', type=int, default=20, help='Hall of fame size')
    parser.add_argument('--mate_prob', type=float, default=0.5, help='Mating rate')
    parser.add_argument('--mut_reduce', type=float, default=0.5, help='Mutation reduction rate')
    parser.add_argument('--mut_alter', type=float, default=0.25, help='Mutation alteration rate')
    parser.add_argument('--mut_extend', type=float, default=0.75, help='Mutation extension rate')
    parser.add_argument('--varis', nargs='+', default=[
        '(I1-3)**2', '(I2-3)**2', '(I4f-1)**2', '(I4s-1)**2', '(I4n-1)**2', '(I5f-1)**2',
        '(I5s-1)**2', '(I5n-1)**2', '(I8fs)**2', '(I8fn)**2', '(I8ns)**2'
    ], help='List of variables')
    parser.add_argument('--funcs', nargs='+', default=['sp.exp'], help='List of functions')
    parser.add_argument('--opts', nargs='+', default=['+', '*'], help='List of operators')
    parser.add_argument('--check_duplicates', type=int, default=1, help='Check for duplicates')
    parser.add_argument('--multithread', type=bool, default=True, help='Use multithreading')
    parser.add_argument('--n_cores', type=int, default=64, help='Number of cores to use')

    return parser.parse_args()


# Define the main function
def main(args):
    all_individuals = start_ga(
        pop_size=args.pop_size,
        max_generations=args.max_gen,
        hofsize=args.hof_size,
        data_lst=args.data_lst,
        hyperparameter=args.alpha,
        save_to=args.save_to,
        exp_data_path='',
        mate_prob=args.mate_prob,
        mut_reduce=args.mut_reduce,
        mut_alter=args.mut_alter,
        mut_extend=args.mut_extend,
        varis=args.varis,
        funcs=args.funcs,
        opts=args.opts,
        check_duplicates=args.check_duplicates,
        multithread=args.multithread,
        n_cores=args.n_cores
    )

    return all_individuals


if __name__ == '__main__':
    args = parse_args()

    all_individuals = main(args)

    # read the CHESRA output
    lab = 'gen' + str(args.max_gen - 1)
    error = pd.read_csv(args.save_to + '/error_' + str(args.alpha) + '.csv')
    comp = pd.read_csv(args.save_to + '/complex_' + str(args.alpha) + '.csv')
    pop = pd.read_csv(args.save_to + '/pop_' + str(args.alpha) + '.csv')

    # extract the best function
    es = np.argmin(error[lab].tolist())
    ps = eval(pop[lab].tolist()[es])[0]
    ind = [[ps]]
    num_params = len(list(dict.fromkeys(re.findall('p[0-9]+', ps)))) + 1
    print('Final CHESRA result:')
    print('\tBest function is psi = %s' % sp.simplify(ps, locals={'sp.exp': sp.exp}))
    print('\tBest fitness is %s' % error[lab].tolist()[es])

    # do the fit
    for data in args.data_lst:
        SSE, Y_a, res, nfev, p_best = run_function(data.fit_data, ind, num_params, plot=True, save_to=args.save_to)
