import random
import numpy as np
import pandas as pd
import func_timeout
import pickle
from deap import base, creator, tools
import time
from multiprocessing import Pool
from CHESRA.function_tree import create_function, compute_function, mutation_alter, cross_over, count_preops, count_consts, \
    find_root, mutation_reduce, mutation_extend


#%% GENETIC ALGORITHM 
penality_value = np.inf
waiting_time = 200

class Ga_Config():
    def __init__(self,
             population_size,
             max_generations,
             hofsize,
             func_initlength,
             hyperp,
             variables,
             const_symb,
             functions,
             operators,
             shears, 
             biax_shears,
             mate_probability,
             mutate_reduce_probability,
             mutate_alter_probability,
             mutate_extend_probability,
             tournament_size,
             save_data_to, 
             get_exp_data,
             data_type,
             pop_path,
             max_length,
             check_duplicates):
        self.population_size = population_size
        self.max_generations = max_generations
        self.hofsize = hofsize,
        self.func_initlength = func_initlength
        self.hyperp = hyperp
        self.variables = variables
        self.const_symb = const_symb
        self.functions = functions
        self.operators = operators
        self.shears = shears
        self.biax_shears = biax_shears
        self.mate_probability = mate_probability
        self.mutate_reduce_probability=mutate_reduce_probability,
        self.mutate_alter_probability=mutate_alter_probability,
        self.mutate_extend_probability = mutate_extend_probability,
        self.tournament_size = tournament_size
        self.save_data_to = save_data_to
        self.get_exp_data = get_exp_data
        self.data_type = data_type
        self.pop_path = pop_path
        self.max_length = max_length
        self.check_duplicates = check_duplicates

def run_ga(toolbox):
    """
    Runs an instance of the genetic algorithm.
    Returns
    -------
        final_population : List[Individuals]
    """
    print('Evaluating initial population.')

    # 3. Calls _initialize_individuals and returns initial population
    if GA_CONFIG.pop_path == 0:
        population = toolbox.population(GA_CONFIG.population_size)
    else:
        population = pickle.load(open(GA_CONFIG.pop_path, 'rb'))

    if GA_CONFIG.check_duplicates is True:
        new_pop_idx = check_duplicates(population)

    hof = tools.HallOfFame(GA_CONFIG.hofsize[0]) #set up hall of fame

    # 4. Calls _evaluate_fitness on every individual in the population
    fitnesses = toolbox.map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = (fit,)

    if GA_CONFIG.hofsize[0] > 0:
        hof.update(population)
    
    # Note: visualize individual fitnesses with: population[0].fitness
    gen_fitnesses = [ind.fitness.values[0] for ind in population]
    gen_complex = [assess_complexity(ind) for ind in population] 
    gen_sRSS = []
    for i in range(0, len(population)):
        gen_sRSS.append(gen_fitnesses[i] - (gen_complex[i] * GA_CONFIG.hyperp))

    print(f'\tBest fitness is {np.min(gen_fitnesses)}')

    # Store initial population details for result processing.
    pickle.dump(population, open(GA_CONFIG.save_data_to+'/population0.pkl', 'wb'))
    final_population = [population]
    df_pop = pd.DataFrame()
    df_func = pd.DataFrame()
    df_fit = pd.DataFrame()
    df_sRSS = pd.DataFrame()
    df_complex = pd.DataFrame()
    df_hof = pd.DataFrame()

    label = 'gen'+ str(0)  
    df_fit[label] = gen_fitnesses
    df_sRSS[label] = gen_sRSS
    df_complex[label] = gen_complex

    df_fit.to_csv(GA_CONFIG.save_data_to+'/error_'+str(GA_CONFIG.hyperp)+'.csv', index=False)
    df_sRSS.to_csv(GA_CONFIG.save_data_to+'/sRSS_'+str(GA_CONFIG.hyperp)+'.csv', index=False)
    df_complex.to_csv(GA_CONFIG.save_data_to+'/complex_'+str(GA_CONFIG.hyperp)+'.csv', index=False)

    pop_list = []
    hof_list = []
    f_list = []

    for i in list(range(0,len(population))):
        x = list(population[i][0])
        pop_list.append([x[0]])
        f_list.append(len(x[1]))

    for i in list(range(0,len(hof))):
        x = list(hof[i][0])
        hof_list.append([x[0]])

    df_pop[label] = pop_list
    df_func[label] = f_list
    df_hof[label] = hof_list
    df_pop.to_csv(GA_CONFIG.save_data_to+'/pop_'+str(GA_CONFIG.hyperp)+'.csv', index=False)
    df_func.to_csv(GA_CONFIG.save_data_to+'/funcs_'+str(GA_CONFIG.hyperp)+'.csv', index=False)
    df_hof.to_csv(GA_CONFIG.save_data_to+'/hofs_'+str(GA_CONFIG.hyperp)+'.csv', index=False)

    for generation in range(1, GA_CONFIG.max_generations):
        print('Generation {}'.format(generation))
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print(current_time)
        # Offspring are chosen through tournament selection. They are then
        # cloned, because they will be modified in-place later on.

        # 5. DEAP selects the individuals 
        if GA_CONFIG.hofsize[0] > 0:
            selected_offspring = toolbox.select(population, len(population) - GA_CONFIG.hofsize[0])
        else: 
            selected_offspring = toolbox.select(population, len(population))

        offspring = [toolbox.clone(i) for i in selected_offspring]

        # 6. Mate the individualse by calling _mate()
        for i_one, i_two in zip(offspring[::2], offspring[1::2]):
            if random.random() < GA_CONFIG.mate_probability:
                toolbox.mate(i_one, i_two)
                del i_one.fitness.values
                del i_two.fitness.values

        # 7. Mutate the individualse by calling _mutate()
        for i in offspring:
            toolbox.mutate(i)
            del i.fitness.values

        # All individuals who were updated, either through crossover or
        # mutation, will be re-evaluated.

        #if GA_CONFIG.check_duplicates == 1: #remove duplicates before 
        #    new_off_idx = check_duplicates(offspring)
        #    for idx in new_off_idx:
        #        del offspring[idx].fitness.values

        # 8. Evaluating the offspring of the current generation
        updated_individuals = [i for i in offspring if not i.fitness.values]
        fitnesses = toolbox.map(toolbox.evaluate, updated_individuals)
        for ind, fit in zip(updated_individuals, fitnesses):
            ind.fitness.values = (fit,)

        if GA_CONFIG.hofsize[0] > 0:
            offspring.extend(hof.items) # add th best back to the population 
                   
        # Check if offspring were dublicated or if hof individials are already in population: if so, create random new individual; if not, keep original ind or hof ind
        if GA_CONFIG.check_duplicates is True:
            new_off_idx = check_duplicates(offspring)
            for idx in new_off_idx:
                del offspring[idx].fitness.values
            updated_individuals = [i for i in offspring if not i.fitness.values]
            fitnesses = toolbox.map(toolbox.evaluate, updated_individuals)
            for ind, fit in zip(updated_individuals, fitnesses):
                ind.fitness.values = (fit,)
        
        if GA_CONFIG.hofsize[0] > 0:
            hof.update(offspring) # update the hall of fame with the generated individuals

        population = offspring 

        gen_fitnesses = [ind.fitness.values[0] for ind in population]
        gen_complex = [assess_complexity(ind) for ind in population]
        gen_sRSS = []
        for i in range(0, len(population)):
            gen_sRSS.append(gen_fitnesses[i] - (gen_complex[i] * GA_CONFIG.hyperp))

        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print(current_time)
        print(f'\tBest fitness is {np.min(gen_fitnesses)}')

        pickle.dump(population, open(GA_CONFIG.save_data_to+'/population.pkl', 'wb'))
        final_population.append(population)
        
        #Save pop and gen as ga loops 
        label = 'gen'+ str(generation)  

        df_fit[label] = gen_fitnesses
        df_sRSS[label] = gen_sRSS
        df_complex[label] = gen_complex

        df_fit.to_csv(GA_CONFIG.save_data_to+'/error_'+str(GA_CONFIG.hyperp)+'.csv', index=False)
        df_sRSS.to_csv(GA_CONFIG.save_data_to+'/sRSS_'+str(GA_CONFIG.hyperp)+'.csv', index=False)
        df_complex.to_csv(GA_CONFIG.save_data_to+'/complex_'+str(GA_CONFIG.hyperp)+'.csv', index=False)

        pop_list = []
        f_list = []
        hof_list = []

        for i in list(range(0,len(population))):
            x = list(population[i][0])
            pop_list.append([x[0]])
            f_list.append(len(x[1]))

        for i in list(range(0,len(hof))):
            x = list(hof[i][0])
            hof_list.append([x[0]])

        df_pop[label] = pop_list
        df_func[label] = f_list
        df_hof[label] = hof_list
        df_pop.to_csv(GA_CONFIG.save_data_to+'/pop_'+str(GA_CONFIG.hyperp)+'.csv', index=False)
        df_func.to_csv(GA_CONFIG.save_data_to+'/funcs_'+str(GA_CONFIG.hyperp)+'.csv', index=False)
        df_hof.to_csv(GA_CONFIG.save_data_to+'/hofs_'+str(GA_CONFIG.hyperp)+'.csv', index=False)
        
        print("Finished saving population.")

    return final_population

def _initialize_individuals():
    """
    Creates a population of x random functions with maximum_length y
    """
    #if GA_CONFIG.eq_file == 0:
    function = create_function(GA_CONFIG.func_initlength, GA_CONFIG.variables, GA_CONFIG.functions, GA_CONFIG.const_symb, GA_CONFIG.operators)
    eq = (compute_function(function))
    
    return [eq, function]

def _mate(i_one, i_two):
    """Performs crossover between two individuals. It selects a random node of two 
    individuals and swaps the rest of the tree at that node. 
    There may be a possibility no parameters are swapped. This probability
    is controlled by `GA_CONFIG.gene_swap_probability`. Modifies
    both individuals in-place.

    This function also has the option to assess the length of each child that is generated.
    If max_length is set to a value than the function will assess if each child is less than
    that threshold. If it is, the child will be included in the generations. If not, the 
    parent will be kept instead. 
    Args:
        i_one: An individual in a population.
        i_two: Another individual in the population.
    """
    #print('max length:', GA_CONFIG.max_length)
    #print('inputted function 1 to be mated: ', i_one[0][0], len(i_one[0][1]))
    #print('inputted function 2 to be mated: ', i_two[0][0], len(i_two[0][1]))
    if GA_CONFIG.max_length == 0:
        child1, child2 = cross_over(i_one[0][1], i_two[0][1], GA_CONFIG.const_symb)
        eq_child1 = (compute_function(child1))
        eq_child2 = (compute_function(child2))

        i_one[0][0] = eq_child1
        i_one[0][1] = child1
        i_two[0][0] = eq_child2
        i_two[0][1] = child2
    
    else:
        child1, child2 = cross_over(i_one[0][1], i_two[0][1], GA_CONFIG.const_symb)
        #print('initial child 1', compute_function(child1), len(child1))
        #print('initial child 2', compute_function(child2), len(child2))

        if len(child1)<GA_CONFIG.max_length:
            eq_child1 = (compute_function(child1))
            i_one[0][0] = eq_child1
            i_one[0][1] = child1
        else:
            pass

        if len(child2)<GA_CONFIG.max_length:
            eq_child2 = (compute_function(child2))
            i_two[0][0] = eq_child2
            i_two[0][1] = child2
        else:
            pass

        #print('final child 1', i_one[0][0])
        #print('final child 2', i_two[0][0])
    pass

def _mutate(individual):
    """Performs a mutation on an individual in the population.
    Chooses a random operator and modifies an individual
    in-place.
    Args:
        individual: An individual to be mutated.
    """

    if random.random() < GA_CONFIG.mutate_alter_probability[0]:
        #print('inputted function to be mutated', individual[0][0], len(individual[0][1]))
        if GA_CONFIG.max_length == 0:
            mut_function = mutation_alter(individual[0][1], GA_CONFIG.variables, GA_CONFIG.functions, GA_CONFIG.const_symb, GA_CONFIG.operators)
            individual[0][1] = mut_function
            individual[0][0] = (compute_function(mut_function))
        else:
            mut_function = mutation_alter(individual[0][1], GA_CONFIG.variables, GA_CONFIG.functions, GA_CONFIG.const_symb, GA_CONFIG.operators)
            #print('inital mutated alter function:', compute_function(mut_function), len(mut_function))
            if len(mut_function) < GA_CONFIG.max_length:
                individual[0][1] = mut_function
                individual[0][0] = (compute_function(mut_function))
            else:
                pass # if the function is bigger than the max length, keep the original
            #print('final mutated alter function: ', individual[0][0])
    else:
        pass

    if random.random() < GA_CONFIG.mutate_reduce_probability[0]:
        mut_function = mutation_reduce(individual[0][1], GA_CONFIG.const_symb)
        individual[0][1] = mut_function
        individual[0][0] = (compute_function(mut_function))
    else:
        pass

    if random.random() < GA_CONFIG.mutate_extend_probability[0]:
        #print('inputted function to be mutated', individual[0][0], len(individual[0][1]))
        if GA_CONFIG.max_length == 0:
            mut_function = mutation_extend(individual[0][1], GA_CONFIG.variables, GA_CONFIG.functions, GA_CONFIG.const_symb, GA_CONFIG.operators)
            individual[0][1] = mut_function
            individual[0][0] = (compute_function(mut_function))
        else:
            mut_function = mutation_extend(individual[0][1], GA_CONFIG.variables, GA_CONFIG.functions, GA_CONFIG.const_symb, GA_CONFIG.operators)
            #print('inital muatated extend function:', compute_function(mut_function), len(mut_function))
            if len(mut_function)<GA_CONFIG.max_length:
                individual[0][1] = mut_function
                individual[0][0] = (compute_function(mut_function))
            else:
                pass
            #print('final mutated extend function: ', individual[0][0])
    else:
        pass

    pass

def run_function(f, ind, num_params, data_path='', max_wait=200, default_value=np.inf, plot=False, save_to='', fs=18):
    """ 
    This allows for a function to be run for a certain amount of time. If it exceeds the time limit set, a default value will be outputed.
    Args:
        f: function you would like to time-limit
        ind: individual found by CHESRA
        num_params: number of parameters in psi
        data_path: path to the experimental data
        max_wait: maximum amount of time you would like the function to run for. 
        default value: output if the function reaches the time limit.
    """
    try:
        return func_timeout.func_timeout(max_wait, f, args=[ind, num_params, data_path],
                                         kwargs={'plot':plot, 'save_to':save_to})
    except func_timeout.FunctionTimedOut:
        pass
    return default_value, [1], [1], [1], [1]

def assess_complexity(ind):
    """ Finds the complexity, i.e. the function length, of the given function found by the GA.
    Args:
        ind: An given function found by the GA
    """
    eq_length = len(ind[0][1])
    length_score = eq_length

    complexity = length_score
    
    return complexity 

def _evaluate_fitness(ind):
    """ Takes in a function found by the GA and calcualtes its fitness score. Specifically, it assesses its complexity
    and standardized residual sum of squares by fitting the function to the experimental data.
    Args:
        ind: A given function found by the GA
    """
    # Check how complex the function is - the more complex, the higher the error score
    complexity = assess_complexity(ind)

    sRSS_sum_shear = 0
    sRSS_sum_biax = 0
    n_biax = 0
    n_shear = 0
    num_params = count_consts(ind[0][1][find_root(ind[0][1])], 0)
    
    for dataset in GA_CONFIG.data_type:
        SSE, Y_a, res, nfev, _ = run_function(dataset.fit_data, ind, num_params+1, data_path=GA_CONFIG.get_exp_data,
                                              max_wait=waiting_time, default_value=penality_value)
        sRSS = SSE / np.sum((np.array(Y_a) - np.mean(Y_a)) ** 2)

        if dataset.exp_type == 'biaxial':
            sRSS_sum_biax += sRSS
            n_biax += 1
        if dataset.exp_type == 'shear':
            sRSS_sum_shear += sRSS
            n_shear += 1

    # n_data = len(GA_CONFIG.data_type)
    fitness = (complexity*GA_CONFIG.hyperp)
    n_exp = (n_biax>0) + (n_shear>0)
    if n_biax > 0:
        fitness += sRSS_sum_biax/n_biax/n_exp
    if n_shear > 0:
        fitness += sRSS_sum_shear/n_shear/n_exp

    return fitness

def start_ga(pop_size, max_generations, hofsize, data_lst, hyperparameter, save_to, exp_data_path,
             mate_prob = 0.6, mut_reduce = 0.1, mut_alter = 0.5, mut_extend = 0.8,
             varis = ['I1', 'I2', 'I3', 'I4f', 'I4s', 'I4n', 'I5f', 'I5s', 'I5n', 'I8fs', 'I8fn', 'I8ns'],
             funcs = ['sp.sqrt', 'pow2', 'sp.exp', 'sp.log', '-'],
             const_symb = ['p'], opts = ['-', '+', '/', '*'],
             multithread = True, pop_path = 0, max_length = 0, n_cores=0, check_duplicates = False):
    """ Function which runs CHESRA.
    Args:
        pop_size: number of individuals in each generation
        max_generations: number of generations the GA will run for. 
        hofsize: select number to turn on elitism. The number represents the number of best individuals that will be preserved in all generations. Use 0 to turn off elitism. 
        data_lst: type of data to include in list form (i.e [shear, biaxial])
        hyperparameter: value that will balance the complexity and fitness
        save_to: path data will save to
        exp_data_path: path where the ga could find the experimental data for fitness calculations
        mate_prob: probability at which each function in the population will mate (crossover)
        mut_reduce: probability at which each function in the population will undergo a reduce mutation
        mut_alter: probability at which each function in the population will undergo a point mutation
        varis: variable list as options for the random formation of each function in the initial population
        funcs: function list as options for the random formation of each function in the initial population
        params: parameters list as options for the random formation of each function in the initial population
        opts: operator list as options for the random formation of each function in the initial population
        multithread: Would you like to run CHERPA with multithreading?
    """

    # 1. Initializing GA hyperparameters
    global GA_CONFIG
    GA_CONFIG = Ga_Config(population_size=pop_size,
                          max_generations=max_generations,
                          hofsize=hofsize,
                          data_type=data_lst,
                          func_initlength=5,
                          hyperp=hyperparameter,
                          variables=varis,
                          const_symb=const_symb,
                          functions=funcs,
                          operators=opts,
                          shears=['fs', 'fn', 'sf', 'sn', 'nf', 'ns'],
                          biax_shears=['ff', 'ss'],
                          mate_probability=mate_prob,
                          mutate_reduce_probability=mut_reduce,
                          mutate_alter_probability=mut_alter,
                          mutate_extend_probability=mut_extend,
                          tournament_size=2,
                          save_data_to=save_to,
                          get_exp_data=exp_data_path,
                          pop_path=pop_path,
                          max_length=max_length,
                          check_duplicates=check_duplicates)

    creator.create('FitnessMin', base.Fitness, weights=(-1.0,))

    creator.create('Individual', list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()

    toolbox.register('init_param',
                    _initialize_individuals)
    toolbox.register('individual',
                    tools.initRepeat,
                    creator.Individual,
                    toolbox.init_param,
                    n=1)
    toolbox.register('population',
                    tools.initRepeat,
                    list,
                    toolbox.individual)

    toolbox.register('evaluate', _evaluate_fitness)
    toolbox.register('select',
                     tools.selTournament,
                     tournsize=GA_CONFIG.tournament_size)
    toolbox.register('mate', _mate)
    toolbox.register('mutate', _mutate)


    ## To speed things up with multi-threading
    if multithread is True:
        if n_cores == 0:
            p = Pool()
        else:
            p = Pool(n_cores)
        toolbox.register("map", p.map)
    else:
        toolbox.register("map", map)

    # 2. Calling the GA to run
    final_population = run_ga(toolbox)

    return final_population

def check_duplicates(data):
    """This function checks if there are multiple duplicates of a function in the population. If yes, the duplicates
    are replaced by random functions."""

    new_eqs = []
    updated_idx = []
    for i in list(range(0, len(data))):
        if data[i][0][0] not in new_eqs:
            new_eqs.append(data[i][0][0])
        else:
            while data[i][0][0] in new_eqs:
                data[i][0] = _initialize_individuals()
            updated_idx.append(i)
            new_eqs.append(data[i][0][0])
    return(updated_idx)

