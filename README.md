# Cardiac Hyperelastic Evolutionary Symbolic Regression Algorithm (CHESRA)

CHESRA was developed to automatically derive cardiac strain energy functions (SEFs) from experimental data. 
It is an evolutionary framework that manipulates symbolic representations of cardiac SEFs to fit experimental 
observations while minimizing SEF complexity. CHESRA takes one or more experimental 
datasets of myocardial stress-strain relations as input and evolves a population 
of SEFs according to the fitness function $f_\text{fit}$ (Fig. 1).

![fig1_study_overview](https://github.com/user-attachments/assets/468243bf-c165-4195-a812-063f50011b53)
Fig. 1: Overview of CHESRA

## Environments

>Python = 3.10.6
> 
>deap = 1.3.3
> 
>func_timeout = 4.3.5
>
>lmfit = 1.0.3
> 
>matplotlib = 3.5.3
>
>numpy = 1.23.4
>
>pandas = 1.4.4
> 
>scipy = 1.9.3
> 
>seaborn = 0.13.2
> 
>sympy = 1.10.1
>
>dolfin = 2019.2.0.dev0
>
>yaml = 6.0.2


## Experiment Reproduction

We provide three reproduction scripts: 1) for creating SEFs with CHESRA and visualizing their fits to experimental data
2) for benchmarking the parameter variability when fitting CHESRA functions to tissue data and comparison to state-of-the art SEFs 3) for benchmarking 
of parameter variability in 3-D digital twins. 

0. Download repository and all required packages

1. Create SEFs with CHESRA

>`cd Experiments/CHESRAFunctions`
> 
>`python3 run_experiment.py`
> 
>`python3 create_figure.py`
> 
**Note:** use the default options to reproduce the results in the paper. Otherwise, all parameters can be adjusted. See

>`python3 run_experiment.py --help`


2. Parameter variability benchmark using tissue data
>`cd Experiments/Tissue_Benchmark`
> 
>`python3 run_experiment.py`
> 
>`python3 create_figure.py`

3. Parameter variability benchmark using a 3D digital twin

>`cd Experiments/DigitalTwin_Benchmark`
> 
>`python3 run_experiment.py -energy_function chesra1`
> 
>`python3 run_experiment.py -energy_function chesra2`
> 
>`python3 run_experiment.py -energy_function holzapfel-ogden`
> 
>`python3 run_experiment.py -energy_function martonova3`
> 
>`python3 create_figure.py`

## The Digital Twin Data
The digital twin benchark requires LV pressure-volume traces, pressure-free biventricular geometries and corresponding synthetic motion fields. For the sake of completeness the original (pressurized) mesh at end-diastole is included as the file `pressurized_bivmesh_60fibres.h5`.

