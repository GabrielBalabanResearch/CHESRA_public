import argparse
import numpy as np
import yaml
import pandas as pd

from simulation.elasticity_estimation import estimate_elasticity_displacementloss

#OPTIMIZATION PARAMETERS
OPT_ALGORITHM = "TNC" #trust-krylov
OPT_TOL = 1e-6

def main(energy_function):
    experiment_params = yaml.load(open("experiment_params.yaml", "r"), Loader = yaml.SafeLoader)

    simulation_params = yaml.load(open("simulation/simulation_config.yaml", "r"), Loader = yaml.SafeLoader)

    lhs_sample = pd.DataFrame(experiment_params["latin_hypercube_samples"][energy_function])

    for i, lhs_params in lhs_sample.iterrows():
        simulation_params["material_parameters"] = dict(lhs_params) 
        simulation_params["mesh"]["path"] =  f"synthetic_targetdata/{energy_function}/unloaded_geo.hdf5"
        simulation_params["mesh"]["microstructure"] =  f"synthetic_targetdata/{energy_function}/unloaded_geo_microstructure.hdf5"
        simulation_params["mechdata"] = f"synthetic_targetdata/pressure_traces.csv"
        simulation_params["output"]["files"]["path"] = f"results/{energy_function}"
        simulation_params["output"]["files"]["logfile"] = f"results/{energy_function}/parameter_estimation.log"
    
        simulation_params["elasticity_estimation"]["lower_bound"] = list(experiment_params["optimization_lower_bounds"][energy_function].values())
        simulation_params["elasticity_estimation"]["displacement_db"] = f"synthetic_targetdata/{energy_function}/displacements.hdf5"
        
        print(yaml.dump(simulation_params, default_flow_style = False))

        estimate_elasticity_displacementloss(simulation_params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-energy_function",
                          choices = ["chesra1", "chesra2", "holzapfel-ogden"],
                          default = "chesra1")
    args = parser.parse_args()
    main(args.energy_function)