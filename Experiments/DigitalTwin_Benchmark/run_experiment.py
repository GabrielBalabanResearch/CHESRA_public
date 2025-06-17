import argparse
import numpy as np
import yaml
import pandas as pd
import dolfin as df
import os

from simulation.elasticity_estimation import estimate_elasticity_displacementloss
from simulation.helpers import set_ffc_params, pretty_print_dict, simpleheartlogger

#OPTIMIZATION PARAMETERS
OPT_ALGORITHM = "TNC" #trust-krylov
OPT_TOL = 1e-6
DISP_NOISE = 0.5

def convert_numpy_scalars_to_python(data):
    """Recursively convert numpy scalar values to Python native types."""
    if isinstance(data, dict):
        return {key: convert_numpy_scalars_to_python(value) for key, value in data.items()}
    elif isinstance(data, np.generic):  # Check for numpy scalar types
        return float(data)
    elif isinstance(data, list):
        return [convert_numpy_scalars_to_python(item) for item in data]
    else:
        return data

def main(energy_function):
    set_ffc_params()
    df.set_log_level(40)
    
    experiment_params = yaml.load(open("experiment_params.yaml", "r"), Loader = yaml.SafeLoader)
    simulation_params = yaml.load(open("simulation/simulation_config.yaml", "r"), Loader = yaml.SafeLoader)
    lhs_sample = pd.DataFrame(experiment_params["latin_hypercube_samples"][energy_function])
    
    run_nums = np.arange(args.run_start, args.run_end)
    lhs_subsample = lhs_sample.iloc[run_nums]

    for i, lhs_params in lhs_subsample.iterrows():
        simulation_params["energy_function"] = energy_function
        simulation_params["material_parameters"] = dict(lhs_params) 
        simulation_params["mesh"]["path"] =  f"digital_twin_data/{energy_function}/unloaded_geo.hdf5"
        simulation_params["mesh"]["microstructure"] =  f"digital_twin_data/{energy_function}/unloaded_geo_microstructure.h5"
        simulation_params["mechdata"] = f"digital_twin_data/pressure_traces.csv"
        simulation_params["output"]["files"]["path"] = f"results/{energy_function}/{i}"
        simulation_params["output"]["files"]["logfile"] = f"results/{energy_function}/parameter_estimation.log"
    
        simpleheartlogger.logpath = f"results/{energy_function}/{i}/paramest.log"

        simulation_params["elasticity_estimation"]["lower_bound"] = list(experiment_params["optimization_lower_bounds"][energy_function].values())
        simulation_params["elasticity_estimation"]["displacement_db"] = f"digital_twin_data/{energy_function}/noisy_disps/{i}_{DISP_NOISE}.hdf5"
        
        simpleheartlogger.log(pretty_print_dict(simulation_params))
        
        config_file = os.path.join(simulation_params["output"]["files"]["path"], "config_params.yaml")
        
        yaml.dump(convert_numpy_scalars_to_python(simulation_params),
                  open(config_file, "w"), 
                  default_flow_style = False)

        estimate_elasticity_displacementloss(simulation_params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-energy_function",
                         choices = ["chesra1", "chesra2", "martonova3", "holzapfel-ogden"],
                         default = "chesra1")

    parser.add_argument("-run_start",
                          type = int,
                          default = 0)

    parser.add_argument("-run_end",
                          type = int,
                          default = 20)
    args = parser.parse_args()
    main(args.energy_function)