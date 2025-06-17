from pathlib import Path
from matplotlib import pyplot as plt
import argparse
import os
import pandas as pd
import yaml
import ast
import re
import numpy as np
import seaborn as sns

RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

def gather_results():
    matparam_names_all = {}
    results_all = {}
    for dirpath, dirnames, filenames in os.walk(RESULTS_FOLDER):
        if "optresults.txt" in filenames:
            energy_function, run_number = Path(dirpath).parts[-2:]

            #config_params = yaml.load(open(config_params_path, "r"))
            optresults = open(os.path.join(dirpath, "optresults.txt"), "r").readlines()
            optresults = parse_optresult_file(optresults)
            matparam_estvals = optresults["x"]

            fun = optresults.get("fun")
            nfev = optresults.get("nfev")
            
            #Extract all numbers from the Run-time string
            runtime = re.findall(r"[-+]?\d*\.?\d+", optresults.get("Run-time"))
            assert(len(runtime)) == 1
            runtime = float(runtime[0])

            loss_lines, loss_vals= find_total_loss_values(os.path.join(dirpath, "paramest.log"))
            initial_fun = loss_vals[0]

            config_params_path = os.path.join(dirpath, "config_params.yaml")
            config_params = yaml.load(open(config_params_path, "r"), Loader = yaml.SafeLoader)
            
            matparam_names = config_params["material_parameters"].keys()
            matparam_inivals = list(config_params["material_parameters"].values())

            if not energy_function in results_all.keys():
                results_all[energy_function] = []
                matparam_names_all[energy_function] = matparam_names

            new_results = [run_number,
                           initial_fun,
                           fun,
                           nfev,
                           runtime] + matparam_inivals + matparam_estvals
            
            results_all[energy_function].append(new_results)

    output_results(results_all, matparam_names_all)

def output_results(results_all, matparam_names_all):
    output_folder = os.path.join("results", "gathered")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for energy_function, results in results_all.items():
        ini_names = ["ini_" + name for name in matparam_names_all[energy_function]]
        est_names = ["est_" + name for name in matparam_names_all[energy_function]]
        other_names = ["run number", "ini fun", "fun", "nfev", "runtime"]
        df = pd.DataFrame(results, columns = other_names + ini_names + est_names)
        output_file = os.path.join(output_folder, energy_function + "_results.csv")
        
        df = df.sort_values(by = ["run number"])
        df.to_csv(output_file, index=False)

        print("Results have been saved to", output_file)

def parse_array_string(array_string):
    # Remove the leading and trailing square brackets
    array_string = array_string.strip('[]')
    
    # Use regex to find all numbers, including those in scientific notation
    numbers = re.findall(r'[\d\.\+\-e]+', array_string)
    
    # Convert the found strings to floats
    numbers = [float(num) for num in numbers]
    
    return numbers

def parse_optresult_file(optresults):
    # Initialize an empty dictionary
    result_dict = {}
    # Iterate over each string in the list
    for i in range(len(optresults)):
        current_line = optresults[i]
        
        if ":" not in current_line:
            #print("skipped\n\t", current_line)
            continue

        # Split each string at the first occurrence of ':'
        key, value = current_line.split(':', 1)
        # Trim whitespace from key and value
        key = key.strip()
        value = value.strip()
        # Remove newline characters
        value = value.replace('\n', '')
        # Check if value starts with 'array' and evaluate it
        if value.startswith('array') or value.startswith('['):
             # Initialize a variable to hold the concatenated array definition
            if value.startswith('array'):
                 array_definition = value.split('array', 1)[1].strip()
            elif value.startswith('['):
                array_definition = value
            
            # Check if the array definition spans multiple lines
            j = i +1

            # Before the loop where you concatenate next_line to array_definition
            while array_definition.endswith('\\') or ('[' in array_definition and ']' not in array_definition) or ('(' in array_definition and ')' not in array_definition):
                # Read the next line. This part depends on how you're reading the file.
                next_line = optresults[j]  # Replace this with actual code to read the next line
                
                # Remove line continuation character if present and strip whitespace
                next_line = next_line.replace('\\', ' ').strip()
                
                # Concatenate the next line to the array definition
                array_definition += " " + next_line
                j += 1

            # Once the complete array definition is obtained, evaluate it
            value = parse_array_string(array_definition)
        else:
            try:
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass  # Keep the value as is if it's not a number
        # Add the key-value pair to the dictionary
        result_dict[key] = value
    return result_dict

def find_total_loss_values(file_path):
    # Define the regex pattern to match "Total loss = xx.xx" and capture the float value
    pattern = r"Total loss = ([+-]?\d*\.\d+|\d+)"
    
    matching_lines = []
    loss_values = []
    
    # Open the file and read line by line
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                matching_lines.append(line.strip())
                loss_values.append(float(match.group(1)))  # Extract and convert the loss value to float
    
    return matching_lines, loss_values

def plot_results():
    gt_params = yaml.load(open("experiment_params.yaml", "r"), Loader = yaml.SafeLoader)["ground_truth"]
    
    lhs_results_root = os.path.join(RESULTS_FOLDER, "gathered")
    
    chesra1_results_file = os.path.join(lhs_results_root,"chesra1_results.csv")
    chesra2_results_file = os.path.join(lhs_results_root, "chesra2_results.csv")
    hao_results_file = os.path.join(lhs_results_root, "holzapfel-ogden_results.csv")
    mart_results_file = os.path.join(lhs_results_root, "martonova3_results.csv")

    chesra1_df = pd.read_csv(chesra1_results_file)
    chesra2_df = pd.read_csv(chesra2_results_file)
    hao_df = pd.read_csv(hao_results_file)
    mart_df = pd.read_csv(mart_results_file)

    chesra1_paramerrors = calc_relerror(chesra1_df, gt_params["chesra1"])
    chesra2_paramerrors = calc_relerror(chesra2_df, gt_params["chesra2"])
    hao_paramerrors = calc_relerror(hao_df, gt_params["holzapfel-ogden"])
    mart_paramerrors = calc_relerror(mart_df, gt_params["martonova3"])

    fig, axs = plt.subplots(1, 4, figsize=(12, 6), sharey=True)

    hao_labels = ["$" + label.replace('_', '_{') + '}$' if '_' in label else "$" + label + "$" for label in gt_params["holzapfel-ogden"].keys()]

    plot_errors(chesra1_paramerrors, 
                gt_params["chesra1"], 
                axs[0],
                "pink",
                "$\psi_{CH1}$")

    plot_errors(chesra2_paramerrors,
                chesra2_gt_yaml,
                axs[1],
                "lightblue",
                "$\psi_{CH2}$")
    
    plot_errors(mart_paramerrors,
                gt_params["martonova3"],
                 axs[2],
                "purple",
                "$\psi_{MA}$")

    plot_errors(hao_paramerrors,
                gt_params["holzapfel-ogden"],
                axs[3],
                "orange",
                "$\psi_{HO}$")

    axs[0].set_ylabel("Normalized error")

    # Adjust layout
    plt.tight_layout()
    plt.savefig("digital_twin_benchmark.png")
    plt.show()
    

def calc_relerror(df, gt_yaml):
    errors = []
    for param in gt_yaml.keys():
        errors.append(np.abs((df["est_" + param] - gt_yaml[param])/gt_yaml[param]))
    return pd.DataFrame(np.array(errors).T, columns = gt_yaml.keys())

def plot_errors(paramerrors, gt_yaml, ax, color, title = None, Y_CUT = 3.5):
    paramnames = gt_yaml.keys()
    x_labels = ["$" + pname.replace('_', '_{') + '}$' if '_' in pname else "$" + pname + "$" for pname in gt_yaml.keys()]

    long_df = pd.melt(paramerrors, value_vars=paramnames, var_name="Parameter", value_name="Value")
    sns.boxplot(x="Parameter",
                y="Value",
                data=long_df,
                ax=ax,
                color = color,
                zorder = 10)
    
    for patch in ax.patches:
        patch.set_alpha(1)
    
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("")
    ax.set_yticks(np.arange(0, Y_CUT*2)/2.0)
    ax.set_ylim(0, Y_CUT)

    # Identify and mark outliers
    for i, param in enumerate(paramnames):
        outliers = paramerrors[paramerrors[param] > Y_CUT][param]
        if not outliers.empty:
            ax.scatter([i] * len(outliers), [Y_CUT] * len(outliers), color="red", marker="^", s=100, label="Outlier")

    if title:
        ax.set_title(title, loc = "left", fontsize = 16, weight = "bold")

if __name__ == "__main__":
    gather_results()
    plot_results()