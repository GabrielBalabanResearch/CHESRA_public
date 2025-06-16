import os
import dolfin as df
import pandas as pd
import numpy as np
import yaml
import datetime
import warnings
from matplotlib import pyplot as plt

#Dolfin optimizations for speed
def set_ffc_params():
	#FFC = fenics form compiler
	#Fast math does not confirm to floating point standard.
	#It is on now but should be removed if there is any doubt about it's effects.

	flags = ["-O3", "-ffast-math", "-march=native"]    
	df.parameters["form_compiler"]["quadrature_degree"] = 4
	df.parameters["form_compiler"]["representation"] = "uflacs"
	#df.parameters["form_compiler"]["representation"] = "quadrature"
	df.parameters["form_compiler"]["cpp_optimize"] = True
	df.parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
	warnings.filterwarnings("ignore", category=UserWarning, message=".*quadrature representation is deprecated.*")

def cross_vectorfunctions(v1, v2):
	"""
	Normalizes a vector function to have unit length.
	"""
	v1_values = v1.vector().get_local()
	v2_values = v2.vector().get_local()

	v1_components = v1.function_space().ufl_element().value_size()
	v2_components = v2.function_space().ufl_element().value_size()

	v2_values_reshaped = v1_values.reshape((-1, v1_components))
	v1_values_reshaped = v2_values.reshape((-1, v2_components))

	v_cross_values = np.cross(v1_values_reshaped, v2_values_reshaped)
	
	v_cross = df.Function(v1.function_space())
	v_cross.vector().set_local(v_cross_values.flatten())
	v_cross.vector().apply("insert")
	return v_cross

def normalize_vectorfunction(v):
	v_values = v.vector().get_local()
	num_components = v.function_space().ufl_element().value_size()
	v_values_reshaped = v_values.reshape((-1, num_components))
		
	norms = np.linalg.norm(v_values_reshaped, axis=1, keepdims=True)
	norms[norms == 0] = 1  # Avoid division by zero
	v_values_normalized = v_values_reshaped / norms

	#v_normalized = df.Function(v.function_space())
	v.vector().set_local(v_values_normalized.flatten())
	v.vector().apply("insert")
	return v

"""
def pretty_print_dict(d, indent=0):
    Recursively print a dictionary in a hierarchical format with leaf nodes on the same line.
    for key, value in d.items():
        if isinstance(value, dict):
            print(" " * indent + str(key) + ":")
            pretty_print_dict(value, indent + 4)
        else:
            print(" " * indent + f"{key}: {value}")
"""

def pretty_print_dict(d, indent=0):
    """Recursively format a dictionary in a hierarchical format with leaf nodes on the same line."""
    result = ""
    for key, value in d.items():
        if isinstance(value, dict):
            result += " " * indent + str(key) + ":\n"
            result += pretty_print_dict(value, indent + 4)
        else:
            result += " " * indent + f"{key}: {value}\n"
    return result

class SolutionLoader(object):
	"""
	Loads a mechanical state into the MechSolver class
	"""
	def __init__(self, dispfilepath, mechdata_traces_path):
		self.dispfilepath = dispfilepath
		self.mechdata_traces_path = mechdata_traces_path
		self.mechdata = pd.read_csv(mechdata_traces_path)

	def load_solution(self, t, mechsolver):
		with df.HDF5File(df.MPI.comm_world, self.dispfilepath, "r") as f:
			if f.has_dataset("displacement/" + str(t)):
				f.read(mechsolver.d, "displacement/" + str(t))
				contraction, p_lv, p_rv = np.array(self.mechdata[self.mechdata["time (ms)"] == t][["contraction", "lv pressure (kPa)", "rv pressure (kPa)"]])[0]
				mechsolver.p_lv.assign(p_lv)
				mechsolver.p_rv.assign(p_rv)
				mechsolver.contraction.assign(contraction)
				return True
		return False

def gather(vec):
	"""
	From fenics_adjoint.utils
	https://github.com/dolfin-adjoint/dolfin-adjoint/blob/main/src/fenics_adjoint/utils.py
	"""
	if isinstance(vec, df.cpp.function.Function):
		vec = vec.vector()
	
	if isinstance(vec, df.cpp.la.GenericVector):
		arr = vec.gather(np.arange(vec.size(), dtype='I'))
	elif isinstance(vec, list):
		return list(map(gather, vec))
	else:
		arr = vec  # Assume it's a gathered numpy array already
	return arr

def mpi_print(orig_msg):
	"""
	Prints only one copy of a string if the string is the same across all processes.
	"""
	msg_str = str(orig_msg)
	all_msgs = df.MPI.comm_world.gather(msg_str, root = 0)
	
	if df.MPI.rank(df.MPI.comm_world) == 0:
		# Check if all messages are identical
		if all(msg == all_msgs[0] for msg in all_msgs):
			print(all_msgs[0])
		else:
			for msg in all_msgs:
				print(msg)

class SimpleHeartLogger(object):
	def __init__(self, logpath = None):
		self.logpath = logpath
		self.tab = 0

	def log(self, message):
		str_message = str(message)

		if self.logpath and not os.path.isdir(os.path.dirname(self.logpath)) and df.MPI.comm_world.rank == 0:
			os.makedirs(os.path.dirname(self.logpath))

		if self.logpath and df.MPI.comm_world.rank == 0:
			with open(self.logpath, "a") as f:
				f.write("\t"*self.tab + str_message + "\n")
		mpi_print("\t"*self.tab + str_message)

	def increase_tab(self):
		self.tab += 1	

	def decrease_tab(self):	
		self.tab -= 1

simpleheartlogger = SimpleHeartLogger()