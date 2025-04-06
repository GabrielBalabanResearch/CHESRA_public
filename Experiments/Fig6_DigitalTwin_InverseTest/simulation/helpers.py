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

def recursive_merge_dicts(dict1, dict2):
	for key, value in dict2.items():
		if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
			# Recursively merge nested dictionaries
			dict1[key] = recursive_merge_dicts(dict1[key], value)
		else:
			# Merge non-dictionary values
			dict1[key] = value
	return dict1

class SimParams(dict):
	def export_to_file(self, filename):
		try:
			with open(filename, "w") as outfile:
				yaml.dump(dict(self), outfile)
		except IOError as e:
			simpleheartlogger.log(f"Error writing to file {filename}: {e}")
	
	def recursive_merge(self, update_params):
		return SimParams(recursive_merge_dicts(update_params, self))
		
	def add_timestamp_to_paths(self):
		comm = df.MPI.comm_world
		rank = df.MPI.rank(comm)
		
		try:
			if rank == 0:
				timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
			else:
				timestamp = None
			 # Broadcast the timestamp from the root process to all others
			timestamp = comm.bcast(timestamp, root=0)

			base_path = self.get("output", {}).get("files", {}).get("path", "")
			self["output"]["files"]["path"] = os.path.join(base_path, timestamp)
		except KeyError as e:
			simpleheartlogger.log(f"Missing key for path construction: {e}")

		logfile = self.get("output", {}).get("files", {}).get("logfile")
		if logfile:
			path, fname = os.path.split(logfile)
			self["output"]["files"]["logfile"] = os.path.join(path, timestamp, fname)

	@staticmethod
	def load_from_file(filename):
		simparams = SimParams(yaml.load(open(filename, "r"), Loader = yaml.SafeLoader))
		if simparams.get("material_parameters") is not None and simparams["material_parameters"].get("path") is not None:
			simparams = SimParams.material_parameters_from_file(simparams)

		if simparams.get("output", {}).get("files", {}).get("timestamp"):
			simparams.add_timestamp_to_paths()
		return simparams

	@staticmethod
	def material_parameters_from_file(simparams):
		matparams = yaml.load(open(simparams["material_parameters"]["path"], "r"), Loader = yaml.FullLoader)
		matparamset = simparams["material_parameters"]["matparamset"]
		energy_function = simparams["energy_function"]
		simparams["material_parameters"] = matparams[matparamset][energy_function]
		return simparams
	
	def __str__(self):
		hdr = "-"*80 + "\nConfiguration\n" + "-"*80 + "\n"
		params = yaml.dump(dict(self), default_flow_style = False)
		return hdr + params

class DisplacementRecorder():
	def __init__(self, hdf5file):
		self.hdf5file = hdf5file 

		if os.path.exists(self.hdf5file) and df.MPI.rank(df.MPI.comm_world) == 0:
			os.remove(self.hdf5file)

	def save(self, disp, t):
		if os.path.exists(self.hdf5file): 
			mode = "a"
		else: 
			mode = "w"
		with df.HDF5File(df.MPI.comm_world, self.hdf5file, mode) as temp_output:
			temp_output.write(disp, "displacement/{}".format(t))

	def reorder_to_xdmf(self, mechsolver, ordered_times):
		xdmf_out = df.XDMFFile(self.hdf5file.replace(".hdf5", "_pview.xdmf"))
		with df.HDF5File(df.MPI.comm_world, self.hdf5file, "r") as temp_output:
			for t in ordered_times:
				temp_output.read(mechsolver.d, "displacement/{}".format(t))
				xdmf_out.write(mechsolver.d, t)

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

def make_solution_recorders(outdir):
	labels = ["time (ms)", "lv pressure (kPa)", "rv pressure (kPa)", "contraction", "lv volume (ml)", "rv volume (ml)"]

	simpleheartlogger.log("\n" +" ".join(labels))
	
	trace_recorder = TraceRecorder(outdir, labels)
	disp_recorder = DisplacementRecorder(os.path.join(outdir, "disps_est_contraction.hdf5"))
	return trace_recorder, disp_recorder

class TraceRecorder():
	def __init__(self, output_dir, labels):
		self.output_dir = output_dir
		self.labels = labels
		self.time_trace =[]
		self.contraction_trace = []
		self.lv_vol_trace = []
		self.rv_vol_trace = []
		self.lv_p_trace= []
		self.rv_p_trace = []		

	def update(self, t, contraction, lv_vol, rv_vol, lv_p, rv_p):
		self.time_trace.append(t)
		self.contraction_trace.append(contraction)
		self.lv_vol_trace.append(lv_vol)
		self.rv_vol_trace.append(rv_vol)
		self.lv_p_trace.append(lv_p)
		self.rv_p_trace.append(rv_p)

	def save(self):
		if df.MPI.rank(df.MPI.comm_world) == 0: 
			data = pd.DataFrame(np.vstack([self.time_trace,
										  self.contraction_trace, 
										  self.lv_vol_trace,
										  self.rv_vol_trace,
										  self.lv_p_trace,
										  self.rv_p_trace]).T,
										  columns = ["time (ms)",
													 "contraction",
													 "lv volume (ml)",
													 "rv volume (ml)",
													 "lv pressure (kPa)",
												 	 "rv pressure (kPa)"])
			
			data = data.sort_values(by = "time (ms)")

			data.to_csv(os.path.join(self.output_dir, "mechdata_traces.csv"), index = False)
			self.plot(data)

	def plot(self, data):
		self.plot_paired_traces(data["time (ms)"],
								data["lv volume (ml)"],
								data["lv pressure (kPa)"],
								outfile = os.path.join(self.output_dir, "LV_PV.png"))

		self.plot_paired_traces(data["time (ms)"],
								data["rv volume (ml)"],
								data["rv pressure (kPa)"],
								outfile = os.path.join(self.output_dir, "RV_PV.png"))

		self.plot_paired_traces(data["time (ms)"],
								data["contraction"],
								data["lv volume (ml)"],
								outfile = os.path.join(self.output_dir, "LV_contract_V.png"))

		self.plot_xy(data["rv volume (ml)"],
					 data["rv pressure (kPa)"],
					 outfile = os.path.join(self.output_dir, "RV_PV_loop.png"))

		self.plot_xy(data["lv volume (ml)"],
					 data["lv pressure (kPa)"],
					 outfile = os.path.join(self.output_dir, "LV_PV_loop.png"))

	def plot_xy(self, trace1, trace2, outfile = None):
		fig, ax = plt.subplots()

		ax.plot(trace1, trace2)

		ax.set_ylabel(trace2.name)
		ax.set_xlabel(trace1.name)
		if outfile:
			plt.savefig(outfile)
		plt.close()

	def plot_paired_traces(self, t_trace, trace1, trace2, outfile = None):
		fig, ax = plt.subplots()

		color = "g"
		ax.plot(t_trace, 
				trace1,
				label = trace1.name,
				color = color)

		ax.set_ylabel(trace1.name, color = color)
		ax.set_xlabel(t_trace.name)
		ax.tick_params(axis = "y", labelcolor = color)
	
		ax2 = ax.twinx()

		color = "r"
		ax2.plot(t_trace,
				 trace2,
				 label  = trace2.name,
				 color = color)

		ax2.set_ylabel(trace2.name, color = color)
		ax2.tick_params(axis = "y", labelcolor = color)
		plt.tight_layout()
		if outfile:
			plt.savefig(outfile)
		plt.close()

	def print_latest(self):
		t = self.time_trace[-1]
		p_lv_target = self.lv_p_trace[-1]
		p_rv_target = self.rv_p_trace[-1]
		contraction_next = self.contraction_trace[-1]
		lv_vol = self.lv_vol_trace[-1]
		rv_vol = self.rv_vol_trace[-1]
		simpleheartlogger.log(" ".join(["{} = {:.3f}".format(label, num) for num, label in zip([t, p_lv_target, p_rv_target, contraction_next, lv_vol, rv_vol], self.labels)]))

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