import dolfin as df 
import os
import numpy as np
from matplotlib import pyplot as plt
from simpleheart.helpers import *
from simpleheart.energy_functions import make_energy_function, ActiveHaoEnergy, MaterialParameters
from ufl_legacy.algorithms.compute_form_data import estimate_total_polynomial_degree

class MicroStructure():
	def __init__(self, e_fibre, e_sheet, e_sheetnormal, V_fibre):
		self.e_fibre = e_fibre
		self.e_sheet = e_sheet
		self.e_sheetnormal = e_sheetnormal
		self.V_fibre = V_fibre

	@staticmethod
	def from_hdf5(hdf5_path, biv_mesh, fibre_dbloc, sheet_dbloc = None, sheetnormal_dbloc = None):
		with df.HDF5File(df.MPI.comm_world, hdf5_path, "r") as hdf5:
			
			ve =  df.VectorElement(family = "Quadrature",
							  	   cell = biv_mesh.ufl_cell(),
								   degree = 4,
								   quad_scheme = "default")
			
			V_fibre = df.FunctionSpace(biv_mesh, ve)
			e_fibre = df.Function(V_fibre)
			hdf5.read(e_fibre, fibre_dbloc)
			e_fibre = normalize_vectorfunction(e_fibre)

			if sheet_dbloc is not None and sheet_dbloc != "":
				e_sheet = df.Function(V_fibre)
				hdf5.read(e_sheet, sheet_dbloc)
				e_sheet = normalize_vectorfunction(e_sheet)
			else:
				e_sheet = None
		
			if sheetnormal_dbloc is not None and sheetnormal_dbloc != "":
				e_sheetnormal = df.Function(V_fibre)
				hdf5.read(e_sheetnormal, sheetnormal_dbloc)
				e_sheetnormal = normalize_vectorfunction(e_sheetnormal)
			else:
				e_sheetnormal = None
		
		return MicroStructure(e_fibre, e_sheet, e_sheetnormal, V_fibre)

	def to_hdf5(self, hdf5_path):
		if not os.path.exists(os.path.dirname(hdf5_path)):
			os.makedirs(os.path.dirname(hdf5_path))

		with df.HDF5File(df.MPI.comm_world, hdf5_path, "w") as test_hdf5_df:
			test_hdf5_df.write(self.e_fibre, "e_fibre")
			if self.e_sheet is not None:
				test_hdf5_df.write(self.e_sheet, "e_sheet")
			if self.e_sheetnormal is not None:
				test_hdf5_df.write(self.e_sheetnormal, "e_sheetnormal")

	def to_xdmf(self, xdmf_path):
		if not os.path.exists(xdmf_path):
			os.makedirs(xdmf_path)

		form_compiler_parameters = {"representation": "quadrature"}
		Vout = df.VectorFunctionSpace(self.e_fibre.function_space().mesh(), "CG", 1)

		fibres_outpath = os.path.join(xdmf_path, "fibres.xdmf")
		e_f_out = df.Function(Vout)

		dx_quad = df.dx(metadata = {"quadrature_degree": estimate_total_polynomial_degree(self.e_fibre)})
		df.solve(df.inner(df.TrialFunction(Vout), df.TestFunction(Vout))*dx_quad == df.inner(df.TestFunction(Vout), self.e_fibre)*dx_quad, e_f_out,
						  form_compiler_parameters = form_compiler_parameters)

		e_f_out = normalize_vectorfunction(e_f_out)
		e_f_xdmf = df.XDMFFile(fibres_outpath)
		e_f_xdmf.write(e_f_out)

		if self.e_sheet is not None:
			sheets_outpath = os.path.join(xdmf_path, "sheets.xdmf")
			e_s_out = df.Function(Vout)
			df.solve(df.inner(df.TrialFunction(Vout), df.TestFunction(Vout))*dx_quad == df.inner(df.TestFunction(Vout), self.e_sheet)*dx_quad, e_s_out,
					 form_compiler_parameters = form_compiler_parameters)
			e_s_out = normalize_vectorfunction(e_s_out)
			e_s_xdmf = df.XDMFFile(sheets_outpath)
			e_s_xdmf.write(e_s_out)

		if self.e_sheetnormal is not None:
			sheetnormals_outpath = os.path.join(xdmf_path, "sheetnormals.xdmf")
			e_n_out = df.Function(Vout)
			df.solve(df.inner(df.TrialFunction(Vout), df.TestFunction(Vout))*dx_quad == df.inner(df.TestFunction(Vout), self.e_sheetnormal)*dx_quad, e_n_out,
					 form_compiler_parameters = form_compiler_parameters)
			e_n_out = normalize_vectorfunction(e_n_out)

			e_n_xdmf = df.XDMFFile(sheetnormals_outpath)
			e_n_xdmf.write(e_n_out)

class BIVGeo():
	"""
	Biventricular mesh geometry and associated data. 
	Spatial coordinates are in cm.
	"""
	def __init__(self,
				 biv_mesh, 
				 mesh_markers,
				 facet_markers,
				 microstructure,
				 simpleheart_config):
			
			self.mesh = biv_mesh
			self.mesh_markers = mesh_markers
			self.facet_markers = facet_markers
			self.microstructure = microstructure

			self.mesh_marker_definitions = simpleheart_config["mesh"]["mesh_markers"]
			self.facet_marker_definitions = simpleheart_config["mesh"]["facet_markers"]

			self.simpleheart_config = simpleheart_config
	
	@staticmethod
	def from_hdf5(simpleheart_config):
		with df.HDF5File(df.MPI.comm_world, simpleheart_config["mesh"]["path"], "r") as hdf5:
			biv_mesh = df.Mesh()
			hdf5.read(biv_mesh, "/geometry/mesh", False)

			mesh_markers = df.MeshFunction("size_t", biv_mesh, 3)
			hdf5.read(mesh_markers, "/geometry/mesh/meshfunction_3")
	
			facet_markers = df.MeshFunction("size_t", biv_mesh, 2)
			hdf5.read(facet_markers, "/geometry/mesh/meshfunction_2")

		microstructure = MicroStructure.from_hdf5(simpleheart_config["mesh"]["microstructure"],
												  biv_mesh,
												  simpleheart_config["mesh"]["dbpath_fibre"],
												  simpleheart_config["mesh"].get("dbpath_sheet"),
												  simpleheart_config["mesh"].get("dbpath_sheetnormal"))
		
		return BIVGeo(biv_mesh, mesh_markers, facet_markers, microstructure, simpleheart_config)

	def to_hdf5(self, hdf5_path):
		if not os.path.exists(os.path.dirname(hdf5_path)) and df.MPI.rank(df.MPI.comm_world) == 0:
			os.makedirs(os.path.dirname(hdf5_path))

		with df.HDF5File(df.MPI.comm_world, hdf5_path, "w") as hdf5:
			hdf5.write(self.mesh, "/geometry/mesh")
			hdf5.write(self.mesh_markers, "/geometry/mesh/meshfunction_3")
			hdf5.write(self.facet_markers, "/geometry/mesh/meshfunction_2")

		self.microstructure.to_hdf5(hdf5_path.rstrip(".h5").rstrip(".hdf5") + "_microstructure.h5")

	def clone(self):
		new_mesh = df.Mesh(self.mesh)

		# Step 1 Copy the microstructure
		micro_elem = self.microstructure.e_fibre.ufl_element()
		ve = df.VectorElement(micro_elem.family(), new_mesh.ufl_cell(), degree = micro_elem.degree(), quad_scheme = "default")
		V_fibre_new = df.FunctionSpace(new_mesh, ve)
		e_f_new = df.Function(V_fibre_new)
		e_s_new  = df.Function(V_fibre_new)

		e_f_new.vector()[:] = self.microstructure.e_fibre.vector()[:]
		e_s_new.vector()[:] = self.microstructure.e_sheet.vector()[:]
		if self.microstructure.e_sheetnormal is not None:
			e_n_new = df.Function(V_fibre_new)
			e_n_new.vector()[:] = self.microstructure.e_sheetnormal.vector()[:]
		else:
			e_n_new = None
		new_microstructure = MicroStructure(e_f_new, e_s_new, e_n_new, V_fibre_new)

		#Step 2: Copy Mesh Markers
		copied_mesh_markers = df.MeshFunction("size_t", new_mesh, self.mesh_markers.dim())
		copied_mesh_markers.array()[:] = self.mesh_markers.array()

		#Step 3: Copy Facet Markers
		copied_facet_markers = df.MeshFunction("size_t", new_mesh, self.facet_markers.dim())
		copied_facet_markers.array()[:] = self.facet_markers.array()

		return BIVGeo(new_mesh,
					  copied_mesh_markers,
					  copied_facet_markers,
					  new_microstructure,
					  self.simpleheart_config)

class MechanicsSolver():
	def __init__(self, bivgeo, simpleheart_config):
		self.bivgeo = bivgeo
		self.simpleheart_config = simpleheart_config

		self.solver_params = simpleheart_config["numerics"]["nonlinear_solver_params"]

		#LV-RV pressure variables 
		self.p_lv = df.Constant(0.0, name = "lv pressure")
		self.p_rv = df.Constant(0.0, name = "rv pressure")

		self.d, self.Dpi, self.DDpi, self.Vol_lv, self.Vol_rv, self.matparams = self.get_variational_forms()

		#No long axis (x-direction ) movement at base
		#Spring condition along short axes at base
		self.dirichlet_BC = df.DirichletBC(self.d.function_space().sub(0), 
										   0,
										   self.bivgeo.facet_markers,
										   self.bivgeo.facet_marker_definitions["base"])
	def reinit(self):
		return MechanicsSolver(self.bivgeo, self.simpleheart_config)

	def get_variational_forms(self):
		"""
		Implements the principle of stationary potential energy. 
		Chapter 8.3, Holzapfel, Nonlinear solid mechanics. 
		One field formulation with weak enforcement of 
		incompressibility.
		"""

		#Mechanical variables
		V = df.VectorFunctionSpace(self.bivgeo.mesh, "CG", self.simpleheart_config["numerics"]["FEM_degree"])
		d = df.Function(V, name = "displacement")
		dtest = df.TestFunction(V)

		F = df.variable(df.grad(d) + df.Identity(3))
		J = df.det(F)
		e_f = self.bivgeo.microstructure.e_fibre
		e_s = self.bivgeo.microstructure.e_sheet
		e_n = self.bivgeo.microstructure.e_sheetnormal
		
		if e_n is None:
			e_n = cross_vectorfunctions(e_f, e_s, self.bivgeo.microstructure.V_fibre)

		N = df.FacetNormal(self.bivgeo.mesh)
		C = J**(-2.0/3.0)*F.T*F
		
		efunc, matparams = make_energy_function(self.bivgeo.mesh, 
										  		 C,
												 e_f,
												 e_s,
												 e_n,
												 self.simpleheart_config["energy_function"],
												 self.simpleheart_config["material_parameters"])

		pi_int = efunc.energy()
		#######################################################################################
		
		pi_compress =  df.Constant(self.simpleheart_config["numerics"]["compress_penalty"])*0.5*(J - 1)**2

		pi_tot = pi_int + pi_compress

		dx = df.dx(metadata = {"quadrature_degree": estimate_total_polynomial_degree(e_f)})

		Dpi = df.derivative(pi_tot*dx, d, dtest)

		#Pressure loading conditions on LV-RV endocardium
		ds = df.Measure("exterior_facet", subdomain_data = self.bivgeo.facet_markers)

		endo_rv_marker = self.bivgeo.facet_marker_definitions["RV_endo"]
		endo_lv_marker = self.bivgeo.facet_marker_definitions["LV_endo"]
		base_marker = self.bivgeo.facet_marker_definitions["base"]
		
		Dpi_lv = df.inner(J*self.p_rv*df.dot(df.inv(F).T, N), dtest)*ds(endo_rv_marker)
		Dpi_rv = df.inner(J*self.p_lv*df.dot(df.inv(F).T, N), dtest)*ds(endo_lv_marker)
		Dpi += Dpi_lv + Dpi_rv

		#Robin boundary condition at base
		k_base = df.Constant(self.simpleheart_config["numerics"]["base_spring_k"], name = "base spring")
		Dpi += df.inner(k_base*d, dtest)*ds(base_marker)
		DDpi = df.derivative(Dpi, d)

		#Cavity volume forms
		X = df.SpatialCoordinate(self.bivgeo.mesh)
		volform = (-1.0/3.0)*df.dot((X+d), J*df.inv(F).T*N)

		Vol_lv = volform*ds(endo_lv_marker)
		Vol_rv = volform*ds(endo_rv_marker)
		
		return d, Dpi, DDpi, Vol_lv, Vol_rv, matparams

	def _scaled_exponential(self, a, b, argument):
		return (a/(2.0*b))*(df.exp(b*argument) - 1)

	def minimize_energy(self, p_lv, p_rv):
		"""
		When Dpi = 0 the mechanical energy is minimized
		"""
		self.p_lv.assign(p_lv)
		self.p_rv.assign(p_rv)
		#self.d.vector()[:] = 0.0
		df.solve(self.Dpi == 0,
				 self.d,
				 self.dirichlet_BC,
				 J = self.DDpi,
				 solver_parameters = self.solver_params)

	def continuity_solve(self, 
						 p_lv_next,
						 p_rv_next,
						 p_lv_inc = 0.05,
						 p_rv_inc = 0.05):
		"""
		Breaks up a change in contraction and pressure into small steps.
		Reduces step size in the case of nonlinear solver nonconvergence.
		"""
		simpleheartlogger.increase_tab()

		report = self.simpleheart_config["output"]["screen"]["continuity_solver"]
		
		if report:
			simpleheartlogger.log("Continuity solve lvp = {:.2f} rvp = {:.2f}".format(p_lv_next, p_rv_next))
		if (np.array([p_lv_inc, p_rv_inc]) < 1.e-12).any():
			raise Exception("Continuity solver failed. Pressure-contraction increment too small.")

		steps_plv = np.ceil(np.abs(p_lv_next - float(self.p_lv))/p_lv_inc)
		steps_prv = np.ceil(np.abs(p_rv_next - float(self.p_rv))/p_rv_inc)

		num_steps = int(np.max([steps_plv, steps_prv]))

		p_lv_levels = np.linspace(float(self.p_lv), p_lv_next, num_steps)[1:]
		p_rv_levels = np.linspace(float(self.p_rv), p_rv_next, num_steps)[1:]

		for p_lv, p_rv in zip(p_lv_levels, p_rv_levels):
			simpleheartlogger.increase_tab()
			if report:
				simpleheartlogger.log("Increment lvp = {:.2f} rvp = {:.2f}".format(p_lv, p_rv))
			p_lv_curr = float(self.p_lv)
			p_rv_curr = float(self.p_rv)
			dispvec_curr = self.d.vector().copy()

			try:
				self.minimize_energy(p_lv, p_rv)
			except Exception as Ex:
				simpleheartlogger.log(Ex)
				simpleheartlogger.log("Nonlinear solver failed, reducing pressure-contraction increments.")
				self.p_lv.assign(p_lv_curr)
				self.p_rv.assign(p_rv_curr)
				self.d.vector()[:] = dispvec_curr
				
				self.continuity_solve(p_lv_next,
									  p_rv_next,
									  p_lv_inc = p_lv_inc/2,
									  p_rv_inc = p_rv_inc/2)
			simpleheartlogger.decrease_tab()
		simpleheartlogger.decrease_tab()

	def get_cavity_volumes(self):
		return df.assemble(self.Vol_lv), df.assemble(self.Vol_rv)