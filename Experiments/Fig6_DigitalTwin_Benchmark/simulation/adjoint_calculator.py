import dolfin as df
from ufl_legacy.algorithms import expand_derivatives as expand
from ufl_legacy import replace

# Taken from dolfin_adjoint.tests_dolfin.manual_hessian
# See http://www.dolfin-adjoint.org/en/latest/
# Work around some UFL bugs -- action(A, x) doesn't like it if A is null. 
ufl_action = df.action
def action(A, x):
	A = expand(A)
	if len(A.integrals()) != 0: # form is not empty:
		return ufl_action(A, x)
	else:
		return A #form is empty, doesn't matter anyway

class AdjointCalculator(object):
	def __init__(self, R, u, bcs, solver_parameters = {}):
		self.adjsystem = AdjSystem(R, u, bcs, solver_parameters)
		self.recorder = Recorder(self.adjsystem)
		
	def add_functional2record(self, Ji, ui, i):
		self.recorder.add_functional2record(Ji, ui, i)

	def addsolve(self, ui, Ji, expressions = [], values = [], constants = []):
		self.recorder.addsolve(ui, Ji, expressions, values, constants)

	def reset_record(self):
		self.recorder = Recorder(self.adjsystem)

	def gradient(self, m):
		DJm = df.Function(m.function_space())
		for ui, Ji in self.recorder.play():
			lmbda = self.adjsystem.adjoint(Ji, ui)
			DJmi = self.adjsystem.gradient(Ji, lmbda, m)
			DJm.vector().axpy(1.0, DJmi.vector())  
		return DJm

	def hessian_dot(self, m, dm):
		Hm = df.Function(dm.function_space())
		for ui, Ji in self.recorder.play():
			lmbda = self.adjsystem.adjoint(Ji, ui)
			tlm = self.adjsystem.tlm(m, dm)
			lmbda2 = self.adjsystem.soa(Ji, m, dm, lmbda, tlm, ui)
			Hmi = self.adjsystem.hessian_dot(Ji, m, dm, lmbda, tlm, lmbda2, ui)
			Hm.vector().axpy(1.0, Hmi.vector())
		return Hm

	def adjointstate(self):
		for ui, Ji in self.recorder.play():
			yield self.adjsystem.adjoint(Ji, ui) 

	def gradient_dot(self, m, dm):
		DJDm_tlm = 0.0
		for ui, Ji in self.recorder.play():
			tlm = self.adjsystem.tlm(m, dm)
			DJDm_tlm += self.adjsystem.gradient_dot(Ji, m, dm, tlm, ui)
		return DJDm_tlm

	def adj_gradient_dot(self, m, dm):
		DJlmbda_Dm_dm = 0.0
		for ui, Ji in self.recorder.play():
			lmbda = self.adjsystem.adjoint(Ji, ui)
			tlm = self.adjsystem.tlm(m, dm)
			lmbda2 = self.adjsystem.soa(Ji, m, dm, lmbda, tlm, ui)
			DJlmbda_Dm_dm += self.adjsystem.adj_gradient_dot(Ji, ui, lmbda, lmbda2, m, dm)
		return DJlmbda_Dm_dm

	def eval_func_adj(self):
		J_adj = 0.0
		for ui, Ji in self.recorder.play():
			lmbda = self.adjsystem.adjoint(Ji, ui)
			J_adj += self.adjsystem.eval_func_adj(Ji, ui, lmbda)
		return J_adj

class Recorder(object):
	def __init__(self, adjsystem):
		self.record = []
		self.adjsystem = adjsystem

	def add_functional2record(self, Ji, ui, i):
		Jic = replace(Ji, {ui:self.record[i][0]})
		self.record[i][1] += Jic

	def addsolve(self, ui, Ji, expressions = [], values = [], constants = []):
		uic = ui.copy(deepcopy = True)
		Jic = replace(Ji, {ui:uic})
		constants_values = [float(c) for c in constants]
		self.record.append([uic, Jic, expressions, values, constants, constants_values])

	def play(self):
		return Player(self.record, self.adjsystem)

class Player(object):
	def __init__(self, record, adjsystem):
		self.record = iter(record)
		self.adjsystem = adjsystem

	def __iter__(self):
		return self

	def __next__(self):
		ui, Ji, expressions, values, constants, constants_values = self.record.__next__()
		for e,v in zip(expressions, values):
			e.t = v
		for c,v in zip(constants, constants_values):
			c.assign(v)

		self.adjsystem.u.assign(ui)
		return ui, Ji

class AdjSystem(object):	
	def __init__(self, R, u, bcs, solver_parameters):
		
		self.R = R
		self.u = u
		self.Vu = u.function_space()
		self.solver_parameters = solver_parameters

		self.dRdu = df.derivative(self.R, u)
		self.dRduT = df.adjoint(self.dRdu)
		self.bcs = [df.DirichletBC(bc) for bc in bcs]
		for bc in self.bcs:
			bc.homogenize()

	def adjoint(self, J, uj):
		F = df.derivative(J, uj)
		lmbda = df.Function(self.Vu)
		df.solve(self.dRduT == F, lmbda, bcs = self.bcs,
				 solver_parameters = self.solver_parameters)
		return lmbda

	def gradient(self, J, lmbda, m):
		dRm = expand(df.derivative(self.R, m))
		dJdm = df.derivative(J, m)
		DJm_vec = df.assemble(-action(df.adjoint(dRm), lmbda) + dJdm)
		return df.Function(m.function_space(), DJm_vec)

	def tlm(self, m, dm):
		"""
		Tangent Linear model
		"""
		t = df.Function(self.Vu)
		dRm = df.derivative(self.R, m, dm)
		df.solve(self.dRdu == -dRm,
				 t,
			 	 self.bcs,
			 	 solver_parameters = self.solver_parameters)
		return t

	def gradient_dot(self, J, m, dm, tlm, uj):
		dJdu = df.derivative(J, uj, tlm) + df.derivative(J, m, dm)
		return df.assemble(dJdu)

	def soa(self, J, m, dm, lmbda, tlm, uj):
		"""
		Second order adjoint model
		"""
		dRdudu = expand(df.derivative(self.dRduT, self.u, tlm))
		dRdudm = expand(df.derivative(self.dRduT, m, dm))

		dJdu = expand(df.derivative(J, uj, df.TestFunction(self.Vu)))
		dJdudm = expand(df.derivative(dJdu, m, dm))
		dJdudu = expand(df.derivative(dJdu, uj, tlm))
		F = -action(dRdudu, lmbda) - action(dRdudm, lmbda) \
			+ dJdudm + dJdudu
		lmbda2 = df.Function(self.Vu)
		df.solve(self.dRduT == F,
			  	 lmbda2,
			  	 bcs = self.bcs,
			  	 solver_parameters = self.solver_parameters)
		return lmbda2

	def hessian_dot(self, J, m, dm, lmbda, tlm, lmbda2, uj):
		Vm = m.function_space()
		dJdm = df.derivative(J, m, df.TestFunction(Vm))
		dRdmT = df.adjoint(expand(df.derivative(self.R, m)))
		dRdmT_lmbda = action(dRdmT, lmbda)

		FH = (-expand(df.derivative(dRdmT_lmbda, self.u, tlm))
			  -expand(df.derivative(dRdmT_lmbda, m, dm))
			  -action(dRdmT, lmbda2) +
			   expand(df.derivative(dJdm, uj, tlm)) +
			   expand(df.derivative(dJdm, m, dm)))

		return df.Function(Vm, df.assemble(FH))

	def adj_gradient_dot(self, Ji, uj, lmbda, lmbda2, m, dm):
		Ji_lmbda = replace(Ji, {uj: lmbda})
		Djadj_Dm = df.derivative(Ji_lmbda, m, dm) + action(df.derivative(Ji_lmbda, lmbda), lmbda2)
		return df.assemble(Djadj_Dm)

	def eval_func_adj(self, Ji, uj, lmbda):
		Ji_lmbda = replace(Ji, {uj: lmbda})
		return df.assemble(Ji_lmbda)

def create_direction_function_from_array(dm, V):
	"""
	Create a direction function for evaluating gradient dot and Hessian dot
	"""
	dm_func = df.Function(V)
	lr = dm_func.vector().local_range()
	local_dm = dm[lr[0]: lr[1]]
	dm_func.vector().set_local(local_dm)
	dm_func.vector().apply("insert")
	return dm_func
