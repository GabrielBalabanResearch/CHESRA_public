from simpleheart.helpers import gather
import dolfin as df
import yaml

class MaterialParameters(object):
	def __init__(self, mesh, names, values):
		self.names = names
		Re = df.FiniteElement(cell = mesh.ufl_cell(), family = "R", degree = 0)

		self.V_matparams = df.FunctionSpace(mesh, df.MixedElement([Re]*len(names)))
		self.values = df.Function(self.V_matparams)
		#self.values.vector()[:] = values
		self.values = df.interpolate(df.Constant(values), self.V_matparams)

	def __str__(self):
		matparams_dict = self.to_dict()
		return yaml.dump({"Material Parameters": matparams_dict}, default_flow_style = False)
	
	@staticmethod
	def from_dict(mesh, matparams_dict, efuncname):
		paramnames = get_paramnames_in_order(efuncname)
		return MaterialParameters(mesh, paramnames, [matparams_dict[p] for p in paramnames])	
	
	def to_dict(self):
		gatheredvec = gather(self.values.vector())
		matparams_dict = dict(zip(self.names, gatheredvec.astype(float).tolist()))
		return matparams_dict

def get_paramnames_in_order(efuncname):
	if efuncname == "chesra1":
		paramnames = ["p1", "p2", "p3"]
	elif efuncname == "chesra2":
		paramnames = ["p1", "p2", "p3", "p4"]
	elif efuncname == "holzapfel-ogden":
		paramnames = ["a", "b", "a_f", "b_f", "a_s", "b_s", "a_fs", "b_fs"]
	return paramnames

def get_efuncclass(efuncname):
	if efuncname == "chesra1":
		Efunc = Chesra1
	elif efuncname == "chesra2":
		Efunc = Chesra2
	elif efuncname == "holzapfel-ogden":
		Efunc = HolzapfelOgden
	return Efunc

def make_energy_function(mesh, Cbar, e_f, e_s, e_n, efunc_name, matparams_dict):
	#paramnames = get_paramnames_in_order(efunc_name)
	Efunc = get_efuncclass(efunc_name)
	#matparams = MaterialParameters(mesh, paramnames, [matparams_dict[p] for p in paramnames])
	matparams = MaterialParameters.from_dict(mesh, matparams_dict, efunc_name)
	return Efunc(Cbar, e_f, e_s, e_n, matparams), matparams

class StrainEnergyFunction(object):
	def __init__(self, Cbar, e_f, e_s, e_n, matparams):
		self.matparams = matparams

		Cbar2 = Cbar*Cbar
		
		self.I1 = df.tr(Cbar)
		self.I2 = 0.5*(df.tr(Cbar)**2 - df.tr(Cbar2))
		self.I3 = df.det(Cbar)

		self.I4_f = df.dot(e_f, Cbar*e_f)
		self.I4_s = df.dot(e_s, Cbar*e_s)
		self.I4_n = df.dot(e_n, Cbar*e_n)
		
		self.I5_f = df.dot(e_f, Cbar2*e_f)
		self.I5_s = df.dot(e_s, Cbar2*e_s)
		self.I5_n = df.dot(e_n, Cbar2*e_n)

		self.I8_fs = df.dot(e_f, Cbar*e_s)
		self.I8_sn = df.dot(e_s, Cbar*e_n)
		self.I8_nf = df.dot(e_n, Cbar*e_f)

		#Normalized, squared, invariants
		self.nI1 = (self.I1 -3)**2
		self.nI2 = (self.I2 -3)**2

		self.nI4_f = (self.I4_f - 1)**2
		self.nI4_s = (self.I4_s - 1)**2
		self.nI4_n = (self.I4_n - 1)**2
		
		self.nI5_f = (self.I5_f - 1)**2
		self.nI5_s = (self.I5_s - 1)**2
		self.nI5_n = (self.I5_n - 1)**2

		self.nI8_fs = self.I8_fs**2
		self.nI8_sn = self.I8_sn**2
		self.nI8_nf = self.I8_nf**2

class Chesra1(StrainEnergyFunction):
	def __init__(self, Cbar, e_f, e_s, e_n, matparams):
		super(Chesra1, self).__init__(Cbar, e_f, e_s, e_n, matparams)
	
	def energy(self):
		p1,p2,p3 = df.split(self.matparams.values)
		return (p1 + self.nI1)*(p2 + p3*(self.nI8_fs + self.nI5_s))

class Chesra2(StrainEnergyFunction):
	def __init__(self, Cbar, e_f, e_s, e_n, matparams):
		super(Chesra2, self).__init__(Cbar, e_f, e_s, e_n, matparams)
	
	def energy(self):
		p1,p2,p3,p4 = df.split(self.matparams.values)
		return p1*(p2 + self.nI5_f)*(p3 + self.nI1)*(p4 + self.nI5_s)
	
class HolzapfelOgden(StrainEnergyFunction):
	def __init__(self, Cbar, e_f, e_s, e_n, matparams):
		super(HolzapfelOgden, self).__init__(Cbar, e_f, e_s, e_n, matparams)
	
	def energy(self):
		a,b,a_f,b_f,a_s,b_s,a_fs,b_fs = df.split(self.matparams.values)

		pi = self._scaled_exponential(a, b, (self.I1 - 3))
		pi += df.conditional(df.gt(self.I4_f, 1.0), self._scaled_exponential(a_f, b_f, (self.I4_f - 1)**2), 0.0)
		pi += df.conditional(df.gt(self.I4_s, 1.0), self._scaled_exponential(a_s, b_s, (self.I4_s - 1)**2), 0.0)
		pi += self._scaled_exponential(a_fs, b_fs, self.I8_fs**2)
		return pi
	
	def _scaled_exponential(self, a, b, argument):
		return (a/(2.0*b))*(df.exp(b*argument) - 1)