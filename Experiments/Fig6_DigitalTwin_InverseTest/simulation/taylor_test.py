#from mpi_utils import mpi_print 
import math
import numpy as np
import dolfin as df
from simpleheart.helpers import mpi_print

def taylor_test_adjoints(evaluator, m_arr, eps_arrays):
	
	mpi_print("-"*80 + "\nTesting the adjoint gradient\n" + "-"*80)
	Im = evaluator(m_arr)
	orders = taylor_test(evaluator, 
						 m_arr,
						 Im,
						 evaluator.gradient(m_arr),
						 eps = eps_arrays["adjgrad"])
	assert ((np.array(orders) - 2.0) > -0.3).all()

	mpi_print("-"*80 + "\nTesting the tangent linear gradient\n" + "-"*80)

	if hasattr(evaluator, "gradient_dot"):
		Im = evaluator(m_arr)
		orders = taylor_test(evaluator, 
							 m_arr,
							 Im,
							 evaluator.gradient_dot,
							 eps = eps_arrays["tlmgrad"])
		assert ((np.array(orders) - 2.0) > -0.3).all()
	
		mpi_print("-"*80 + "\nTesting the functional adjoint gradient dot\n" + "-"*80)
	else:
		mpi_print("No gradient dot function defined. Skipping test")
		
	if hasattr(evaluator, "eval_func_adj"):
		Im = evaluator(m_arr)
	
		Im_adj = evaluator.eval_func_adj(m_arr)
		#graddot = evaluator.adj_gradient_dot(m_arr)
		orders = taylor_test(evaluator.eval_func_adj, 
							 m_arr,
							 Im_adj,
							 evaluator.adj_gradient_dot,
							 eps = eps_arrays["adjgraddot"])

		assert df.near(orders[-1], 2.0, 0.1)
	else:
		mpi_print("No functional adjoint gradient dot function defined. Skipping test")

	mpi_print("-"*80 + "\nTesting the Hessian dot\n" + "-"*80)
	Im = evaluator(m_arr)
	orders = taylor_test(evaluator, 
						 m_arr,
						 Im,
						 evaluator.gradient(m_arr),
						 Hm = evaluator.hessian_dot,
						 eps = eps_arrays["hessdot"])

	assert ((np.array(orders) - 3.0) > -0.3).all()

def taylor_test(I, m, Im, Jm, Hm = None, eps = [0.1, 0.05, 0.025, 0.0125], printgrad = False):	
	mpi_print("\n" + "-"*80 + "\nRunning Taylor Test\n" + "-"*80)
	
	dms =  np.array([m*eps for eps in eps])
	
	#Precompute Hessians because calling I records a different a set of solutions
	if hasattr(Jm, "__call__"):
		Jms = np.array([Jm(m, dm) for dm in dms])
	else:
		Jms = np.array([Jm.dot(dm) for dm in dms])

	if Hm:
		Hms = np.array([0.5*Hm(m, dm).dot(dm) for dm in dms])
	R0s = np.array([I(m + dm) - Im for dm in dms])
	
	R1s = R0s - Jms

	orders0 = convergence_order(R0s)
	orders1 = convergence_order(np.abs(R1s))

	mpi_print("\nTaylor test results. Approximation order = 0")
	convergence_report(orders0, R0s)

	mpi_print("\nTaylor test results. Approximation order = 1")
	convergence_report(orders1, R1s)	
	if Hm:
		R2s = R1s - np.array(Hms)
		orders2 = convergence_order(np.abs(R2s))
		mpi_print("\nTaylor test results. Approximation order = 2")
		convergence_report(orders2, R2s)
		return orders2
	else:
		return orders1

def convergence_report(orders, remainders):
	mpi_print("Remainder    Convergence Rate")
	for r, order in zip(remainders, [0.0] + orders):
		mpi_print("{:.5e} {:.3f}".format(r, order))
	
def convergence_order(errors, base = 2):
	orders = [0.0] * (len(errors)-1)
	for i in range(len(errors)-1):
		try:
			orders[i] = math.log(errors[i]/errors[i+1], base)
		except ZeroDivisionError:
			orders[i] = numpy.nan
	return orders