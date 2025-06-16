import dolfin as df
import pandas as pd
import numpy as np
import os
import time

from .motion_simulation import BIVGeo, MechanicsSolver
from .helpers import simpleheartlogger, gather
from .energy_functions import get_paramnames_in_order
from .adjoint_calculator import AdjointCalculator, create_direction_function_from_array
from .taylor_test import taylor_test_adjoints
from scipy.optimize import minimize

class HeartMechDataLoader(object):
    def __init__(self, displacement_db_path, mechdata_path, mesh, elem_degree = 1):
        self.displacement_db_path = displacement_db_path
        self.V = df.VectorFunctionSpace(mesh, "CG", elem_degree)
        self.mechdata_df = pd.read_csv(mechdata_path)

    def __call__(self, t):
        d = df.Function(self.V)
        dispfile = df.HDF5File(df.MPI.comm_world, self.displacement_db_path, "r")
        
        dispfile.read(d, "displacement/{}".format(t))
        mechdata_t = self.mechdata_df[self.mechdata_df["time (ms)"]  == t]
        return d, mechdata_t
    
class DisplacementSquareLoss(object):
    def __init__(self, dataloader, mechsolver, printlosses = False):
        self.dataloader = dataloader
        self.mechsolver = mechsolver
        self.printlosses = printlosses
        
    def __call__(self, matparamvals):
        self.mechsolver = self.mechsolver.reinit()
        
        self.mechsolver.matparams.values.assign(df.Constant(matparamvals))
                
        df.MPI.comm_world.Barrier()  # Ensure all processes reach this point before proceeding
    
        if self.printlosses:
            simpleheartlogger.log("Evaluating displacement square loss")
            simpleheartlogger.log(self.mechsolver.matparams)

        adjoint_solverparams = {"linear_solver": self.mechsolver.solver_params["snes_solver"]["linear_solver"],
                                "lu_solver": self.mechsolver.solver_params["snes_solver"]["lu_solver"]}
        
        self.adjcalc = AdjointCalculator(self.mechsolver.Dpi,
                                         self.mechsolver.d,
                                        [self.mechsolver.dirichlet_BC],
                                         adjoint_solverparams)
        
        times = self.dataloader.mechdata_df["time (ms)"]
        total_loss = []
        for t in times:
            disp_target, mechdata = self.dataloader(float(t))
            self.mechsolver.continuity_solve(float(mechdata["lv pressure (kPa)"].iloc[0]),
									         float(mechdata["rv pressure (kPa)"].iloc[0]),
                                             0)
            
            loss_t_func = (disp_target - self.mechsolver.d)**2*df.dx
            
            mechconstants = [self.mechsolver.p_lv, self.mechsolver.p_rv, self.mechsolver.contraction]
            self.adjcalc.addsolve(self.mechsolver.d, loss_t_func, constants = mechconstants)

            loss_t = df.assemble(loss_t_func)
            total_loss.append(loss_t)
            if self.printlosses:
                simpleheartlogger.log("t = {:.2f} loss = {:.2f}".format(t, loss_t))
            else:
                simpleheartlogger.log("t = {:.2f}".format(t))
        total_loss = sum(total_loss) 

        if self.printlosses:
                simpleheartlogger.log("Total loss = {:.2f}".format(total_loss))
        return total_loss

    def _reevaluate_if_new_matparams(self, matparamvals):
        saved_matparamfunc = self.mechsolver.matparams.values
        if not np.isclose(matparamvals, gather(saved_matparamfunc.vector())).all():
            self(matparamvals)

    def gradient(self, m):
        self._reevaluate_if_new_matparams(m)
        if self.printlosses:
            simpleheartlogger.log("Evaluating functional gradient")
            simpleheartlogger.log(self.mechsolver.matparams)
        grad = gather(self.adjcalc.gradient(self.mechsolver.matparams.values).vector())
        if self.printlosses:
            simpleheartlogger.log(f"Functional gradient = {grad}")
        return grad

    def gradient_dot(self, m, dm):
        self._reevaluate_if_new_matparams(m)
        dmfunc = create_direction_function_from_array(dm, self.mechsolver.matparams.values.function_space())
        return self.adjcalc.gradient_dot(self.mechsolver.matparams.values, dmfunc)

    def eval_func_adj(self, m):
        self._reevaluate_if_new_matparams(m)
        return self.adjcalc.eval_func_adj()

    def adj_gradient_dot(self, m, dm):
        self._reevaluate_if_new_matparams(m)
        dm_func = create_direction_function_from_array(dm, self.mechsolver.matparams.values.function_space())
        return self.adjcalc.adj_gradient_dot(self.mechsolver.matparams.values, dm_func)

    def hessian_dot(self, m, dm):
        self._reevaluate_if_new_matparams(m)
        if self.printlosses:
            simpleheartlogger.log("Evaluating hessian dot")
            simpleheartlogger.log(self.mechsolver.matparams)
        dm_func = create_direction_function_from_array(dm, self.mechsolver.matparams.values.function_space())
        return gather(self.adjcalc.hessian_dot(self.mechsolver.matparams.values, dm_func).vector())

def estimate_elasticity_displacementloss(simpleheart_config):
    bivgeo = BIVGeo.from_hdf5(simpleheart_config)
    mechsolver = MechanicsSolver(bivgeo, simpleheart_config)

    dataloader = HeartMechDataLoader(simpleheart_config["elasticity_estimation"]["displacement_db"],
                                     simpleheart_config["mechdata"],
                                     bivgeo.mesh,
                                     simpleheart_config["numerics"]["FEM_degree"])
    
    squareloss = DisplacementSquareLoss(dataloader,
                                        mechsolver,
                                        simpleheart_config["elasticity_estimation"]["printlosses"])
    
    matparamnames = get_paramnames_in_order(simpleheart_config["energy_function"])
    ini_guess = np.array([simpleheart_config["material_parameters"][mpname] for mpname in matparamnames])

    #########################################################################################
    # Taylor test for adjoint gradient and hessian-vector products
    #########################################################################################
    #eps_array_base = 2**-np.arange(3,6).astype(float) #np.array([0.125, 0.0625, 0.03125])
    #eps_arrays = {"adjgrad": eps_array_base,
	#		      "tlmgrad": eps_array_base,
	#			  "adjgraddot": eps_array_base,
	#			  "hessdot": eps_array_base}
    
    #taylor_test_adjoints(squareloss, ini_guess, eps_arrays)
    #exit()
    #########################################################################################

    t0 = time.time()

    opt_method = simpleheart_config["elasticity_estimation"]["method"]

    simpleheartlogger.log("-"*80 + f"\nStarting Optimization with method {opt_method}\n" + "-"*80)
    
    opt_result = minimize(squareloss,
                          ini_guess,
                          method = opt_method,
                          jac = squareloss.gradient,
                          #hessp = squareloss.hessian_dot,
                          bounds = [[lb, None] for lb in simpleheart_config["elasticity_estimation"]["lower_bound"]],
                          callback = lambda x: simpleheartlogger.log(x),
                          tol = simpleheart_config["elasticity_estimation"]["optimization_tol"])
    
    runtime = time.time() - t0
    simpleheartlogger.log("Finished estimating material parameters run-time = {:.2f} seconds".format(runtime))
    #print_opt_iteration(opt_result)
    simpleheartlogger.log(opt_result)

    output_optresults(opt_result,
                      runtime,
                      simpleheart_config["output"]["files"]["path"])

def output_optresults(opt_result, runtime, output_path):
    output_results_path = os.path.join(output_path,
                                       "optresults.txt")
    
    if df.MPI.rank(df.MPI.comm_world) == 0:
        if not os.path.exists(os.path.dirname(output_results_path)):
            os.makedirs(os.path.dirname(output_results_path))

        with open(output_results_path, "w") as f:
            f.write(str(opt_result))
            f.write("\nRun-time: {:.2f} seconds.".format(runtime))
    simpleheartlogger.log("Wrote results to {}".format(output_results_path))
